"""MLX-native flat safetensors bundle load + optional key mapping."""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_unflatten

from backend.engine.common.bundle_weight_mapping import WeightMapper
from backend.engine.common.bundle_weights._cache import DQ_WEIGHT_DL_CACHE
from backend.engine.common.bundle_weights.definitions import ComponentDefinition
from backend.engine.common.bundle_weights.loaded_weights import LoadedWeights, MetaData
from backend.engine.common.bundle_weights.resolution.path_resolution import PathResolution

logger = logging.getLogger(__name__)


class WeightLoader:
    @staticmethod
    def load(
        weight_definition: type,
        model_path: str | None = None,
    ) -> LoadedWeights:
        root_path = PathResolution.resolve(
            path=model_path,
            patterns=weight_definition.get_download_patterns(),
        )

        components: dict[str, dict] = {}
        quantization_level = None
        bundle_format_version = None
        raw_weights_cache: dict[tuple, dict] = {}

        for component in weight_definition.get_components():
            weights, q_level, version = WeightLoader._load_component(root_path, component, raw_weights_cache)
            components[component.name] = weights

            if quantization_level is None and q_level is not None:
                quantization_level = q_level
                bundle_format_version = version

        return LoadedWeights(
            components=components,
            meta_data=MetaData(
                quantization_level=quantization_level,
                bundle_format_version=bundle_format_version,
            ),
        )

    @staticmethod
    def _load_component(
        root_path: Path | None,
        component: ComponentDefinition,
        raw_weights_cache: dict[tuple, dict] | None = None,
    ) -> tuple[dict, int | None, str | None]:
        if component.download_url is not None:
            file_path = WeightLoader._download_from_url(component.download_url, component.name)
            raw_weights = WeightLoader._load_weights_file(file_path, component.loading_mode)
        else:
            if root_path is None:
                raise ValueError(f"No root_path and no download_url for component: {component.name}")
            component_path = root_path / component.hf_subdir

            weights, q_level, version = WeightLoader._try_load_flat_packaged_format(component_path)
            if weights is not None:
                return weights, q_level, version

            cache_key = (str(component_path), component.loading_mode, tuple(component.weight_files or []))
            if raw_weights_cache is not None and cache_key in raw_weights_cache:
                raw_weights = raw_weights_cache[cache_key]
            else:
                raw_weights = WeightLoader._load_safetensors(
                    component_path, component.loading_mode, component.weight_files
                )
                if raw_weights_cache is not None:
                    raw_weights_cache[cache_key] = raw_weights

        if component.weight_prefix_filters is not None:
            raw_weights = {
                k: v
                for k, v in raw_weights.items()
                if any(k.startswith(prefix) for prefix in component.weight_prefix_filters)
            }

        if component.precision is not None:
            raw_weights = WeightLoader._convert_precision(raw_weights, component.precision)

        if component.mapping_getter is None:
            if component.bulk_transform is not None:
                raw_weights = {k: component.bulk_transform(v) for k, v in raw_weights.items()}
            return tree_unflatten(list(raw_weights.items())), None, None

        mapped_weights = WeightMapper.apply_mapping(
            hf_weights=raw_weights,
            mapping=component.mapping_getter(),
            num_blocks=component.num_blocks,
            num_layers=component.num_layers,
        )
        return mapped_weights, None, None

    @staticmethod
    def _try_load_flat_packaged_format(path: Path) -> tuple[dict | None, int | None, str | None]:
        if not path.exists():
            return None, None, None

        shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
        if not shard_files:
            return None, None, None

        data = mx.load(str(shard_files[0]), return_metadata=True)
        if len(data) <= 1:
            return None, None, None

        meta = data[1]
        quantization_level_str = meta.get("quantization_level")
        _legacy_pkg_ver_key = "".join(("m", "flux", "_version"))
        bundle_version = meta.get(_legacy_pkg_ver_key) or meta.get("bundle_format_version")

        if quantization_level_str is None and bundle_version is None:
            return None, None, None

        if quantization_level_str in (None, "None", "null", ""):
            quantization_level = None
        else:
            quantization_level = int(quantization_level_str)

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            shard_data = mx.load(str(shard), return_metadata=True)
            all_weights.update(dict(shard_data[0].items()))

        unflattened = tree_unflatten(list(all_weights.items()))
        return unflattened, quantization_level, bundle_version

    @staticmethod
    def _download_from_url(url: str, component_name: str) -> Path:
        cache_dir = DQ_WEIGHT_DL_CACHE / component_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        filename = url.split("/")[-1]
        file_path = cache_dir / filename

        if not file_path.exists():
            logger.info("Downloading %s weights from %s...", component_name, url)
            try:
                urllib.request.urlretrieve(url, file_path)
                logger.info("Downloaded to %s", file_path)
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                logger.error("Failed to download: %s", e)
                raise FileNotFoundError(f"Model file not found at {file_path}") from e

        return file_path

    @staticmethod
    def _load_weights_file(file_path: Path, loading_mode: str) -> dict[str, mx.array]:
        if loading_mode in ("mlx_native", "single"):
            data = mx.load(str(file_path), return_metadata=True)
            return dict(data[0].items())
        raise ValueError(f"Unsupported loading mode for bundle loader: {loading_mode!r}")

    @staticmethod
    def _load_safetensors(path: Path, loading_mode: str, weight_files: list[str] | None = None) -> dict[str, mx.array]:
        if loading_mode == "mlx_native":
            return WeightLoader._load_mlx_native(path, weight_files)
        if loading_mode == "multi_json":
            return WeightLoader._load_multi_json(path)
        if loading_mode == "single":
            return WeightLoader._load_single(path)
        if loading_mode == "multi_glob":
            return WeightLoader._load_multi_glob(path)
        raise ValueError(f"Unknown loading mode: {loading_mode}")

    @staticmethod
    def _load_mlx_native(path: Path, weight_files: list[str] | None = None) -> dict[str, mx.array]:
        if weight_files:
            missing = [f for f in weight_files if not (path / f).exists()]
            if missing:
                raise FileNotFoundError(f"Missing specified weight files in {path}: {missing}")
            shard_files = [path / f for f in weight_files]
        else:
            shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
            if not shard_files:
                raise FileNotFoundError(f"No safetensors files found in {path}")

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            weights = mx.load(str(shard))
            all_weights.update(weights)

        return all_weights

    @staticmethod
    def _load_multi_json(path: Path) -> dict[str, mx.array]:
        index_path = path / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)

        files_to_load: dict[str, list[str]] = {}
        for param_name, file_name in index["weight_map"].items():
            files_to_load.setdefault(file_name, []).append(param_name)

        all_weights: dict[str, mx.array] = {}
        for file_name, param_names in files_to_load.items():
            file_path = path / file_name
            file_weights = mx.load(str(file_path))
            for param_name in param_names:
                if param_name in file_weights:
                    all_weights[param_name] = file_weights[param_name]

        return all_weights

    @staticmethod
    def _load_single(path: Path) -> dict[str, mx.array]:
        safetensors_files = [f for f in path.glob("*.safetensors") if not f.name.startswith("._")]
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")

        weights_file = safetensors_files[0]
        data = mx.load(str(weights_file), return_metadata=True)
        return dict(data[0].items())

    @staticmethod
    def _load_multi_glob(path: Path) -> dict[str, mx.array]:
        shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            data, _ = mx.load(str(shard), return_metadata=True)
            all_weights.update(dict(data.items()))

        return all_weights

    @staticmethod
    def _convert_precision(weights: dict[str, mx.array], precision: mx.Dtype) -> dict[str, mx.array]:
        return {k: v if v.dtype == precision else v.astype(precision) for k, v in weights.items()}

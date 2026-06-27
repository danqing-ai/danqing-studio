"""Helpers ported from HiDream-O1 models/pipeline.py + models/utils.py."""
from __future__ import annotations

import math
from typing import Sequence
import numpy as np

PATCH_SIZE = 32
TIMESTEP_TOKEN_NUM = 1
NOISE_SCALE_DEFAULT = 7.5
T_EPS = 0.001
TMS_TOKEN_ID = 151673   # from qwen3_vl_transformers.py — Qwen3VLModel.tms_token_id
CONDITION_IMAGE_SIZE = 384   # vision-tower-side size for reference images

PREDEFINED_RESOLUTIONS = [
    (2048, 2048),
    (2304, 1728), (1728, 2304),
    (2560, 1440), (1440, 2560),
    (2496, 1664), (1664, 2496),
    (3104, 1312), (1312, 3104),
    (2304, 1792), (1792, 2304),
]


def find_closest_resolution(width: int, height: int) -> tuple[int, int]:
    img_ratio = width / height
    best, min_diff = None, float("inf")
    for w, h in PREDEFINED_RESOLUTIONS:
        diff = abs(w / h - img_ratio)
        if diff < min_diff:
            min_diff, best = diff, (w, h)
    return best


def patchify(img_chw: np.ndarray, patch: int = PATCH_SIZE) -> np.ndarray:
    C, H, W = img_chw.shape
    assert H % patch == 0 and W % patch == 0
    x = img_chw.reshape(C, H // patch, patch, W // patch, patch)
    x = np.transpose(x, (1, 3, 0, 2, 4))
    return x.reshape(H // patch * W // patch, C * patch * patch)


def unpatchify(patches_nd, h_patches, w_patches, patch=PATCH_SIZE, channels=3):
    x = patches_nd.reshape(h_patches, w_patches, channels, patch, patch)
    x = np.transpose(x, (2, 0, 3, 1, 4))
    return x.reshape(channels, h_patches * patch, w_patches * patch)


def build_t2i_text_sample(prompt, height, width, tokenizer, processor, model_config):
    image_token_id = model_config.image_token_id
    video_token_id = model_config.video_token_id
    vision_start_token_id = model_config.vision_start_token_id

    image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)
    boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
    tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")

    messages = [{"role": "user", "content": prompt}]
    template_caption = (
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        + boi_token + tms_token * TIMESTEP_TOKEN_NUM
    )
    input_ids = np.asarray(
        tokenizer.encode(template_caption, add_special_tokens=False),
        dtype=np.int64,
    ).reshape(1, -1)

    image_grid_thw = np.asarray(
        [[1, height // PATCH_SIZE, width // PATCH_SIZE]], dtype=np.int64
    )

    vision_tokens = np.full((1, image_len), image_token_id, dtype=input_ids.dtype)
    vision_tokens[0, 0] = vision_start_token_id
    input_ids_pad = np.concatenate([input_ids, vision_tokens], axis=-1)

    position_ids, _ = get_rope_index_fix_point(
        spatial_merge_size=1,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
        input_ids=input_ids_pad,
        image_grid_thw=image_grid_thw,
        skip_vision_start_token=[1],
    )

    txt_seq_len = input_ids.shape[-1]
    all_seq_len = position_ids.shape[-1]

    token_types = np.zeros((1, all_seq_len), dtype=np.int64)
    bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
    token_types[0, bgn: bgn + image_len + TIMESTEP_TOKEN_NUM] = 1
    # Tag the tms positions distinctly so vinput_mask excludes them — they're
    # for the timestep embedding, not actual image patches.
    token_types[0, txt_seq_len - TIMESTEP_TOKEN_NUM: txt_seq_len] = 3
    vinput_mask = (token_types == 1)
    token_types_bin = (token_types > 0).astype(np.int64)

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "token_types": token_types_bin,
        "vinput_mask": vinput_mask,
    }


def get_rope_index_fix_point(
    spatial_merge_size, image_token_id, video_token_id, vision_start_token_id,
    input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None,
    skip_vision_start_token=None, fix_point=4096,
):
    if input_ids is None:
        raise ValueError("input_ids is required")
    if attention_mask is None:
        attention_mask = np.ones_like(input_ids)

    B, S = input_ids.shape
    position_ids = np.ones((3, B, S), dtype=input_ids.dtype)

    image_index = 0
    video_index = 0
    mrope_position_deltas: list[int] = []

    for i in range(B):
        ids_i = input_ids[i][attention_mask[i] == 1]
        vision_start_indices = np.argwhere(ids_i == vision_start_token_id).reshape(-1)
        vision_tokens = ids_i[vision_start_indices + 1] if len(vision_start_indices) else np.array([], dtype=ids_i.dtype)
        image_nums = int((vision_tokens == image_token_id).sum())
        video_nums = int((vision_tokens == video_token_id).sum())

        toks = ids_i.tolist()
        llm_pos_ids: list[np.ndarray] = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        local_fix_point = fix_point

        for _ in range(image_nums + video_nums):
            ed_image = toks.index(image_token_id, st) if (image_token_id in toks[st:] and remain_images > 0) else len(toks) + 1
            ed_video = toks.index(video_token_id, st) if (video_token_id in toks[st:] and remain_videos > 0) else len(toks) + 1
            if ed_image < ed_video:
                t, h, w = image_grid_thw[image_index]
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = video_grid_thw[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t = int(t)
            llm_grid_h = int(h) // spatial_merge_size
            llm_grid_w = int(w) // spatial_merge_size
            text_len = ed - st
            text_len -= int(skip_vision_start_token[image_index - 1])
            text_len = max(0, text_len)

            st_idx = (llm_pos_ids[-1].max() + 1) if llm_pos_ids else 0
            llm_pos_ids.append(np.broadcast_to(np.arange(text_len) + st_idx, (3, text_len)).copy())

            t_index = np.repeat(np.arange(llm_grid_t), llm_grid_h * llm_grid_w)
            h_index = np.tile(np.repeat(np.arange(llm_grid_h), llm_grid_w), llm_grid_t)
            w_index = np.tile(np.arange(llm_grid_w), llm_grid_t * llm_grid_h)

            if int(skip_vision_start_token[image_index - 1]):
                if local_fix_point > 0:
                    local_fix_point = local_fix_point - st_idx
                llm_pos_ids.append(np.stack([t_index, h_index, w_index]) + local_fix_point + st_idx)
                local_fix_point = 0
            else:
                llm_pos_ids.append(np.stack([t_index, h_index, w_index]) + text_len + st_idx)

            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(toks):
            st_idx = (llm_pos_ids[-1].max() + 1) if llm_pos_ids else 0
            text_len = len(toks) - st
            llm_pos_ids.append(np.broadcast_to(np.arange(text_len) + st_idx, (3, text_len)).copy())

        llm_positions = np.concatenate(llm_pos_ids, axis=1).reshape(3, -1)
        position_ids[..., i, attention_mask[i] == 1] = llm_positions
        mrope_position_deltas.append(int(llm_positions.max() + 1 - input_ids.shape[1]))

    deltas = np.asarray(mrope_position_deltas, dtype=np.int64).reshape(-1, 1)
    return position_ids, deltas


def resize_pilimage(pil_image, image_size: int, patch_size: int = PATCH_SIZE, resampler=None):
    """Port of HiDream-O1 utils.py:resize_pilimage.

    Reduce by 2x box resamples until min dim < 2*image_size, then bicubic-fit
    + center-crop to the largest patch-aligned size that doesn't exceed
    image_size**2 area.
    """
    from PIL import Image
    if resampler is None:
        resampler = Image.BICUBIC
    while min(pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    m = patch_size
    width, height = pil_image.width, pil_image.height
    S_max = image_size * image_size
    scale = math.sqrt(S_max / (width * height))

    candidates = [
        (round(width * scale) // m * m, round(height * scale) // m * m),
        (round(width * scale) // m * m, math.floor(height * scale) // m * m),
        (math.floor(width * scale) // m * m, round(height * scale) // m * m),
        (math.floor(width * scale) // m * m, math.floor(height * scale) // m * m),
    ]
    candidates = sorted(candidates, key=lambda x: x[0] * x[1], reverse=True)
    new_w, new_h = next((c for c in candidates if c[0] * c[1] <= S_max), candidates[-1])

    s1 = width / new_w
    s2 = height / new_h
    if s1 < s2:
        pil_image = pil_image.resize([new_w, round(height / s1)], resample=resampler)
        top = (round(height / s1) - new_h) // 2
        pil_image = pil_image.crop((0, top, new_w, top + new_h))
    else:
        pil_image = pil_image.resize([round(width / s2), new_h], resample=resampler)
        left = (round(width / s2) - new_w) // 2
        pil_image = pil_image.crop((left, 0, left + new_w, new_h))
    return pil_image


def calculate_dimensions(max_size: int, ratio: float) -> tuple[int, int]:
    """Port of HiDream-O1 utils.py:calculate_dimensions.

    Pick (w, h) such that max(w*h) <= max_size**2 and w/h ≈ ratio, both
    multiples of 32 (PATCH_SIZE).
    """
    width = math.sqrt(max_size * max_size * ratio)
    height = width / ratio
    width = int(width / 32) * 32
    height = int(height / 32) * 32
    return width, height


def patchify_ref_image(pil_image, patch: int = PATCH_SIZE) -> np.ndarray:
    """Convert a PIL image (already patch-aligned) into HiDream's diffusion-side
    patches: [N_patches, 3*patch*patch] with float32 in [-1, 1].

    Mirrors the upstream `TENSOR_TRANSFORM` (ToTensor + Normalize 0.5/0.5).
    """
    arr = np.asarray(pil_image.convert("RGB"), dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
    arr = (arr - 0.5) / 0.5                                                # [-1, 1]
    arr = arr.transpose(2, 0, 1)                                           # [3, H, W]
    return patchify(arr, patch=patch)                                      # [N, 3*p*p]


def build_edit_text_sample(
    prompt: str,
    ref_image_paths: Sequence[str],
    height: int,
    width: int,
    tokenizer,
    processor,
    model_config,
) -> dict:
    """Build the unified token sequence + position_ids + masks for image edit
    or multi-reference subject-driven generation.

    Faithful port of the multi-ref branch of HiDream-O1 pipeline.py
    generate_image. Single-reference (K=1) is the well-tested path.

    Returns:
      input_ids        [1, txt_seq_len]
      position_ids     [3, 1, total_seq_len]
      token_types      [1, total_seq_len]   (0=AR, 1=tgt+tms, 2=ref)
      vinput_mask      [1, total_seq_len]   (True where diffusion patches go)
      vinput_mask_tgt_only [1, total_seq_len]  (True ONLY for the tgt span; for slicing the prediction)
      pixel_values     [N_vision_patches, vision_patch_dim]   (vision tower input)
      image_grid_thw   [K, 3]                                 (vision tower grid for refs)
      ref_patches      [1, sum(N_ref_patches), 3*32*32]       (clean ref patches for vinputs cat)
      tgt_image_len    int                                    (number of target patches)
    """
    from PIL import Image

    image_token_id = model_config.image_token_id
    video_token_id = model_config.video_token_id
    vision_start_token_id = model_config.vision_start_token_id
    spatial_merge_size = model_config.vision_config.spatial_merge_size

    ref_pils = [Image.open(p).convert("RGB") for p in ref_image_paths]
    K = len(ref_pils)

    if K == 1:
        max_size = max(height, width)
    elif K == 2:
        max_size = max(height, width) * 48 // 64
    elif K <= 4:
        max_size = max(height, width) // 2
    elif K <= 8:
        max_size = max(height, width) * 24 // 64
    else:
        max_size = max(height, width) // 4

    ref_pils_resized: list = []
    ref_patch_lists: list = []
    for pil in ref_pils:
        pil_r = resize_pilimage(pil, max_size, PATCH_SIZE)
        ref_pils_resized.append(pil_r)
        ref_patch_lists.append(patchify_ref_image(pil_r))

    ref_image_lens = [arr.shape[0] for arr in ref_patch_lists]
    total_ref_len = sum(ref_image_lens)
    ref_patches = np.concatenate(ref_patch_lists, axis=0)[None]   # [1, sum(N), 3*32*32]

    tgt_image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)

    if K <= 4:
        cond_img_size = CONDITION_IMAGE_SIZE
    elif K <= 8:
        cond_img_size = CONDITION_IMAGE_SIZE * 48 // 64
    else:
        cond_img_size = CONDITION_IMAGE_SIZE // 2

    ref_pils_vlm = []
    for pil_r in ref_pils_resized:
        cw, ch = calculate_dimensions(cond_img_size, pil_r.width / pil_r.height)
        ref_pils_vlm.append(pil_r.resize((cw, ch), resample=Image.LANCZOS))

    image_grid_thw_tgt = np.asarray([[1, height // PATCH_SIZE, width // PATCH_SIZE]], dtype=np.int64)
    image_grid_thw_ref = np.zeros((K, 3), dtype=np.int64)
    for i, pil_r in enumerate(ref_pils_resized):
        rw, rh = pil_r.size
        image_grid_thw_ref[i] = [1, rh // PATCH_SIZE, rw // PATCH_SIZE]

    boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
    tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")

    content = [{"type": "image"} for _ in range(K)]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    template_caption = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    proc = processor(text=[template_caption], images=ref_pils_vlm, padding="longest", return_tensors="pt")

    input_ids_2 = np.asarray(
        tokenizer.encode(boi_token + tms_token * TIMESTEP_TOKEN_NUM, add_special_tokens=False),
        dtype=np.int64,
    ).reshape(1, -1)
    proc_input_ids = np.asarray(proc.input_ids, dtype=np.int64)
    input_ids = np.concatenate([proc_input_ids, input_ids_2], axis=-1)

    igthw_cond = np.asarray(proc.image_grid_thw, dtype=np.int64).copy()
    for i in range(K):
        igthw_cond[i, 1] //= spatial_merge_size
        igthw_cond[i, 2] //= spatial_merge_size
    igthw_all = np.concatenate([igthw_cond, image_grid_thw_tgt, image_grid_thw_ref], axis=0)

    # Build the per-image vision-token spans appended after the text:
    #   tgt span (tgt_image_len tokens, first slot is vision_start)
    #   then for each ref: span of ref_image_lens[i] tokens, first slot vision_start
    vt_pieces = []
    vt_tgt = np.full((1, tgt_image_len), image_token_id, dtype=input_ids.dtype)
    vt_tgt[0, 0] = vision_start_token_id
    vt_pieces.append(vt_tgt)
    for rl in ref_image_lens:
        vt_ref = np.full((1, rl), image_token_id, dtype=input_ids.dtype)
        vt_ref[0, 0] = vision_start_token_id
        vt_pieces.append(vt_ref)
    vision_tokens = np.concatenate(vt_pieces, axis=1)
    input_ids_pad = np.concatenate([input_ids, vision_tokens], axis=-1)

    position_ids, _ = get_rope_index_fix_point(
        spatial_merge_size=1,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
        input_ids=input_ids_pad,
        image_grid_thw=igthw_all,
        video_grid_thw=None,
        attention_mask=None,
        skip_vision_start_token=[0] * K + [1] + [1] * K,
    )

    txt_seq_len = input_ids.shape[-1]
    all_seq_len = position_ids.shape[-1]

    token_types_raw = np.zeros((1, all_seq_len), dtype=np.int64)
    bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
    end = bgn + tgt_image_len + TIMESTEP_TOKEN_NUM
    token_types_raw[0, bgn:end] = 1                 # tgt span (and tms inside it)
    token_types_raw[0, end: end + total_ref_len] = 2  # ref spans
    token_types_raw[0, txt_seq_len - TIMESTEP_TOKEN_NUM: txt_seq_len] = 3  # tms

    vinput_mask = np.logical_or(token_types_raw == 1, token_types_raw == 2)
    vinput_mask_tgt_only = (token_types_raw == 1)   # excludes tms (=3) and refs (=2)
    token_types_bin = (token_types_raw > 0).astype(np.int64)

    # Pixel values from the processor are pre-flattened patches of vision-tower size.
    # Shape (after np conversion) is [num_vision_patches, vision_patch_dim].
    pixel_values_np = np.asarray(proc.pixel_values, dtype=np.float32)
    image_grid_thw_for_visual = np.asarray(proc.image_grid_thw, dtype=np.int64)

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "token_types": token_types_bin,
        "vinput_mask": vinput_mask,
        "vinput_mask_tgt_only": vinput_mask_tgt_only,
        "pixel_values": pixel_values_np,
        "image_grid_thw": image_grid_thw_for_visual,
        "ref_patches": ref_patches,
        "tgt_image_len": tgt_image_len,
    }


def build_attention_mask(token_types_bin: np.ndarray, dtype_min: float) -> np.ndarray:
    """text rows causal, gen rows bidirectional. Returns [B, 1, S, S] additive."""
    B, S = token_types_bin.shape
    mask = np.full((B, 1, S, S), dtype_min, dtype=np.float32)
    causal_2d = np.triu(np.full((S, S), dtype_min, dtype=np.float32), k=1)
    for b in range(B):
        m = causal_2d.copy()
        gen = token_types_bin[b].astype(bool)
        m[gen, :] = 0.0
        mask[b, 0] = m
    return mask

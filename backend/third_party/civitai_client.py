"""
CivitAI API client
Supports searching LoRAs and models, no NSFW filtering
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import aiohttp


@dataclass
class CivitAIFile:
    """CivitAI file info"""
    name: str
    download_url: str
    size_kb: float = 0.0
    format: str = ""  # SafeTensor, PickleTensor, etc.
    pickle_scan_result: str = ""
    virus_scan_result: str = ""
    scanned_at: Optional[str] = None
    primary: bool = False


@dataclass
class CivitAIModelVersion:
    """CivitAI model version"""
    id: int
    name: str
    description: str = ""
    download_url: str = ""
    trained_words: List[str] = field(default_factory=list)
    base_model: str = ""  # e.g. "Flux.1 D", "SDXL 1.0"
    files: List[CivitAIFile] = field(default_factory=list)
    created_at: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    images: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CivitAIModel:
    """CivitAI model info"""
    id: int
    name: str
    description: str = ""
    type: str = ""  # LORA, Checkpoint, etc.
    nsfw: bool = False
    tags: List[str] = field(default_factory=list)
    creator: Dict[str, str] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    model_versions: List[CivitAIModelVersion] = field(default_factory=list)


class CivitAIClient:
    """CivitAI API client

    Note: does not filter NSFW content, returns all search results
    """

    BASE_URL = "https://civitai.com/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def search(self, query: str = "", types: Optional[List[str]] = None,
                    limit: int = 20, page: int = 1, cursor: Optional[str] = None,
                    sort: str = "Highest Rated", nsfw: Optional[bool] = None) -> Dict[str, Any]:
        """Search models

        CivitAI uses cursor-based pagination; page parameter is not supported when query is present.
        When query is not empty, the page parameter is ignored and cursor-based pagination is used instead.

        Args:
            query: Search keyword
            types: Model type list, e.g. ["LORA", "Checkpoint"]
            limit: Items per page
            page: Page number (only effective when no query)
            cursor: Pagination cursor (used when query is present)
            sort: Sort method
            nsfw: Whether to include NSFW content (requires API key)

        Returns:
            Dict with items and metadata
        """
        session = await self._get_session()

        params: Dict[str, Any] = {
            "limit": limit,
            "sort": sort,
        }

        if query:
            params["query"] = query
            if cursor:
                params["cursor"] = cursor
        else:
            params["page"] = page

        if types:
            params["types"] = ",".join(types)

        if nsfw is not None:
            params["nsfw"] = "true" if nsfw else "false"

        try:
            async with session.get(f"{self.BASE_URL}/models", params=params) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"CivitAI API error {resp.status}: {text}")

                data = await resp.json()
                items = [self._parse_model(item) for item in data.get("items", [])]
                metadata = data.get("metadata", {})
                return {"items": items, "metadata": metadata}

        except aiohttp.ClientError as e:
            raise Exception(f"CivitAI network request failed: {e}")

    async def get_model(self, model_id: int) -> CivitAIModel:
        """Get model details"""
        session = await self._get_session()

        try:
            async with session.get(f"{self.BASE_URL}/models/{model_id}") as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"CivitAI API error {resp.status}: {text}")

                data = await resp.json()
                return self._parse_model(data)

        except aiohttp.ClientError as e:
            raise Exception(f"CivitAI network request failed: {e}")

    async def get_model_version(self, version_id: int) -> CivitAIModelVersion:
        """Get a single model version (files + download URLs)."""
        session = await self._get_session()

        try:
            async with session.get(f"{self.BASE_URL}/model-versions/{version_id}") as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"CivitAI API error {resp.status}: {text}")

                data = await resp.json()
                files = []
                for f in data.get("files", []):
                    files.append(
                        CivitAIFile(
                            name=f.get("name", ""),
                            download_url=f.get("downloadUrl", ""),
                            size_kb=f.get("sizeKB", 0.0),
                            format=f.get("metadata", {}).get("format", ""),
                            pickle_scan_result=f.get("pickleScanResult", ""),
                            virus_scan_result=f.get("virusScanResult", ""),
                            scanned_at=f.get("scannedAt"),
                            primary=f.get("primary", False),
                        )
                    )
                return CivitAIModelVersion(
                    id=data.get("id", 0),
                    name=data.get("name", ""),
                    description=data.get("description", ""),
                    download_url=data.get("downloadUrl", ""),
                    trained_words=data.get("trainedWords", []),
                    base_model=data.get("baseModel", ""),
                    files=files,
                    created_at=data.get("createdAt"),
                    stats=data.get("stats", {}),
                    images=data.get("images", []),
                )

        except aiohttp.ClientError as e:
            raise Exception(f"CivitAI network request failed: {e}")

    def _parse_model(self, data: Dict[str, Any]) -> CivitAIModel:
        """Parse model data returned by API"""
        versions = []
        for v in data.get("modelVersions", []):
            files = []
            for f in v.get("files", []):
                files.append(CivitAIFile(
                    name=f.get("name", ""),
                    download_url=f.get("downloadUrl", ""),
                    size_kb=f.get("sizeKB", 0.0),
                    format=f.get("metadata", {}).get("format", ""),
                    pickle_scan_result=f.get("pickleScanResult", ""),
                    virus_scan_result=f.get("virusScanResult", ""),
                    scanned_at=f.get("scannedAt"),
                    primary=f.get("primary", False)
                ))

            versions.append(CivitAIModelVersion(
                id=v.get("id", 0),
                name=v.get("name", ""),
                description=v.get("description", ""),
                download_url=v.get("downloadUrl", ""),
                trained_words=v.get("trainedWords", []),
                base_model=v.get("baseModel", ""),
                files=files,
                created_at=v.get("createdAt"),
                stats=v.get("stats", {}),
                images=v.get("images", [])
            ))

        return CivitAIModel(
            id=data.get("id", 0),
            name=data.get("name", ""),
            description=data.get("description", ""),
            type=data.get("type", ""),
            nsfw=data.get("nsfw", False),
            tags=data.get("tags", []),
            creator=data.get("creator", {}),
            stats=data.get("stats", {}),
            model_versions=versions
        )

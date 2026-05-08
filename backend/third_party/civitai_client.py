"""
CivitAI API 客户端
支持搜索 LoRA 和模型，不过滤 NSFW 内容
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import aiohttp


@dataclass
class CivitAIFile:
    """CivitAI 文件信息"""
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
    """CivitAI 模型版本"""
    id: int
    name: str
    description: str = ""
    download_url: str = ""
    trained_words: List[str] = field(default_factory=list)
    base_model: str = ""  # 如 "Flux.1 D", "SDXL 1.0"
    files: List[CivitAIFile] = field(default_factory=list)
    created_at: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    images: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CivitAIModel:
    """CivitAI 模型信息"""
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
    """CivitAI API 客户端

    注意：不过滤 NSFW 内容，返回所有搜索结果
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
        """搜索模型

        CivitAI 使用 cursor 分页，不支持 page 参数与 query 同时使用。
        当 query 不为空时，page 参数会被忽略，改用 cursor 分页。

        Args:
            query: 搜索关键词
            types: 模型类型列表，如 ["LORA", "Checkpoint"]
            limit: 每页数量
            page: 页码（仅无 query 时生效）
            cursor: 分页游标（有 query 时使用）
            sort: 排序方式
            nsfw: 是否包含 NSFW 内容（需要 API key）

        Returns:
            包含 items 和 metadata 的字典
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
            raise Exception(f"CivitAI 网络请求失败: {e}")

    async def get_model(self, model_id: int) -> CivitAIModel:
        """获取模型详情"""
        session = await self._get_session()

        try:
            async with session.get(f"{self.BASE_URL}/models/{model_id}") as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"CivitAI API error {resp.status}: {text}")

                data = await resp.json()
                return self._parse_model(data)

        except aiohttp.ClientError as e:
            raise Exception(f"CivitAI 网络请求失败: {e}")

    def _parse_model(self, data: Dict[str, Any]) -> CivitAIModel:
        """解析 API 返回的模型数据"""
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

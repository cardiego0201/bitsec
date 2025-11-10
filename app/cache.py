import asyncio
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo.errors import DuplicateKeyError


DEFAULT_MONGODB_URI = "mongodb://localhost:27017"
DEFAULT_DB_NAME = "bitsec"
DEFAULT_COLLECTION_NAME = "analyses"


@dataclass(frozen=True)
class CachedAnalysisResult:
    vulnerability: str
    confidence: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vulnerability": self.vulnerability,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedAnalysisResult":
        return cls(
            vulnerability=data["vulnerability"],
            confidence=float(data["confidence"]),
            rationale=data["rationale"],
        )


class AnalysisCache:
    def __init__(self) -> None:
        self._uri = os.environ.get("MONGODB_URI", DEFAULT_MONGODB_URI)
        self._db_name = os.environ.get("MONGODB_DB", DEFAULT_DB_NAME)
        self._collection_name = os.environ.get("MONGODB_COLLECTION", DEFAULT_COLLECTION_NAME)
        self._client: Optional[AsyncIOMotorClient] = None
        self._collection: Optional[AsyncIOMotorCollection] = None
        self._connect_lock = asyncio.Lock()

    @staticmethod
    def compute_hash(code: str) -> str:
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    async def connect(self) -> None:
        async with self._connect_lock:
            if self._client is not None and self._collection is not None:
                return

            self._client = AsyncIOMotorClient(self._uri)
            database = self._client[self._db_name]
            self._collection = database[self._collection_name]
            await self._collection.create_index("code_hash", unique=True)

    async def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            self._collection = None

    @property
    def collection(self) -> AsyncIOMotorCollection:
        if self._collection is None:
            raise RuntimeError("AnalysisCache.connect() must be called before use.")
        return self._collection

    async def get_entry(self, code_hash: str) -> Optional[Dict[str, Any]]:
        return await self.collection.find_one({"code_hash": code_hash})

    async def start_processing(self, code_hash: str) -> bool:
        now = datetime.now(timezone.utc)
        try:
            await self.collection.insert_one(
                {
                    "code_hash": code_hash,
                    "status": "processing",
                    "created_at": now,
                    "updated_at": now,
                }
            )
            return True
        except DuplicateKeyError:
            previous = await self.collection.find_one_and_update(
                {"code_hash": code_hash, "status": "failed"},
                {
                    "$set": {"status": "processing", "updated_at": now},
                    "$unset": {"result": "", "error": "", "clear_code": ""},
                },
            )
            return previous is not None

    async def mark_completed(
        self,
        code_hash: str,
        result: CachedAnalysisResult,
        clear_code: Optional[str],
    ) -> None:
        now = datetime.now(timezone.utc)
        await self.collection.update_one(
            {"code_hash": code_hash},
            {
                "$set": {
                    "status": "completed",
                    "result": result.to_dict(),
                    "clear_code": clear_code,
                    "updated_at": now,
                },
                "$unset": {"error": ""},
            },
        )

    async def mark_failed(self, code_hash: str, error_message: str) -> None:
        now = datetime.now(timezone.utc)
        await self.collection.update_one(
            {"code_hash": code_hash},
            {
                "$set": {
                    "status": "failed",
                    "error": error_message,
                    "updated_at": now,
                },
                "$unset": {"result": "", "clear_code": ""},
            },
        )

    async def wait_for_result(
        self,
        code_hash: str,
        poll_interval: float = 0.5,
        timeout_seconds: Optional[float] = None,
    ) -> Optional[CachedAnalysisResult]:
        start_time = datetime.now(timezone.utc)
        while True:
            entry = await self.get_entry(code_hash)
            if entry is None:
                return None

            status = entry.get("status")
            if status == "completed":
                return CachedAnalysisResult.from_dict(entry["result"])
            if status == "failed":
                raise RuntimeError(entry.get("error") or "Analysis failed.")

            if timeout_seconds is not None:
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                if elapsed > timeout_seconds:
                    raise TimeoutError("Timed out waiting for analysis result.")

            await asyncio.sleep(poll_interval)



"""
ChromaDB Embedder —— 把 schema metadata 向量化并存入 ChromaDB。

使用 BAAI/bge-small-zh 作 embedding 模型（本地运行、免 API 费、中文友好）。

Schema Agent 通过此模块检索最相关的表/字段。
"""

import chromadb
from hashlib import sha256
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Dict

from config.settings import get_settings
from storage.chromadb.schema_metadata import (
    SCHEMA_FIELDS,
    SCHEMA_CATALOG_VERSION,
    get_documents_for_embedding,
    schema_catalog_fingerprint,
)


class SchemaEmbedder:
    """Schema 向量化 & 检索封装。"""

    def __init__(self, model_name: str = None, persist_dir: str = None):
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.collection_name = settings.CHROMA_COLLECTION

        self._model: SentenceTransformer | None = None
        self._client: chromadb.PersistentClient | None = None
        self._collection: chromadb.Collection | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"[Embedder] 加载模型: {self.model_name} ...")
            self._model = SentenceTransformer(self.model_name)
            if hasattr(self._model, "get_embedding_dimension"):
                get_dimension = self._model.get_embedding_dimension
            else:
                get_dimension = self._model.get_sentence_embedding_dimension
            print(f"[Embedder] 模型加载完成 (dim={get_dimension()})")
        return self._model

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            print(f"[Embedder] 连接 ChromaDB: {self.persist_dir}")
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            try:
                self._collection = self.client.get_collection(self.collection_name)
            except Exception:
                # 不存在则后续 build() 时创建
                pass
        return self._collection

    def build(self, force_rebuild: bool = False) -> int:
        """
        构建（或重建）schema metadata 的向量索引。

        Args:
            force_rebuild: True 时先删除旧 collection 再重建

        Returns:
            索引的文档数量
        """
        if force_rebuild:
            try:
                self.client.delete_collection(self.collection_name)
                print(f"[Embedder] 删除旧 collection: {self.collection_name}")
            except Exception:
                pass

        docs = get_documents_for_embedding()
        print(f"[Embedder] 生成 {len(docs)} 条文档的 embedding ...")

        embeddings = self.model.encode(docs, show_progress_bar=True).tolist()

        ids = [
            "field_" + sha256(
                f"{field['data_source']}:{field['table_name']}:{field['column_name']}".encode()
            ).hexdigest()[:20]
            for field in SCHEMA_FIELDS
        ]
        metadatas = [
            {
                "table_name": f["table_name"],
                "column_name": f["column_name"],
                "dtype": f["dtype"],
                "business_term": f["business_term"],
                "data_source": f["data_source"],
                "catalog_version": f["catalog_version"],
                "source_id": f["source_id"],
                "access_scope": f["access_scope"],
                "owner_id": f.get("owner_id", ""),
            }
            for f in SCHEMA_FIELDS
        ]

        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "Versioned Schema metadata for Text2SQL",
                "catalog_version": SCHEMA_CATALOG_VERSION,
                "catalog_fingerprint": schema_catalog_fingerprint(),
            },
        )
        collection.modify(metadata={
            "description": "Versioned Schema metadata for Text2SQL",
            "catalog_version": SCHEMA_CATALOG_VERSION,
            "catalog_fingerprint": schema_catalog_fingerprint(),
        })

        # 如果已有数据，先清空
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=docs,
            metadatas=metadatas,
        )

        self._collection = collection
        print(f"[Embedder] 索引完成: {collection.count()} 条记录")
        return collection.count()

    def index_is_current(self) -> bool:
        collection = self.collection
        if collection is None:
            return False
        metadata = collection.metadata or {}
        return metadata.get("catalog_fingerprint") == schema_catalog_fingerprint()

    def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        data_source: str = "ai_analytics",
        principal_id: str | None = None,
    ) -> List[Dict]:
        """
        语义检索最相关的表/字段。

        Args:
            query: 用户自然语言查询（如"上个月GMV最高的前10个客户"）
            top_k: 返回 top-k 结果

        Returns:
            [{table_name, column_name, business_term, score, ...}, ...]
        """
        if self.collection is None:
            raise RuntimeError("Collection 未初始化，请先调用 build()")

        query_embedding = self.model.encode([query]).tolist()
        access_filter = {"access_scope": {"$eq": "public"}}
        if principal_id:
            access_filter = {"$or": [
                {"access_scope": {"$eq": "public"}},
                {"owner_id": {"$eq": principal_id}},
            ]}
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
            where={"$and": [
                {"data_source": {"$eq": data_source}},
                access_filter,
            ]},
        )

        formatted = []
        for i, (meta, doc, dist) in enumerate(zip(
            results["metadatas"][0],
            results["documents"][0],
            results["distances"][0],
        )):
            formatted.append({
                "rank": i + 1,
                "table_name": meta["table_name"],
                "column_name": meta["column_name"],
                "business_term": meta["business_term"],
                "dtype": meta["dtype"],
                "data_source": meta["data_source"],
                "catalog_version": meta["catalog_version"],
                "source_id": meta["source_id"],
                "access_scope": meta["access_scope"],
                "document": doc,
                "score": round(1 - dist, 4),  # distance → similarity
            })

        return formatted


# 全局单例
_embedder: SchemaEmbedder | None = None


def get_embedder() -> SchemaEmbedder:
    """获取 SchemaEmbedder 单例。"""
    global _embedder
    if _embedder is None:
        _embedder = SchemaEmbedder()
    return _embedder

"""
Milvus Vector Store Integration
================================

Professional Milvus integration for the RAG system.
"""

import os
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from tqdm import tqdm

try:
    from pymilvus import (
        connections, Collection, CollectionSchema,
        FieldSchema, DataType, utility,
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

import sys
sys.path.append('..')
from config import settings

logger = logging.getLogger(__name__)

@dataclass
class MilvusConfig:
    host: str = "localhost"
    port: int = 19530
    user: str = ""
    password: str = ""
    collection_name: str = "rag_documents"
    embedding_dim: int = 1024
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"
    nlist: int = 1024
    nprobe: int = 16
    ef: int = 64
    m: int = 16

@dataclass
class SearchResult:
    id: int
    score: float
    text: str
    metadata: Dict = field(default_factory=dict)

class MilvusVectorStore:
    def __init__(self, config: MilvusConfig = None, collection_name: str = None,
                 host: str = "localhost", port: int = 19530, embedding_dim: int = 1024):
        if not MILVUS_AVAILABLE:
            raise ImportError("pymilvus not installed. Run: pip install pymilvus")
        self.config = config or MilvusConfig(host=host, port=port,
            collection_name=collection_name or "rag_documents", embedding_dim=embedding_dim)
        self.collection = None
        self._connected = False
        self.stats = {'vectors_inserted': 0, 'searches_performed': 0, 'connection_time': None}

    def connect(self) -> bool:
        try:
            print(f"Connecting to Milvus at {self.config.host}:{self.config.port}...")
            connections.connect(alias="default", host=self.config.host, 
                port=self.config.port, user=self.config.user, password=self.config.password)
            self._connected = True
            self.stats['connection_time'] = datetime.now().isoformat()
            print("Connected to Milvus successfully")
            return True
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            return False

    def disconnect(self):
        if self._connected:
            connections.disconnect("default")
            self._connected = False

    def create_collection(self, drop_existing: bool = False, description: str = "RAG embeddings") -> bool:
        if not self._connected:
            self.connect()
        if utility.has_collection(self.config.collection_name):
            if drop_existing:
                utility.drop_collection(self.config.collection_name)
            else:
                self.collection = Collection(self.config.collection_name)
                self.collection.load()
                return True
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.config.embedding_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="page_start", dtype=DataType.INT32),
            FieldSchema(name="page_end", dtype=DataType.INT32),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="vendor", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="doc_title", dtype=DataType.VARCHAR, max_length=512),
        ]
        schema = CollectionSchema(fields=fields, description=description)
        self.collection = Collection(name=self.config.collection_name, schema=schema)
        self._create_index()
        self.collection.load()
        print(f"Collection created: {self.config.collection_name}")
        return True

    def _create_index(self):
        if self.config.index_type == "HNSW":
            params = {"metric_type": self.config.metric_type, "index_type": "HNSW",
                "params": {"M": self.config.m, "efConstruction": 200}}
        else:
            params = {"metric_type": self.config.metric_type, "index_type": self.config.index_type,
                "params": {"nlist": self.config.nlist}}
        self.collection.create_index(field_name="embedding", index_params=params)

    def insert_chunks(self, chunks: List[Dict], embeddings: List[List[float]], 
                      batch_size: int = 1000, show_progress: bool = True) -> int:
        if not self.collection:
            raise ValueError("Collection not initialized")
        total = len(chunks)
        inserted = 0
        print(f"Inserting {total:,} vectors...")
        iterator = range(0, total, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Inserting", unit="batch")
        for i in iterator:
            batch_c = chunks[i:i + batch_size]
            batch_e = embeddings[i:i + batch_size]
            data = [batch_e,
                [c.get('text', '')[:65000] for c in batch_c],
                [c.get('chunk_id', f'chunk_{i+j}') for j, c in enumerate(batch_c)],
                [c.get('source', '') for c in batch_c],
                [c.get('page_start', 0) for c in batch_c],
                [c.get('page_end', 0) for c in batch_c],
                [c.get('section', '')[:500] for c in batch_c],
                [c.get('vendor', '') or '' for c in batch_c],
                [c.get('chunk_type', 'text') for c in batch_c],
                [c.get('title', '')[:500] for c in batch_c]]
            self.collection.insert(data)
            inserted += len(batch_c)
        self.collection.flush()
        self.stats['vectors_inserted'] += inserted
        return inserted

    def search(self, query_embedding: List[float], top_k: int = 10, 
               filter_expr: str = None, output_fields: List[str] = None) -> List[SearchResult]:
        if not self.collection:
            raise ValueError("Collection not initialized")
        output_fields = output_fields or ["text", "chunk_id", "source", "section", "vendor", "chunk_type"]
        if self.config.index_type == "HNSW":
            params = {"metric_type": self.config.metric_type, "params": {"ef": self.config.ef}}
        else:
            params = {"metric_type": self.config.metric_type, "params": {"nprobe": self.config.nprobe}}
        results = self.collection.search(data=[query_embedding], anns_field="embedding",
            param=params, limit=top_k, expr=filter_expr, output_fields=output_fields)
        self.stats['searches_performed'] += 1
        search_results = []
        for hits in results:
            for hit in hits:
                search_results.append(SearchResult(id=hit.id, score=hit.score,
                    text=hit.entity.get('text', ''),
                    metadata={'chunk_id': hit.entity.get('chunk_id', ''),
                        'source': hit.entity.get('source', ''),
                        'section': hit.entity.get('section', ''),
                        'vendor': hit.entity.get('vendor', ''),
                        'chunk_type': hit.entity.get('chunk_type', '')}))
        return search_results

    def hybrid_search(self, query_embedding: List[float], top_k: int = 10,
                      vendor: str = None, chunk_type: str = None) -> List[SearchResult]:
        filters = []
        if vendor:
            filters.append(f'vendor == "{vendor}"')
        if chunk_type:
            filters.append(f'chunk_type == "{chunk_type}"')
        filter_expr = " and ".join(filters) if filters else None
        return self.search(query_embedding, top_k, filter_expr)

    def get_collection_stats(self) -> Dict:
        if not self.collection:
            return {}
        return {'name': self.config.collection_name, 'num_entities': self.collection.num_entities,
            'index_type': self.config.index_type, 'stats': self.stats}

    def drop_collection(self):
        if utility.has_collection(self.config.collection_name):
            utility.drop_collection(self.config.collection_name)
        self.collection = None

def check_milvus_connection(host: str = "localhost", port: int = 19530) -> bool:
    if not MILVUS_AVAILABLE:
        print("pymilvus not installed")
        return False
    try:
        connections.connect(alias="test", host=host, port=port)
        connections.disconnect("test")
        print(f"Milvus running at {host}:{port}")
        return True
    except Exception as e:
        print(f"Cannot connect: {e}")
        return False

if __name__ == "__main__":
    check_milvus_connection()
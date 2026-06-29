"""
SAR (Schema-Aware Retriever) inference — retrieve top-k similar examples.
Adapted from SchemaRAG SAR_use.py:
- Loads trained SchemaAwareModel weights.
- Encodes the question using BGE, computes schema-aware embedding.
- Retrieves top-k corpus entries by cosine similarity.
- Returns (question, sql/mql, structural_type) for use as few-shot examples.

Two retriever backends:

SARRetriever (original):
    Loads corpus + SAR model, pre-computes all embeddings in memory at startup.
    No ChromaDB required. Startup ~30 sec (re-encodes all questions each time).

ChromaSARRetriever (Phase 13):
    Queries a pre-built ChromaDB index. Startup is instant — no re-encoding.
    Requires: build_chroma_index.py to have been run first.

Use get_sar_retriever(config, track) to select the right backend from config.yaml:
    sar.backend: "memory"  → SARRetriever
    sar.backend: "chroma"  → ChromaSARRetriever
"""

from __future__ import annotations

import json
from typing import Dict, List

import torch
import torch.nn.functional as F

from src.device import get_device
from src.sar.sar_model import SchemaAwareModel


class SARRetriever:
    def __init__(
        self,
        model_path: str,
        corpus_path: str,
        bge_model:  str = "BAAI/bge-large-en-v1.5",
        embed_dim:  int = 1024,
    ):
        from FlagEmbedding import FlagModel

        self.device     = get_device()
        self.flag_model = FlagModel(bge_model, use_fp16=(self.device != "cpu"))

        self.sar_model = SchemaAwareModel(embed_dim=embed_dim).to(self.device)
        self.sar_model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.sar_model.eval()

        with open(corpus_path, encoding="utf-8") as f:
            self.corpus = json.load(f)
        print(f"Loaded corpus: {len(self.corpus)} entries")

        print("Pre-computing corpus embeddings ...")
        questions = [item["question"] for item in self.corpus]
        raw_embs  = self.flag_model.encode(questions)
        q_tensor  = torch.tensor(raw_embs, dtype=torch.float32).to(self.device)
        N, D = q_tensor.shape
        dt = torch.zeros(N, 1, D, device=self.device)
        dc = torch.zeros(N, 1, 1, D, device=self.device)
        tm = torch.zeros(N, 1, dtype=torch.bool, device=self.device)
        cm = torch.zeros(N, 1, 1, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            self.corpus_embs = F.normalize(
                self.sar_model(q_tensor, dt, dc, tm, cm), dim=-1
            )  # [N, D]

    def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        """
        Return the top-k most structurally similar corpus entries.

        Args:
            question: Natural language question at inference time.
            top_k:    Number of examples to return.

        Returns:
            List of corpus dicts (question, sql / mql_pipeline, structural_type, db_name).
        """
        raw  = self.flag_model.encode([question])
        q_in = torch.tensor(raw, dtype=torch.float32).to(self.device)  # [1, D]
        D = q_in.shape[1]
        dt = torch.zeros(1, 1, D, device=self.device)
        dc = torch.zeros(1, 1, 1, D, device=self.device)
        tm = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        cm = torch.zeros(1, 1, 1, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            q_emb = F.normalize(
                self.sar_model(q_in, dt, dc, tm, cm), dim=-1
            )  # [1, D]

        scores  = torch.matmul(q_emb, self.corpus_embs.T).squeeze(0)  # [N]
        top_idx = torch.topk(scores, k=min(top_k, len(self.corpus))).indices.tolist()
        return [self.corpus[i] for i in top_idx]


# ---------------------------------------------------------------------------
# ChromaDB-backed retriever (Phase 13) — instant startup, persistent index
# ---------------------------------------------------------------------------

class ChromaSARRetriever:
    """
    SAR retrieval backed by a pre-built ChromaDB index.

    Startup is instant (no re-encoding). The SAR model is still needed at
    query time to encode the incoming question before searching the index.

    Requires:
        scripts/build_chroma_index.py to have been run for this corpus.

    Args:
        model_path:      Path to sar_model.pt (SchemaAwareModel weights).
        chroma_dir:      Directory containing the ChromaDB files.
        collection_name: ChromaDB collection name used at build time (e.g. "sar_sql").
        bge_model:       BGE model name (must match the one used at build time).
        embed_dim:       Embedding dimension (must match SchemaAwareModel).
    """

    def __init__(
        self,
        model_path:       str,
        chroma_dir:       str,
        collection_name:  str,
        bge_model:        str = "BAAI/bge-large-en-v1.5",
        embed_dim:        int = 1024,
    ):
        import chromadb
        from FlagEmbedding import FlagModel

        self.device     = get_device()
        self.flag_model = FlagModel(bge_model, use_fp16=(self.device != "cpu"))

        self.sar_model = SchemaAwareModel(embed_dim=embed_dim).to(self.device)
        self.sar_model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.sar_model.eval()

        client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = client.get_collection(collection_name)
        print(f"ChromaDB collection '{collection_name}': {self.collection.count()} entries")

    def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        """
        Return the top-k most structurally similar corpus entries from ChromaDB.

        Args:
            question: Natural language question at inference time.
            top_k:    Number of examples to return.

        Returns:
            List of dicts with question, sql / mql_pipeline, structural_type, db_name.
        """
        raw  = self.flag_model.encode([question])
        q_in = torch.tensor(raw, dtype=torch.float32).to(self.device)
        D    = q_in.shape[1]
        dt = torch.zeros(1, 1, D,    device=self.device)
        dc = torch.zeros(1, 1, 1, D, device=self.device)
        tm = torch.zeros(1, 1,       dtype=torch.bool, device=self.device)
        cm = torch.zeros(1, 1, 1,    dtype=torch.bool, device=self.device)

        with torch.no_grad():
            q_emb = F.normalize(
                self.sar_model(q_in, dt, dc, tm, cm), dim=-1
            ).squeeze(0).cpu().numpy().tolist()

        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["metadatas", "documents"],
        )

        entries = []
        for meta in results["metadatas"][0]:
            entry: Dict = {
                "question":        meta.get("question", ""),
                "db_name":         meta.get("db_name", ""),
                "structural_type": json.loads(meta.get("structural_type", "{}")),
            }
            if "sql" in meta:
                entry["sql"] = meta["sql"]
            if "mql_pipeline" in meta:
                entry["mql_pipeline"]   = json.loads(meta["mql_pipeline"])
                entry["mql_collection"] = meta.get("mql_collection", "")
            entries.append(entry)

        return entries


# ---------------------------------------------------------------------------
# Factory — select backend from config
# ---------------------------------------------------------------------------

def get_sar_retriever(sar_config: dict, track: str = "sql"):
    """
    Build and return the appropriate SAR retriever from config.yaml.

    Args:
        sar_config: The `sar` section of configs/config.yaml.
        track:      "sql" or "nosql".

    Config keys used:
        backend:          "memory" (default) or "chroma"
        sql_checkpoint:   path to sar_sql/sar_model.pt
        nosql_checkpoint: path to sar_nosql/sar_model.pt
        sql_chroma_dir:   path to chroma_sql/ index (chroma backend only)
        nosql_chroma_dir: path to chroma_nosql/ index (chroma backend only)
        sql_collection:   ChromaDB collection name (default: "sar_sql")
        nosql_collection: ChromaDB collection name (default: "sar_nosql")
    """
    backend = sar_config.get("backend", "memory")
    is_sql  = (track == "sql")

    model_key  = "sql_checkpoint"   if is_sql else "nosql_checkpoint"
    corpus_key = "sql_corpus"       if is_sql else "nosql_corpus"
    chroma_key = "sql_chroma_dir"   if is_sql else "nosql_chroma_dir"
    coll_key   = "sql_collection"   if is_sql else "nosql_collection"
    coll_def   = "sar_sql"          if is_sql else "sar_nosql"

    model_path = sar_config.get(model_key)
    if not model_path:
        raise ValueError(f"sar.{model_key} not set in config.yaml")

    if backend == "chroma":
        chroma_dir = sar_config.get(chroma_key)
        if not chroma_dir:
            raise ValueError(f"sar.{chroma_key} not set in config.yaml (required for backend=chroma)")
        return ChromaSARRetriever(
            model_path=model_path,
            chroma_dir=chroma_dir,
            collection_name=sar_config.get(coll_key, coll_def),
            bge_model=sar_config.get("encoder_model", "BAAI/bge-large-en-v1.5"),
            embed_dim=sar_config.get("embed_dim", 1024),
        )

    # Default: in-memory SARRetriever
    corpus_path = sar_config.get(corpus_key)
    if not corpus_path:
        raise ValueError(f"sar.{corpus_key} not set in config.yaml (required for backend=memory)")
    return SARRetriever(
        model_path=model_path,
        corpus_path=corpus_path,
        bge_model=sar_config.get("encoder_model", "BAAI/bge-large-en-v1.5"),
        embed_dim=sar_config.get("embed_dim", 1024),
    )

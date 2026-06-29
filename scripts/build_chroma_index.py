"""
Phase 13 — ChromaDB Index Building

Encodes the entire RAG corpus using the trained SAR model (BGE + SchemaAwareModel)
and stores the embeddings in a persistent ChromaDB collection.

Run once after Phase 12A/12B SAR training. After this, SARRetriever startup
is instant (no re-encoding) because ChromaDB serves pre-built embeddings.

Usage (SQL):
    python -m scripts.build_chroma_index \
        --corpus  Data/rag_corpus/spider_sql_rag.json \
        --model   /path/to/sar_sql/sar_model.pt \
        --out     indexes/chroma_sql \
        --name    sar_sql

Usage (NoSQL):
    python -m scripts.build_chroma_index \
        --corpus  Data/rag_corpus/spider_nosql_rag.json \
        --model   /path/to/sar_nosql/sar_model.pt \
        --out     indexes/chroma_nosql \
        --name    sar_nosql

On Colab (models are on Drive):
    python -m scripts.build_chroma_index \
        --corpus  Data/rag_corpus/spider_sql_rag.json \
        --model   /content/drive/MyDrive/codegen/checkpoints/sar_sql/sar_model.pt \
        --out     /content/drive/MyDrive/codegen/indexes/chroma_sql \
        --name    sar_sql
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from src.sar.sar_model import SchemaAwareModel
from src.device import get_device


def build_chroma_index(
    corpus_path: str,
    model_path: str,
    chroma_dir: str,
    collection_name: str,
    bge_model: str = "BAAI/bge-large-en-v1.5",
    embed_dim: int = 1024,
    batch_size: int = 500,
):
    import chromadb
    from FlagEmbedding import FlagModel

    device = get_device()
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load corpus
    # ------------------------------------------------------------------
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"Corpus: {len(corpus)} entries  ({corpus_path})")

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print("Loading BGE model ...")
    flag_model = FlagModel(bge_model, use_fp16=(device == "cuda"))

    print(f"Loading SAR model from {model_path} ...")
    sar_model = SchemaAwareModel(embed_dim=embed_dim).to(device)
    sar_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    sar_model.eval()

    # ------------------------------------------------------------------
    # Encode all questions: BGE → SchemaAwareModel → normalized 1024-dim
    # ------------------------------------------------------------------
    print("Encoding corpus questions with BGE ...")
    questions = [item["question"] for item in corpus]
    raw_embs  = flag_model.encode(questions)            # numpy [N, D]

    print("Projecting through SchemaAwareModel ...")
    q_tensor = torch.tensor(raw_embs, dtype=torch.float32).to(device)
    N, D = q_tensor.shape

    # Zero-filled schema tensors (same as in SARRetriever.infer.py)
    dt = torch.zeros(N, 1, D,    device=device)
    dc = torch.zeros(N, 1, 1, D, device=device)
    tm = torch.zeros(N, 1,       dtype=torch.bool, device=device)
    cm = torch.zeros(N, 1, 1,    dtype=torch.bool, device=device)

    with torch.no_grad():
        corpus_embs = F.normalize(sar_model(q_tensor, dt, dc, tm, cm), dim=-1)

    embeddings = corpus_embs.cpu().numpy().tolist()     # list[list[float]]

    # ------------------------------------------------------------------
    # Build ChromaDB collection
    # ------------------------------------------------------------------
    os.makedirs(chroma_dir, exist_ok=True)
    print(f"Building ChromaDB collection '{collection_name}' at {chroma_dir} ...")

    client     = chromadb.PersistentClient(path=chroma_dir)

    # Delete and recreate if it already exists (clean rebuild)
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        print(f"  Dropping existing collection '{collection_name}' for clean rebuild")
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Build metadata — all values must be str/int/float/bool for ChromaDB
    metadatas = []
    for item in corpus:
        meta: dict = {
            "question":        item.get("question", ""),
            "db_name":         item.get("db_name", ""),
            "structural_type": json.dumps(item.get("structural_type", {})),
        }
        if "sql" in item:
            meta["sql"] = item["sql"]
        if "mql_pipeline" in item:
            meta["mql_pipeline"]   = json.dumps(item["mql_pipeline"])
            meta["mql_collection"] = item.get("mql_collection", "")
        metadatas.append(meta)

    # Add in batches (ChromaDB has a default limit of 41665 per batch)
    ids = [str(i) for i in range(len(corpus))]
    for start in range(0, len(corpus), batch_size):
        end = min(start + batch_size, len(corpus))
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=questions[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  Indexed {end}/{len(corpus)} entries")

    total = collection.count()
    print(f"\nDone. ChromaDB collection '{collection_name}': {total} entries → {chroma_dir}")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="spider_sql_rag.json or spider_nosql_rag.json")
    parser.add_argument("--model",  required=True, help="Path to sar_model.pt")
    parser.add_argument("--out",    required=True, help="Directory to save ChromaDB files")
    parser.add_argument("--name",   required=True, help="ChromaDB collection name (e.g. sar_sql)")
    parser.add_argument("--bge",    default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--batch",  type=int, default=500)
    args = parser.parse_args()

    build_chroma_index(
        corpus_path=args.corpus,
        model_path=args.model,
        chroma_dir=args.out,
        collection_name=args.name,
        bge_model=args.bge,
        embed_dim=args.embed_dim,
        batch_size=args.batch,
    )


if __name__ == "__main__":
    main()

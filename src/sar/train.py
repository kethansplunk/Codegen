"""
SAR (Schema-Aware Retriever) training — contrastive learning.
Adapted from SchemaRAG train_SAR.py:
- SchemaAwareModel imported from src/sar/sar_model.py (same architecture).
- FlagModel (BAAI/bge-large-en-v1.5) produces base embeddings.
- Contrastive loss: positives = same structural_type, negatives = different.
- Structural type from our spider_sql_rag.json (7-dim vector).
- Embedding cache (pickle) preserved for speed — BGE encoding is slow.
- MPS / CUDA / CPU device support via src/device.py.

Run on Colab T4:
    python -m src.sar.train \
        --corpus Data/rag_corpus/spider_sql_rag.json \
        --out    models/sar_sql
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.sar.sar_model import SchemaAwareModel
from src.device import get_device


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

def _cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


class EmbeddingCache:
    def __init__(self, path: str):
        self.path  = path
        self.cache: Dict[str, np.ndarray] = {}
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.cache = pickle.load(f)

    def get(self, text: str) -> Optional[np.ndarray]:
        return self.cache.get(_cache_key(text))

    def set(self, text: str, emb: np.ndarray):
        self.cache[_cache_key(text)] = emb

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.cache, f)


def encode_with_cache(
    texts: List[str],
    flag_model,
    cache: EmbeddingCache,
    batch_size: int = 64,
) -> np.ndarray:
    results = []
    to_encode_idx, to_encode_txt = [], []

    for i, t in enumerate(texts):
        cached = cache.get(t)
        if cached is not None:
            results.append((i, cached))
        else:
            to_encode_idx.append(i)
            to_encode_txt.append(t)

    if to_encode_txt:
        for b in range(0, len(to_encode_txt), batch_size):
            batch = to_encode_txt[b : b + batch_size]
            embs  = flag_model.encode(batch)
            for j, emb in enumerate(embs):
                idx = to_encode_idx[b + j]
                cache.set(to_encode_txt[b + j], emb)
                results.append((idx, emb))
        cache.save()

    results.sort(key=lambda x: x[0])
    return np.array([r[1] for r in results])


# ---------------------------------------------------------------------------
# Structural type helpers
# ---------------------------------------------------------------------------

def struct_type_key(st: dict) -> tuple:
    return (
        st.get("num_joins",    0),
        st.get("num_tables",   0),
        st.get("has_group_by", False),
        st.get("has_order_by", False),
        st.get("has_having",   False),
        st.get("has_subquery", False),
        st.get("has_set_op",   False),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SARDataset(Dataset):
    """
    Each item: (anchor, positive, negative) triplet based on structural type.
    Anchor and positive share the same structural_type key; negative differs.
    """

    def __init__(self, corpus: list, question_embeds: np.ndarray, schema_embeds: np.ndarray):
        self.corpus          = corpus
        self.question_embeds = question_embeds
        self.schema_embeds   = schema_embeds

        # Group indices by structural type
        type_to_indices: Dict[tuple, List[int]] = {}
        for i, item in enumerate(corpus):
            k = struct_type_key(item.get("structural_type", {}))
            type_to_indices.setdefault(k, []).append(i)

        self.type_to_indices = {k: v for k, v in type_to_indices.items() if len(v) >= 2}
        self.all_keys        = list(self.type_to_indices.keys())
        self.other_keys      = {k: [ok for ok in self.all_keys if ok != k] for k in self.all_keys}

        # Build triplets
        import random
        self.triplets: List[Tuple[int, int, int]] = []
        for k, indices in self.type_to_indices.items():
            if not self.other_keys.get(k):
                continue
            for i, anc_idx in enumerate(indices):
                pos_idx = random.choice([x for x in indices if x != anc_idx])
                neg_key = random.choice(self.other_keys[k])
                neg_idx = random.choice(self.type_to_indices[neg_key])
                self.triplets.append((anc_idx, pos_idx, neg_idx))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, i):
        a, p, n = self.triplets[i]
        return (
            torch.tensor(self.question_embeds[a], dtype=torch.float32),
            torch.tensor(self.question_embeds[p], dtype=torch.float32),
            torch.tensor(self.question_embeds[n], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_sar(
    corpus_path: str,
    output_dir: str,
    cache_path: str = "Data/sar_emb_cache.pkl",
    bge_model:  str = "BAAI/bge-large-en-v1.5",
    embed_dim:  int = 1024,
    epochs:     int = 10,
    batch_size: int = 32,
    lr:         float = 1e-4,
    margin:     float = 0.3,
):
    from FlagEmbedding import FlagModel

    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"Corpus: {len(corpus)} entries")

    flag_model = FlagModel(bge_model, use_fp16=(device == "cuda"))
    cache      = EmbeddingCache(cache_path)

    questions = [item["question"] for item in corpus]
    print("Encoding questions ...")
    q_embs = encode_with_cache(questions, flag_model, cache)

    # For schema embeddings we use just the question for now;
    # the SchemaAwareModel will enrich at forward pass
    s_embs = q_embs.copy()

    dataset    = SARDataset(corpus, q_embs, s_embs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    sar_model = SchemaAwareModel(embed_dim=embed_dim).to(device)
    optimizer = torch.optim.AdamW(sar_model.parameters(), lr=lr)
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    for epoch in range(epochs):
        sar_model.train()
        total_loss = 0.0

        for anc, pos, neg in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
            optimizer.zero_grad()

            # Phase 12A will supply real table/column tensors; use zero-filled
            # dummies with all-masked padding so Stage 2 cross-attention is
            # bypassed and only question_proj + output_proj receive gradients.
            B, D = anc.shape
            dt = torch.zeros(B, 1, D, device=device)
            dc = torch.zeros(B, 1, 1, D, device=device)
            tm = torch.zeros(B, 1, dtype=torch.bool, device=device)
            cm = torch.zeros(B, 1, 1, dtype=torch.bool, device=device)

            anc_out = F.normalize(sar_model(anc, dt, dc, tm, cm), dim=-1)
            pos_out = F.normalize(sar_model(pos, dt, dc, tm, cm), dim=-1)
            neg_out = F.normalize(sar_model(neg, dt, dc, tm, cm), dim=-1)

            loss = triplet_loss(anc_out, pos_out, neg_out)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} avg loss: {avg:.4f}")

    torch.save(sar_model.state_dict(), os.path.join(output_dir, "sar_model.pt"))
    print(f"SAR model saved to {output_dir}/sar_model.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus",    required=True, help="spider_sql_rag.json")
    parser.add_argument("--out",       default="models/sar")
    parser.add_argument("--bge",       default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--epochs",    type=int,   default=10)
    parser.add_argument("--batch",     type=int,   default=32)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--margin",    type=float, default=0.3)
    args = parser.parse_args()

    train_sar(
        corpus_path=args.corpus,
        output_dir=args.out,
        bge_model=args.bge,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()

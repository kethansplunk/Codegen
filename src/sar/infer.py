"""
SAR (Schema-Aware Retriever) inference — retrieve top-k similar examples.
Adapted from SchemaRAG SAR_use.py:
- Loads trained SchemaAwareModel weights.
- Encodes the question using BGE, computes schema-aware embedding.
- Retrieves top-k corpus entries by cosine similarity.
- Returns (question, sql/mql, structural_type) for use as few-shot examples.
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

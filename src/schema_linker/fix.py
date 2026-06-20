"""
SchemaLinker post-prediction fix using embedding similarity.
Adapted from SchemaRAG SchemaLinker_fix.py:
- Removed hardcoded paths; takes schema_text and predictions as arguments.
- FlagModel path is passed in (set to BAAI/bge-large-en-v1.5 by default).
- Uses our format_schema_text output (same # Table: / (col:type) format).
- MPS / CUDA / CPU device support.

Purpose: After SchemaLinker predicts ["actor.nationality"], this snaps each
prediction to the nearest real "table.column" pair in the database using cosine
similarity on BGE embeddings, correcting hallucinated column names.
"""

from __future__ import annotations

import re
from typing import List, Set

import torch
import torch.nn.functional as F


def extract_table_columns_from_schema(schema_text: str) -> List[str]:
    """
    Parse our schema text format and return all 'table.column' pairs.

    Expected input format (from build_cot_data.format_schema_text):
        # Table: actor
        [(actor_id:INT, Primary Key, Examples: [1, 2]),
         (name:TEXT, Examples: [Tom Hanks]),
        ]
    """
    result: List[str] = []
    current_table: str | None = None
    for line in schema_text.splitlines():
        line = line.strip()
        if line.startswith("# Table:"):
            current_table = line[len("# Table:"):].strip()
        elif current_table and line.startswith("(") and ":" in line:
            col_name = line[1 : line.find(":")].strip()
            result.append(f"{current_table}.{col_name}")
    return result


def fix_schema_links(
    predicted_links: List[str],
    schema_text: str,
    model_name: str = "BAAI/bge-large-en-v1.5",
) -> List[str]:
    """
    Snap each predicted schema link to the nearest real table.column in the
    database using BGE cosine similarity.

    Args:
        predicted_links: Raw predictions from SchemaLinker, e.g.
                         ["actor.nationality", "movie.tilte"].
        schema_text:     Full schema string from format_schema_text().
        model_name:      BGE model name or local path.

    Returns:
        Fixed predictions where each entry is a real table.column pair.
    """
    from FlagEmbedding import FlagModel  # lazy import — only needed at inference

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    real_columns = extract_table_columns_from_schema(schema_text)
    if not real_columns or not predicted_links:
        return predicted_links

    flag_model = FlagModel(model_name, use_fp16=(device != "cpu"))

    db_embeds   = torch.tensor(flag_model.encode(real_columns),    dtype=torch.float32)
    pred_embeds = torch.tensor(flag_model.encode(predicted_links), dtype=torch.float32)

    db_embeds   = F.normalize(db_embeds,   p=2, dim=1)
    pred_embeds = F.normalize(pred_embeds, p=2, dim=1)

    sim_matrix  = torch.matmul(pred_embeds, db_embeds.T)       # (pred_len, db_len)
    max_indices = torch.argmax(sim_matrix, dim=1).tolist()

    return [real_columns[idx] for idx in max_indices]

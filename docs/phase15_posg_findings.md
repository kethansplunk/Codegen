# Phase 15A — POSG Findings (SQL track)

## Summary

`scripts/run_posg_sql.py` wires the previously-disconnected Phase 12–14 pieces
(SAR retrieval → Generator, 5 candidates → `ParetoOptimal` selection → EX
comparison) into an actual pipeline and validates whether POSG selection adds
value over taking the Generator's first sampled candidate.

**Result: POSG measurably helps.** On a 30-question sample drawn from Spider's
full dev split (unseen by the Generator), filtered to multi-join/subquery/
GROUP BY questions and spread across ~20 different databases:

| | exact-match | EX |
|---|---|---|
| POSG | 20.0% (6/30) | **63.3% (19/30)** |
| Greedy (candidate 0 only) | 16.7% (5/30) | 60.0% (18/30) |

POSG diverged from greedy on 6/30 examples. In every one of those 6, POSG's
pick was equal-or-better than greedy's by EX — **never worse** in any test run
during this investigation.

## Methodology — two false starts before a valid test

1. **Train-split smoke tests gave 80–100% exact-match and looked great, but were invalid.**
   `Data/cot_data/sql_cot_train.json` is data the Generator was fine-tuned on.
   `unique_candidates` (out of 5 sampled) was almost always 1, even on
   questions filtered for structural difficulty (JOIN/subquery/GROUP BY) —
   the model was recalling memorized answers, not reasoning, so there was no
   genuine sampling uncertainty for POSG to arbitrate. **Difficulty of the SQL
   was the wrong variable to control for; whether the model had seen the
   question before was the actual confound.**

2. **A small (10-example) dev-split sample looked diverse but wasn't.**
   Built via `scripts/build_dev_eval_set.py` (runs the real DeepSeek-backed
   SchemaLinker on Spider's dev split — data never trained on). Exact-match
   dropped to 60–70%, confirming the train numbers were memorization-inflated.
   But once `Data/Spider/database/*.sqlite` was obtained and EX scoring
   actually ran, **all 10 examples scored 100% EX** — because the `--limit 30`
   dev-set build happened to concentrate almost entirely on one database
   (`concert_singer`, Spider's simplest, most famous example schema).
   POSG never diverged from greedy in this sample either — not because it
   doesn't work, but because greedy was already perfect, leaving nothing to
   improve.

3. **The valid test: full dev-set build (1034 questions, all ~20 databases), n=30, `--hard` filter.**
   This is the result summarized above. `pareto_front_size > 1` occurred in
   26/30 examples here (vs. exactly 5/5 in every earlier test) — evidence
   that POSG's scoring dimensions genuinely discriminate between candidates
   once real quality differences exist between them, not just cosmetic ones.

## The 6 divergence examples, categorized

- **Genuine correctness fix (1 case)** — greedy executed but returned the
  wrong result (joined through an incorrect intermediate table); POSG picked
  a simpler candidate matching gold's actual join structure. `EX greedy=0.0
  → EX posg=1.0`.
- **Crash avoidance (2 cases)** — greedy's SQL had a syntax error / failed to
  execute (`executability=0.0`); `ParetoOptimal.find_pareto_optimal` filters
  to only executable candidates before comparing anything else, so POSG
  correctly excluded the broken candidate. Final EX was still 0 in both
  (the surviving candidate ran but wasn't fully correct either), but this is
  a real robustness improvement — a single-sample pipeline would have
  returned a hard error instead of at least a plausible answer.
- **Equally-correct, better-phrased (3 cases)** — both picks were EX=1.0, but
  POSG's choice scored measurably higher on `schema_conformity`/
  `example_consistency` (not tied — e.g. one case was `1.0/1.0` vs.
  `0.6/0.944`) and more closely matched gold's phrasing (`DISTINCT`, `NOT IN`
  vs. `EXCEPT`).

## Root cause of the known blind spot (confirmed, not yet fixed)

`ParetoOptimal._extract_schema_from_sql` (`src/posg/posg_sql.py`) extracts a
**`set()`** of identifier tokens from SQL text via regex. A Python set has no
concept of order or binding, so `stadium AS T1 JOIN concert AS T2` and
`concert AS T1 JOIN stadium AS T2` produce the *identical* token set — the
metric cannot tell which alias is bound to which physical table. This is
purely a text-parsing limitation in `posg_sql.py` itself; it has **no
relation to SAR's cross-attention/schema-fusion gap** (that gap affects
retrieval quality before generation; this affects scoring of already-generated
SQL text, a separate, later stage that never touches SAR's embeddings).

In small/homogeneous samples (train data, or the single-database dev sample),
this blind spot was the dominant effect and made POSG look like a no-op —
candidates only ever differed by alias/case, which the metric can't see, so
scores tied and the tie-break (`if score > best_score`, strict `>`, defaults
to whichever candidate was evaluated first) always landed back on greedy.
On the large diverse sample, real correctness/executability differences
dominated instead, and the metric worked as designed.

**Not fixed, low priority given the evidence above**: if candidates ever
differ *only* by a semantically-meaningful alias reassignment (e.g. grouping
by the wrong table's key after a swap), the metric still can't catch it.
Fix would be alias-resolution in `_extract_schema_from_sql` before extracting
the token set — deferred, since no test run has yet shown it costing a wrong
answer in practice.

## Caveats on the headline number

- **63.3% EX is on the `--hard`-filtered subset only** (JOIN/subquery/GROUP BY
  questions) — not overall system accuracy. Easy single-table questions were
  consistently ~100% correct in every test. Don't conflate this with the
  plan's >82% EX target, which is presumably an all-difficulty average.
- Sample size is 30, all from Spider dev — a larger run would tighten the
  confidence interval on the EX numbers above.
- `key_fields` in `Data/cot_data/sql_dev_eval_full.json` came from the live
  DeepSeek SchemaLinker (API mode), not oracle labels — this is the correct,
  realistic setup (matches what Phase 16's assembled pipeline will actually
  see), but means SchemaLinker errors are a possible contributor to the
  remaining wrong answers, not yet isolated from Generator/POSG error.

## Environment notes (useful if re-running this)

- `sar.backend` in `configs/config.yaml` was switched from `chroma` to
  `memory` — ChromaDB's `PersistentClient` cannot open an index over a
  Google Drive FUSE mount (`InternalError: File exists (os error 17)`),
  and the Phase 13 Chroma index turned out to be saved under Drive's
  "Computers" backup section, which `google.colab.drive.mount()` cannot see
  at all (only "My Drive" and Shared Drives are reachable). `memory` backend
  re-encodes the 7000-entry SQL RAG corpus at startup in ~1 second on GPU —
  negligible cost at this scale, so there was no need to chase the Chroma/
  Drive fix.
- `Data/Spider/database/*.sqlite` (166 databases) is not in git (see
  `.gitignore`) and was not present on any machine used mid-investigation;
  it was recovered from a sibling local project (`Codegen_practice/data/spider`)
  and manually uploaded to Drive for Colab to use.

## Next steps

- Phase 15B: repeat this same validation methodology for `posg_nosql.py` /
  the NoSQL generator, to check whether the same pattern (no-op on easy/
  memorized data, real value on diverse hard data) generalizes.
- Optionally fix the alias-resolution gap in `_extract_schema_from_sql` if a
  future, larger test run surfaces a case where it actually costs a wrong
  answer.
- Optionally re-run without `--hard` on a large sample to get an EX number
  comparable to the plan's >82% target.

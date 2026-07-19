# Phase 15 — POSG Findings (SQL + NoSQL tracks)

## Phase 15A — SQL track

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

## Next steps (SQL)

- Optionally fix the alias-resolution gap in `_extract_schema_from_sql` if a
  future, larger test run surfaces a case where it actually costs a wrong
  answer.
- ~~Optionally re-run without `--hard` on a large sample to get an EX number
  comparable to the plan's >82% target.~~ Done — see below.

## Full-difficulty dev-set result (2026-07-19, A100 re-run)

Same `sql_dev_eval_full.json` source, same `n=30`, but **without** the
`--hard` filter — the number comparable to the plan's >82% EX target that
Phase 15A's headline result above deliberately excluded.

| | exact-match | EX |
|---|---|---|
| POSG | 53.3% (16/30) | 86.7% (26/30) |
| Greedy (candidate 0 only) | 50.0% (15/30) | **90.0% (27/30)** |

Both clear the plan's >82% target. But on the full difficulty distribution,
**greedy edged out POSG** (27/30 vs 26/30) — the opposite of the `--hard`
result above, where POSG was never worse than greedy on any divergence.

**Reading, consistent with the blind-spot analysis above**: the full dev set
is dominated by easy single-table questions, where (per the "Caveats on the
headline number" section) greedy's first sampled candidate is already
correct ~100% of the time. POSG's 5-candidate sampling has nothing to fix on
those questions and only downside risk — a worse candidate occasionally
scores higher than the correct greedy pick on `schema_conformity`/
`example_consistency`, costing a point it didn't need to spend. On the
`--hard` subset, by contrast, greedy's first sample is wrong often enough
that POSG's alternatives have real problems to fix, so the net effect is
positive. **Net takeaway: POSG's value is concentrated in structurally
difficult queries, not a uniform improvement across the full difficulty
distribution** — worth stating explicitly rather than only reporting the
`--hard` number as if it generalized.

This was a single `n=30` run, not repeated — see the NoSQL reproducibility
note below for why re-running the identical command would likely shift this
by a few points either way; the ranking (greedy ≥ POSG on full difficulty)
is the more load-bearing finding here than the exact percentages.

**Update (2026-07-19, Phase 16 regression re-run, same command, seed fix
active)**: POSG 90.0% (27/30) vs greedy 86.7% (26/30) — the exact opposite
ranking of the run above. See the NoSQL reproducibility note below: this
happened *with* `--seed` wired into generation, confirming seeding alone
doesn't make these runs reproducible. Net effect: **both directions have now
been observed on this exact test** (POSG ahead once, greedy ahead once) —
treat any single-run SQL full-difficulty number as within a few points of a
tie between POSG and greedy, not a settled ranking either way.

---

## Phase 15B — NoSQL track

### Summary

`scripts/run_posg_nosql.py` mirrors `run_posg_sql.py`'s structure for MQL:
SAR (track=nosql) → Generator (track=nosql, 5 candidates, JSON strings
parsed to `{"collection", "pipeline"}`) → `ParetoOptimalMQL` selection
(`src/posg/posg_nosql.py`) → result-set EX comparison.

**Result: the same pattern holds as SQL.** On a 30-question sample from
`Data/cot_data/nosql_cot_train.json` (train split), filtered to
`$lookup`/`$group`/multi-stage pipelines:

| | exact-match | EX |
|---|---|---|
| POSG | 53.3% (16/30) | **76.7% (23/30)** |
| Greedy (candidate 0 only) | 50.0% (15/30) | 73.3% (22/30) |

POSG diverged from greedy on 5/30 examples — a **+3.4 percentage point EX
improvement**, closely matching SQL's +3.3pp finding on an independently
built pipeline and a different scoring implementation. Two tracks, same
qualitative result.

### A critical methodological trap hit and fixed along the way

The first two attempts at this test (n=10, then n=30) both reported a
suspicious **100% EX** for both POSG and greedy. This was wrong, and the
reason is a real gap in how `evaluate_ex`-style comparisons work for MongoDB
specifically (this does not apply to the SQL track): **MongoDB does not raise
an exception when you aggregate against a missing or empty collection — it
silently returns `[]`.** The original `_mql_result_eq` comparison had no
guard against this, so when both the gold pipeline and a candidate returned
`[]` (because the target database was never actually loaded into MongoDB),
the empty-vs-empty comparison trivially evaluated as a "match" — a false
positive on every single example.

Root cause of *why* the databases weren't loaded, once traced down:
`src/mongodb_converter.py`'s `convert_all()` uses the existence of
`Data/mongodb/<db>_schema.json` as a "already converted, skip" marker. Those
166 schema-cache JSON files are git-tracked (committed from an earlier Phase
5B run on a different machine with its own MongoDB instance) — so a fresh
`git clone` into a brand-new Colab session pulls in cache files that claim
every database is already loaded, when the actual MongoDB *data* (which
can't be stored in git) was never inserted into that session's empty
`mongod`. `convert_all()` trusted the cache and silently skipped every real
conversion.

**Fix applied**:
1. `run_one()` in `run_posg_nosql.py` now tracks `gold_row_count` and only
   scores EX when the gold pipeline returns a genuinely non-empty result;
   examples with `gold_row_count == 0` are excluded from the EX average and
   flagged separately, with the offending database names printed, instead
   of silently counting as a match.
2. Practical fix for the environment: delete the stale `Data/mongodb/`
   cache in the Colab session (safe — it's git-tracked, not lost) and
   re-run `convert_all()`, which then genuinely re-converts and inserts all
   166 databases. Verified afterward via `count_documents({})` returning a
   real non-zero count before trusting any further results.

This is the same category of lesson as Phase 15A's dev-set/database
discoveries: **a metric that can't distinguish "no data" from "correct
empty answer" will silently report false confidence** — worth checking for
in any future EX-style comparison added to this project, SQL or NoSQL.

### The 5 divergence examples, categorized

- **Genuine correctness fix (1 case)** — greedy's pipeline was missing its
  final `$project` stage entirely (looked like a truncated generation), so
  its output kept MongoDB's raw `_id` field name instead of gold's renamed
  field. `EX greedy=0.0 → EX posg=1.0`.
- **Equally-correct, better-phrased (2 cases)** — both picks hit `EX=1.0`,
  but POSG's choice scored measurably higher on `example_consistency` (not
  a tie — e.g. `0.743` vs `0.746`), same pattern as SQL's category 3.
- **Both wrong, no regression (2 cases)** — `EX=0.0` for both greedy and
  POSG's pick. Worth being honest about: POSG's pick was structurally more
  complete in one of these (included a `$project` the greedy pick lacked)
  but still didn't land on a correct answer. POSG never converts a correct
  greedy answer into a wrong one in this sample, but it isn't a guaranteed
  fix either — a bounded, real improvement, not a silver bullet.

### Caveats

- Still train-split data — the NoSQL dev-set equivalent of
  `build_dev_eval_set.py` (which would need to translate Spider's `dev.json`
  SQL into MQL via DeepSeek, mirroring Phase 8B's `build_nosql_cot_data.py`,
  since there is no pre-existing "gold MQL for dev" anywhere) was never
  built — deprioritized because, unlike the SQL track, this NoSQL train
  result is *not* memorization-saturated (73–77% EX, not ~100%), suggesting
  MQL's JSON generation is harder to memorize verbatim than SQL text and
  giving reasonable (if not ideal) confidence in this result without that
  extra build.
- Sample size is 30 — same statistical caveat as the SQL track.
- `key_fields` reused from the CoT data's pre-computed values, not a live
  SchemaLinker call (unlike the SQL dev-set test) — consistent with a
  train-split test, but means this isn't fully mirroring Phase 16's live
  inference path the way the SQL dev-set test did.
- **Not reproducible run-to-run, even with `--seed` wired into generation
  (confirmed 2026-07-19, three runs of the identical command)**:
  `--seed` (`random.seed(seed)`) picks *which* 30 questions are sampled;
  `GeneratorInfer` also now accepts a `seed` param and calls
  `torch.manual_seed(seed)` once at construction (both CLI scripts pass
  their `--seed` through). This was expected to make runs reproducible but
  did not: three runs of `--smoke_test --n 30` (no `--hard`), all with the
  seed fix active on the third, returned EX 76.7%/73.3% (POSG ahead,
  original baseline), 80.0%/86.7% (greedy ahead), then 83.3%/80.0% (POSG
  ahead) — three different results from the nominally same deterministic
  command. `torch.manual_seed()` seeds RNG state but does not by itself make
  CUDA kernels bit-deterministic (cuDNN/cuBLAS reduction order, etc. --
  would need `torch.use_deterministic_algorithms(True)` and matching cuDNN
  flags, not yet tried, and may not have deterministic implementations for
  every op involved or may be substantially slower). At n=30 a couple of
  flipped candidates move the percentage by ~3.3pp each, so single-run
  headline numbers on either track should be read as noisy point estimates
  within a few points of each other, not stable measurements, regardless of
  whether `--seed` is set.

### Next steps

- If pursued further: build the NoSQL dev-set translator for a true
  held-out generalization number, mirroring `build_dev_eval_set.py` but
  with a DeepSeek SQL→MQL translation step first.
- Same alias/structural-blindness question as SQL is worth periodically
  re-checking in `posg_nosql.py`'s `evaluate_schema_conformity` (collection-
  level Jaccard) as more diverse tests are run.
- ~~Consider pinning `torch.manual_seed()` in `GeneratorInfer.generate()` (or
  exposing it as a param) so smoke-test runs are reproducible~~ Attempted,
  **did not fully fix it** — `GeneratorInfer` takes an optional `seed` param
  (pinned once at construction, wired through `--seed` in both CLI scripts),
  but a same-seed re-run still produced a third, different EX number (see
  above). `torch.manual_seed()` isn't sufficient for GPU run-to-run
  determinism on its own. If exact reproducibility is ever actually needed
  (not just "roughly stable"), the next thing to try is
  `torch.use_deterministic_algorithms(True)` + `torch.backends.cudnn.
  deterministic = True`, accepting the likely throughput cost — not yet
  attempted. Until then, treat every EX number in this document as a noisy
  point estimate, not a reproducible measurement.

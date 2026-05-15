# AGENTS.md — LLMToys: Natural Language -> SQL Pipeline

## Project Overview

This project implements multiple tools that use different LLMs for different purposes. The current scope is:

- **Natural Language -> SQL (NL2SQL)**: a text-to-SQL pipeline for Spanish-language business analytics queries against a multi-tenant SQL Server / PostgreSQL schema. The pipeline has three autonomous stages:

```
run_semantic_schema_pruning.py  ->  out/semantic_pruned_schema.yaml
run_semantic_resolver.py        ->  out/semantic_plan.yaml
run_sql_solver.py               ->  out/solver_result.sql / solver_result.yaml
```

Each stage is a self-contained module (`semantic_prune/`, `semantic_resolver/`, `sql_solver_generator/`) that communicates only through YAML artifacts. Cross-imports between stages are not allowed.

---

## Target Hardware: MSI Raider GE66 12UHS

All model configurations, VRAM budgets, and CPU-offload defaults are calibrated for this machine:

| Component  | Spec                                                         |
| ---------- | ------------------------------------------------------------ |
| **CPU**    | Intel Core i9-12900HX (16 cores / 24 threads, Alder Lake-HX) |
| **GPU**    | NVIDIA GeForce RTX 3080 Ti Laptop — **16 GiB GDDR6**         |
| **RAM**    | 64 GiB DDR5                                                  |
| **CUDA**   | 13.0 (PyTorch index `pytorch-cu130`)                         |
| **Python** | 3.12-3.13 (requires `>=3.12,<3.14.1`)                        |

### Model Sizing Guidelines For This GPU

| Model                      | Params          | Precision        | VRAM usage                                 | Extra flags required                                                                                                                                                                               |
| -------------------------- | --------------- | ---------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| E2Rank-0.6B                | 0.6B            | bf16             | ~1.2 GiB                                   | -                                                                                                                                                                                                  |
| Qwen3-Embedding-0.6B       | 0.6B            | bf16             | ~1.2 GiB                                   | sequential_engines=True                                                                                                                                                                            |
| Qwen3-Reranker-0.6B        | 0.6B            | bf16             | ~1.2 GiB                                   | sequential_engines=True                                                                                                                                                                            |
| XiYanSQL-QwenCoder-7B-2504 | 7B              | bf16             | ~11.2 GiB                                  | enforce_eager=True, cpu_offload_gb=3.0, gpu_memory_utilization=0.90, max_model_len=2048, max_tokens=384, stop=["### END_OF_OUTPUT"]                                                                |
| Gemma-4-E4B                | 4B-class        | AWQ 4-bit / fp16 | ~10.5 GiB storage / ~18 GiB bf16 effective | use `Chunity/gemma-4-E4B-it-AWQ-4bit`, dtype=float16, enforce_eager=True, cpu_offload_gb=0, gpu_memory_utilization=0.82, max_model_len=2048, block_size=64, max_num_seqs=1, async_scheduling=False |
| Qwen3-30B-A3B (MoE)        | 30B / 3B active | bf16             | fits with AUTO_CPU_OFFLOAD                 | vllm_config_qwen36.py                                                                                                                                                                              |
| Gemma-4-26B-A4B (MoE)      | 26B / 4B active | AWQ 4-bit / bf16 | ~16.0 GiB AWQ / ~52 GiB bf16               | use `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`, `quantization="compressed-tensors"`, `dtype=float16`; bf16 still falls back to AUTO_CPU_OFFLOAD in `vllm_config_gemma4.py`                             |

**Do not** introduce models that exceed these constraints without adjusting `cpu_offload_gb`, `gpu_memory_utilization`, `max_model_len`, and `enforce_eager` accordingly. Prefer 0.6B-7B models for pipeline stages; larger MoE models are for `run_model.py` only.

---

## Runner / Executable Conventions

### No argparse - ever

Runner scripts (`run_*.py`) **must not use `argparse`** or any other CLI argument parsing library (`click`, `typer`, `fire`, etc.).

Configuration is provided exclusively via:

1. **Environment variables** loaded with `python-dotenv` (`from dotenv import load_dotenv; load_dotenv()`)
2. **Module-level constants** (ALL_CAPS) at the top of the file, serving as commented defaults
3. **YAML / JSON artifacts** produced by previous pipeline stages

Correct example pattern:

```python
load_dotenv()
SEMANTIC_QUERY: str = os.getenv("SEMANTIC_QUERY", "what is the average number of active records per entity?")
DB_SCHEMA_PATH: str = os.getenv("DB_SCHEMA_PATH", "schema-docs/db_schema.yaml")
```

All runners must expose a `main()` function and use `if __name__ == "__main__": main()`.

---

## Architecture

### Stage 1 - `semantic_prune/`

Schema pruning with `Alibaba-NLP/E2Rank-0.6B`. It embeds the natural-language query and all schema elements, retrieves top-K candidates via cosine similarity, and then reranks them with E2Rank's listwise pseudo-query technique.

- Config: `semantic_prune/config.py` (`GPU_MEM_UTIL=0.30`, `MAX_MODEL_LEN=30464`)
- Output: `out/semantic_pruned_schema.yaml` -> `{query: ..., pruned_schema: {...}}`

### Stage 2 - `semantic_resolver/`

Semantic plan compilation with `Qwen3-Embedding-0.6B` (retrieval) and `Qwen3-Reranker-0.6B` (binary yes/no scoring via `allowed_token_ids`). It builds join paths, detects `GROUP BY` columns, and validates semantic compatibility.

- Config: `semantic_resolver/config.py` (`GPU_MEM_UTIL=0.30` for embedding / `0.25` for reranking)
- Embedding and reranker engines must be loaded sequentially (`sequential_engines=True`) to avoid EngineCore deadlock on 16 GiB VRAM
- Output: `out/semantic_plan.yaml` -> `{semantic_plan: {retrieved_candidates: [...], compiled_plan: {...}}}`

### Stage 3 - `sql_solver_generator/`

SQL generation using the vLLM-served `XiYanSQL-QwenCoder-7B-2504`. `SQLQuerySpec` is compiled deterministically from `SemanticPlan`; the LLM must emit only `final_sql` in YAML, which is then validated and normalized with `sqlglot`.

- Config: `sql_solver_generator/config.py` (`GPU_MEM_UTIL=0.90`, `MAX_MODEL_LEN=2048`, `MAX_TOKENS=384`, `TEMPERATURE=0.0`, `CPU_OFFLOAD_GB=3.0`, `ENFORCE_EAGER=True`)
- Compact prompt payloads are required: the full `retrieved_candidates` + `pruned_schema` + `semantic_rules` payload causes context overflow. Use `_compact_*` helpers, run token preflight checks, and use `### END_OF_OUTPUT` as the stop sequence.
- Output: `out/solver_result.sql` + `out/solver_result.yaml`

### `llm_core/`

Centralized vLLM infrastructure. `VLLMDefaults` (`dtype=bfloat16`, `AUTO_CPU_OFFLOAD=True`, `AUTO_QUANTIZE=True` -> AWQ 4-bit fallback when VRAM is insufficient). Concrete runners: `Gemma4Runner`, `Gemma4E4BRunner`, `Qwen3Runner`.

---

## Build & Test

```bash
# Install (uv)
uv sync

# Run tests
uv run python -m pytest tests/ -v

# Run the full pipeline (edit .env or the module-level constants in each runner first)
uv run run_semantic_schema_pruning.py
uv run run_semantic_resolver.py
uv run run_sql_solver.py
```

Use the repo-local virtual environment with:

```bash
source .venv/bin/activate
```

---

## Code Conventions

| Convention               | Detail                                                                                                                                                                                                                                                                                                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Encoding header**      | `#!/usr/bin/env python3` + `# -*- coding: utf-8 -*-` on all executables                                                                                                                                                                                                                                                                                            |
| **Annotations**          | `from __future__ import annotations` in every module                                                                                                                                                                                                                                                                                                               |
| **Constants**            | ALL_CAPS at module top                                                                                                                                                                                                                                                                                                                                             |
| **Docstrings**           | All generated code must be fully documented in Spanish: class docstrings, method/function docstrings, inline comments in the body, and file-level headers. No undocumented classes, methods, or non-trivial logic blocks.                                                                                                                                          |
| **Document language**    | This document, and any future extensions to it, must remain in English.                                                                                                                                                                                                                                                                                            |
| **Config**               | Each module owns a `config.py`; no global singleton                                                                                                                                                                                                                                                                                                                |
| **Autonomy**             | `semantic_prune`, `semantic_resolver`, and `sql_solver_generator` must remain fully independent - no cross-imports                                                                                                                                                                                                                                                 |
| **Formatter**            | `black` (>=26.x)                                                                                                                                                                                                                                                                                                                                                   |
| **External config**      | Business rules, semantic rules, and prompts must live in YAML assets - never hardcode them                                                                                                                                                                                                                                                                         |
| **Heuristic config**     | NL2SQL technical heuristics such as token vocabularies, weighted token tables, ranking-dimension preferences, stopword lists, regex fragments, prompt-compaction caps, repair weights, per-kind caps, binary yes/no token candidates, chat suffixes, and similar resolver/solver/prune tuning must live in `nl2sql/config/settings.yaml`, not hardcoded in Python. |
| **No legacy residue**    | By default, every implemented request must leave touched code without legacy residue, backward-compatibility shims, or dead code. Keep such code only if the user explicitly asks for it or a documented repository constraint requires it.                                                                                                                        |
| **Directive conflicts**  | If a user request contradicts this document, the agent must say so explicitly, point out the conflicting directive, and proceed only once that exception is clear.                                                                                                                                                                                                 |
| **No business literals** | Versioned source code must not hardcode business-specific identifiers; inject them from `schema-docs/*` in runtime code or `tests/fixtures/*` in tests.                                                                                                                                                                                                            |
| **No argparse**          | See [Runner Conventions](#runner--executable-conventions)                                                                                                                                                                                                                                                                                                          |

---

## Schema & Semantic Files

| File                                  | Purpose                                                |
| ------------------------------------- | ------------------------------------------------------ |
| `schema-docs/db_schema.yaml`          | Full DB schema (tables, columns, PKs, FKs, types)      |
| `schema-docs/semantic_rules.yaml`     | Domain metrics, dimensions, synonyms, time expressions |
| `schema-docs/catalogos-embebidos.yml` | Embedded catalog values for filters                    |
| `out/semantic_pruned_schema.yaml`     | Stage 1 output - consumed by Stage 2                   |
| `out/semantic_plan.yaml`              | Stage 2 output - consumed by Stage 3                   |

---

## Known Constraints & Gotchas

- **XiYanSQL 7B on 16 GiB**: requires `enforce_eager=True` (prevents `torch.compile` / inductor OOM) + `cpu_offload_gb=3.0`. Without `enforce_eager`, allocation during autotuning kills the process.
- **Embedding / pooling runtimes on vLLM 0.21+**: `semantic_prune` and `semantic_resolver` must run with `enforce_eager=True`; for short-lived engines on 16 GiB VRAM, `torch.compile` + cudagraph profiling reduces effective KV cache capacity and can prevent startup even when the model fits in eager mode.
- **Sequential engines**: load the embedding engine -> run -> destroy -> load the reranker engine. Do not hold both in VRAM at the same time.
- **`BatchEncoding` from `tokenizer.apply_chat_template(tokenize=True)`**: this returns `BatchEncoding`, not a list. Extract `input_ids` before constructing `TokensPrompt`.
- **CTE table names in SQL validation**: the AST validator must ignore CTE aliases when checking `sql_uses_unknown_table`; otherwise it raises false positives on CTE names such as `base`.
- **`final_group_by` for scalar metrics**: leave it empty for scalar aggregates (for example, `metric_avg_per_entity`). Populate it only when the outer query also groups.
- **Prompt patching for the window-over-aggregate bug**: the prompt must explicitly forbid `AVG(COUNT(...)) OVER (...)` constructs and require a CTE / subquery with `GROUP BY` instead.

## Policy For New NL2SQL Heuristics

A fix belongs in code only if it satisfies at least one of these conditions:

- it protects safe SQL execution or prevents a general class of runtime failures;
- it normalizes artifact contracts or removes fragile parsing / manual mapping;
- it validates syntax, AST shape, dialect behavior, or context / VRAM limits;
- it implements a general rule documented in YAML or in a structured contract.

If a fix responds to a specific business phrase, a specific metric, or a single user example, it must go first into `schema-docs/semantic_rules.yaml` as a declarative rule, curated example, or semantic signal. Python code must not become a repository of business aliases or a parallel deterministic SQL renderer next to the solver.

In addition, versioned source code must not contain concrete business identifiers. Any domain-specific table name, metric name, join path, or external system name must be injected from `schema-docs/*`; tests must use generic fixtures under `tests/fixtures/*`.

If an NL2SQL tuning table / list / mapping / regex appears in Python and it is not strictly dialect syntax or library plumbing, it must be moved into `nl2sql/config/settings.yaml` in the same task. Do not leave behind "just this fallback" or "just this temporary cap" embedded in code.

## Repository Memory Summary

### `llm_core`

- The current contract is `VLLMRuntimeDefaults` + `VLLMModelRunner`; `VLLMDefaults` no longer applies.
- `vllm_engine.py` is model-agnostic; model-specific behavior lives in `model_registry.py`, runtime profiles, and `vllm_config_*` modules.
- `prompt_optimizer.py` uses the new contract and requires `llmlingua` for effective compression.
- `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` is supported again on `vllm 0.21.x` as the `QuantizedVariant` fallback for Gemma 26B; keep `quantization="compressed-tensors"`, `dtype="float16"`, and an `awq_4bit` estimate of about `16.0 GiB`.
- In `vllm 0.19.x`, `cpu_offload_gb > 0` can break initialization; the builder must force `disable_hybrid_kv_cache_manager=True` when offload is enabled.
- The final NCCL warning in `run_model.py` with Qwen3 remains open and does not have a clean confirmed workaround from `llm_core`.

### Semantic Contract

- `schema-docs/semantic_rules.yaml` centralizes prune / resolver / solver configuration under a top-level `semantic_contract` root with the groups `business_invariants`, `retrieval_heuristics`, and `sql_safety`.
- That root must also exist in in-memory fixtures / tests; the minimum shape is `{"semantic_contract": {"business_invariants": {}, "retrieval_heuristics": {}, "sql_safety": {}}}`.
- `retrieval_heuristics.semantic_synonyms` is a mapping `entity -> [synonyms]`, not a list of rows.
- `sql_safety.execution_safety` replaces the old `solver_rules.yaml`; do not reintroduce that asset.

### Unified Config And Runtime Bundle

- `nl2sql.config` is the only supported entry point for loading and validating `nl2sql/config/settings.yaml`; prune, resolver, solver, and orchestrator code must consume typed models or `NL2SQLRuntimeBundle`, not reload YAML locally.
- `NL2SQL_CONFIG_PATH` is the only supported variable for redirecting the unified configuration. Do not reintroduce legacy per-stage overrides such as `SEMANTIC_PRUNE_*_PATH`, `SEMANTIC_RESOLVER_*_PATH`, `SQL_SOLVER_*_PATH`, or `NL2SQL_NARRATIVE_PROMPT_PATH`.
- Custom YAML files used in tests, overrides, or fixtures may be partial. The central loader must deep-merge them against the canonical `settings.yaml` before validating with Pydantic; do not require the full document to be duplicated just to test one section.
- YAML 1.1 may coerce unquoted tokens such as `yes`, `no`, `on`, or `off` into booleans. Typed validators for binary vocabularies, stop tokens, and similar lists must normalize that coercion instead of breaking config loading.
- Shared orchestrator helpers must preload `runtime_bundle` while keeping public builders usable without explicit arguments in tests / runners; do not reintroduce per-stage YAML loading just to recover ergonomics.
- `SolverInput.semantic_rules` may arrive as a path, mapping, or already-validated `SemanticContract`. Do not degrade a typed contract to a dict / path only to satisfy a legacy shape.

### NL2SQL Retrieval And Pruned Schema

- Semantic prune must preserve dependencies of relevant metrics declared in rules: `source_catalog`, `required_relationships`, and `table.column` references inside `formula`.
- The SQL solver does not repair structure omitted by prune; it only normalizes inputs and validates SQL / rules.
- The resolver compiles joins from `pruned_schema` + semantic assets; it does not revalidate against `db_schema` at runtime.
- Short acronyms in plural form must still resolve to short synonyms (`EA` + `s`).
- Retrieved `semantic_examples` must inject their referenced metrics / dimensions / model so curated top-K metrics are not lost.
- If an entity declares `time_field`, it must point to the correct operational time field in its source table.

### Validation And Guardrails

- `ValidationStage` must reject any `WHERE` clause not declared in `spec.selected_filters` or `spec.time_filter`.
- It must also flag `unknown_bare_column` when SQL uses bare identifiers outside the schema / alias set.
- If the resolver leaves `compiled_plan.verification.is_semantically_aligned = false`, the solver must abort before loading prompts or rules.
- For `derived_metric`, the SQL must contain an inner CTE / subquery with `GROUP BY base_group_by`; the expected structural issues are `derived_metric_missing_grouped_subquery` or `derived_metric_missing_two_level_query`.
- `LlmRouter` must retry with a structural repair rule if that `derived_metric` issue appears.
- `plan_compiler` must merge `compiler_rules.yaml` with `semantic_rules.yaml` so Spanish heuristics are not lost.
- If the query asks for a state and the chosen measure does not encode it, the resolver must emit `unmapped_qualifier_in_question:<token>` and reduce `confidence`.

### Prompt Budgeting

- Every short-context local LLM stage must run token preflight checks before invoking vLLM and must degrade payloads by variant when needed.
- Current coverage: semantic verifier, SQL solver, and final narrative generation.
- On this RTX 3080 Ti Laptop 16 GiB machine, Gemma-4-E4B AWQ has already been validated with a real prompt of 367 tokens over `max_model_len=2048` and `safety_margin=256`.
- Retrieval / rerank 4B models exist, but they are not safe drop-in defaults; they require benchmarking and VRAM / context tuning before default adoption.
- In `sql_solver_generator`, `solver_semantic_context` must not include `candidate_plan_set` when the plan already has high confidence and there is no semantic mismatch. That extra payload should be included only for uncertain or misaligned plans; including it unconditionally can break the solver's 2048-token budget even for valid queries.

### Test Noise

- `pytest` warnings come from `sentencepiece==0.2.1`, pulled in by `vllm`.
- On Python 3.13 it emits SWIG `DeprecationWarning` entries (`SwigPyPacked`, `SwigPyObject`, `swigvarlink`).
- The correct mitigation is focused `filterwarnings` configuration in `pyproject.toml`, not application code changes.

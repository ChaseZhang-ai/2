# Contributing

Thanks for looking. This is an archived startup engine released as-is, so the
bar for changes is: **does it make the code more correct or more honest?**

## Running things

```bash
python -m unittest discover -s tests -v   # 42 tests, no dependencies, Python 3.9+
PYTHONPATH=. python examples/bench_numbers.py
make cli                                   # exercise every CLI subcommand
```

## Ground rules

- **No new runtime dependencies in `zhulu/`.** The core ships as zero-dependency
  pure Python on purpose (on-prem delivery). Optional heavy deps belong behind a
  caller-injected function (see how embeddings are passed to `HybridRetriever`).
- **Anything measured has a script.** If you change a benchmark number, change
  `examples/bench_numbers.py` and the docs together. Do not hand-edit results.
- **Keep the ComfyUI round-trip lossless.** `WorkflowGraph.to_comfy()` /
  `from_comfy()` must stay exact; there is a unit test.
- **Refusing to answer is correct.** Do not weaken the RAG abstention gate to make
  the demo look better.

## Adding a workflow

1. Build the graph in `zhulu/generation/<name>.py` using `WorkflowGraph`.
2. Make sure `signature()` collapses requests that differ only in prompt/seed
   (add volatile keys to `_VOLATILE_KEYS` in `generation/graph.py` if needed).
3. Declare a realistic `vram_mb` per node in the node `meta`.
4. Add a test asserting it is a DAG and that the signature behaves.


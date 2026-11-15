# Contributing

Thanks for looking. This is an archived startup engine released as-is, so the
bar for changes is: **does it make the code more correct or more honest?**

## Running things

```bash
python -m unittest discover -s tests -v   # 42 tests, no dependencies, Python 3.9+
PYTHONPATH=. python examples/bench_numbers.py
make cli                                   # exercise every CLI subcommand
```


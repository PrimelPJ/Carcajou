# Contributing

Contributions are welcome. Please open an issue first to discuss what you want to change.

## Setup

```bash
git clone https://github.com/PrimelPJ/Carcajou && cd Carcajou
pip install -e ".[dev,plots]"
```

## Before submitting a PR

- Run `ruff check src tests scripts` — zero warnings required
- Run `pytest -q` — all 32 tests must pass
- If you change benchmark methodology, regenerate `results/` with the appropriate script and include the diff

## Commit style

Short imperative subject line (≤ 72 chars), blank line, then body if needed. No trailing periods.

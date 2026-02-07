# Tests

This repository contains two kinds of tests:

1) **Core sanity tests** (should run without local artifacts)
2) **Artifact-dependent tests** (skipped unless you have local model/data files)

## Why tests are skipped

Some tests require local artifacts that are not committed to the repo (models, datasets).
If these artifacts are missing, tests will **skip** (not fail), so CI/clone users can still run the suite deterministically.

## Required artifacts (local)

Expected layout (example):

- `models/` (checkpoints / exported models)
- `data/` (dataset folders used by evaluation scripts)

Exact paths are referenced inside the tests. If you want these tests to run,
place the expected folders locally or update the paths in the tests to match your environment.

## Run

From repo root:

```bash
python -m pytest -q
```

To see skip reasons:

```bash
python -m pytest -rs -q
```


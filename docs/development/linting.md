# Linting

Linting is done via `ruff`.
It can be run manually via:

```bash
ruff check
ruff format
```

If you have other `ruff` installations in your `$PATH`, you can also run `ruff` via:

```bash
python -m ruff
```

Linter settings are specified in the `pyproject.toml` file.

## Type Checking

We currently use `mypy` for type checking throughout Conch.
`mypy` can be run automatically via `pre-commit` or manually via either:

```bash
mypy .
```

or

```bash
./tools/mypy.sh
```

Currently, third-party code (e.g. `conch/third_party/`) is excluded from type checking.

## Pre-commit

Linting can also be run automatically via `pre-commit`.
After installing the repo as an editable, run:

```bash
pre-commit install
```

`pre-commit` will then automatically execute when you commit to the repo.
You can also run `pre-commit` manually via:

```bash
pre-commit run --all-files
```

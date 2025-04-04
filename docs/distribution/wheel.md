# Building a wheel

1. Install extra dependencies

```bash
pip install build wheel twine
```

1. Build the wheel file

```bash
./scripts/wheel/build.sh
```

By default, this builds a wheel for your current platform (detected via `nvidia-smi`, `rocm-smi`, or similar).
Optionally, you can override the platform to build the wheel for by specifying it as an argument to the script.
Acceptable platforms are: `cuda` or `rocm`.

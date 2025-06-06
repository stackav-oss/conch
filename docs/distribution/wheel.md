# Building a wheel

1. Install extra dependencies

```bash
pip install build wheel twine
```

1. (Optional) Remove old files in `dist/`

```bash
rm -rf dist/*
```

1. Build the wheel file

```bash
./tools/wheel/build.sh
```

1. (Optional) Upload to PyPi

```bash
python -m twine upload dist/*
```

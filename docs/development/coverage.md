# Code Coverage

A coverage report can be generated via:

```bash
coverage run -m pytest
```

To view the report, run:

```bash
coverage report -m --ignore-errors
```

It appears that coverage is not calculated for the kernel source itself.

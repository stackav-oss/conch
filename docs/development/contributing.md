# Contributing to Conch

If you are interested in contributing to Conch, you are absolutely welcome to!
Please follow the [getting started instructions](/docs/getting_started/developer_environment.md) to configure your development environment.

## General guidelines

Please familiarize yourself with the documentation about the [operation structure](/docs/conch/structure.md) and [Triton best practices](/docs/conch/triton.md).
Please also review the [linting](/docs/development/linting.md) and [code coverage](/docs/development/coverage.md) documentation.

## PR Etiquette

### Linting and testing

Before submitting your PR, please ensure all linters are passing and tests are passing

```bash
pre-commit run --all-files
pytest
```

### Performance

If your PR is affecting the performance of an operation, please run the corresponding microbenchmark and share the results.

### Platform support

If you don't have access to hardware for all supported platforms, that's okay!
The maintainers of Conch can ensure your tests are passing on all supported platforms until we have a proper CI system.

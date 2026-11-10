# Releasing

Releases are GitHub releases only. There is no PyPI account, token or hosting involved.

1. Set `version` in `pyproject.toml`.
2. Move the `unreleased` heading in `CHANGELOG.md` to the version and date.
3. Merge to `main`, then create a GitHub release with tag `v<version>` (e.g. `v0.1.0`).
   `.github/workflows/release.yml` builds the wheel and sdist and attaches them to the release.

Users install with:

```bash
pip install git+https://github.com/12burb/pdf2openapi@v0.1.0
```

or download the wheel from the release page.

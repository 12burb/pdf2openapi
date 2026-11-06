# Releasing

## One-time PyPI setup

1. Create an account at https://pypi.org and enable 2FA.
2. Go to https://pypi.org/manage/account/publishing/ and add a pending trusted publisher:
   - PyPI project name: `pdf2openapi`
   - Owner: `12burb`, repository: `pdf2openapi`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. In GitHub, create an environment named `pypi` (Settings > Environments). Optionally require
   your approval before it runs.

No API token is needed; PyPI trusts the GitHub workflow directly.

## Each release

1. Set `version` in `pyproject.toml` and `__version__` in `src/pdf2openapi/__init__.py`.
2. Move the `unreleased` heading in `CHANGELOG.md` to the version and date.
3. Merge to `main`, then create a GitHub release with tag `v<version>` (e.g. `v0.1.0`).
   Publishing the release runs `.github/workflows/release.yml`, which builds and uploads to PyPI.

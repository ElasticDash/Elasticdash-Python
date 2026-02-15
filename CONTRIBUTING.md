# Contributing

Thank you for your interest in contributing to ElasticDash Python SDK!

## Attribution & Licensing

ElasticDash Python SDK is a fork of [Langfuse Python SDK](https://github.com/langfuse/langfuse-python). By contributing to this project, you agree that:

1. Your contributions will be licensed under the MIT License (same as the original project)
2. Your contributions become part of the ElasticDash project
3. Original Langfuse copyright and attribution are preserved in accordance with the MIT License
4. You have the right to contribute the code you're submitting

All new contributions are automatically copyrighted by ElasticDash while respecting the original Langfuse copyright. See the [LICENSE](LICENSE) and [NOTICE](NOTICE) files for complete attribution details.

## Upstream Contributions

If your contribution would benefit the original Langfuse project and doesn't depend on ElasticDash-specific features:

- Consider contributing to [Langfuse](https://github.com/langfuse/langfuse-python) first
- Then port the change to ElasticDash
- This benefits the entire LLM observability ecosystem

We encourage maintaining a positive relationship with the upstream Langfuse project.

## Development

### Add Poetry plugins

```bash
poetry self add poetry-dotenv-plugin
```

### Install dependencies

```bash
poetry install --all-extras
```

### Add Pre-commit

```bash
poetry run pre-commit install
```

### Type Checking

To run type checking on the elasticdash package, run:
```sh
poetry run mypy elasticdash --no-error-summary
```

### Tests

#### Setup

- Add .env based on .env.template

#### Run

- Run all

  ```bash
  poetry run pytest -s -v --log-cli-level=INFO
  ```

- Run a specific test

  ```bash
  poetry run pytest -s -v --log-cli-level=INFO tests/test_core_sdk.py::test_flush
  ```

- E2E tests involving OpenAI and Serp API are usually skipped, remove skip decorators in [tests/test_langchain.py](tests/test_langchain.py) to run them.

### Update openapi spec

1. Generate Fern Python SDK in [elasticdash](https://github.com/elasticdash/elasticdash) and copy the files generated in `generated/python` into the `elasticdash/api` folder in this repo.
2. Execute the linter by running `poetry run ruff format .`
3. Rebuild and deploy the package to PyPi.

### Publish release

Releases are automated via GitHub Actions using PyPI Trusted Publishing (OIDC).

To create a release:

1. Go to [Actions > Release Python SDK](https://github.com/elasticdash/elasticdash-python/actions/workflows/release.yml)
2. Click "Run workflow"
3. Select the version bump type:
   - `patch` - Bug fixes (1.0.0 → 1.0.1)
   - `minor` - New features (1.0.0 → 1.1.0)
   - `major` - Breaking changes (1.0.0 → 2.0.0)
   - `prerelease` - Pre-release versions (1.0.0 → 1.0.0a1)
4. For pre-releases, select the type: `alpha`, `beta`, or `rc`
5. Click "Run workflow"

The workflow will automatically:
- Bump the version in `pyproject.toml` and `elasticdash/version.py`
- Build the package
- Publish to PyPI
- Create a git tag and GitHub release with auto-generated release notes

### SDK Reference

Note: The generated SDK reference is currently work in progress.

The SDK reference is generated via pdoc. You need to have all extra dependencies installed to generate the reference.

```sh
poetry install --all-extras
```

To update the reference, run the following command:

```sh
poetry run pdoc -o docs/ --docformat google --logo "https://logger.elasticdash.com/icon.svg" elasticdash
```

To run the reference locally, you can use the following command:

```sh
poetry run pdoc --docformat google --logo "https://logger.elasticdash.com/icon.svg" elasticdash
```

## Credits

Thanks to the PostHog team for the awesome work on [posthog-python](https://github.com/PostHog/posthog-python). This project is based on it as it was the best starting point to build an async Python SDK.

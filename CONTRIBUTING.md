# Contributing

Use Python 3.11 or newer and follow the [README](README.md) setup instructions.
Please follow the [code of conduct](CODE_OF_CONDUCT.md).

## Local hooks

```sh
.venv/bin/python -m pip install -c .github/ci-constraints.txt ruff
git config --local core.hooksPath .githooks
```

Pre-commit checks staged content for safety and whitespace, and runs Ruff on the
working tree without editing files. It rejects private documents, Git-ignored
paths, credential markers, non-regular files, and files over 5 MiB. Staged and
working-tree `.gitignore` policies must agree. Ruff must match the CI pin.
Required lint failures remain blocking.

Pre-push runs `pytest -m 'not local_data'`, excluding optional local datasets and
external snapshots. Neither hook downloads dependencies or data.

## Changes and review

- Open an issue for bugs or substantial changes. Include a minimal synthetic
  reproduction, expected behavior, and relevant versions.
- Keep pull requests focused. Explain the change and include relevant tests.
- Preserve public APIs, rollout identities, sampling-unit boundaries, and explicit
  missing-evidence semantics. Explain estimator, threshold, or data-inclusion
  changes and include before/after benchmark results when applicable.
- Keep datasets, external source snapshots, generated reports, private documents,
  and credentials out of patches. Use synthetic fixtures and preserve source
  attribution and licensing restrictions.

## Checks

```sh
ruff format --check src bench tests
ruff check src bench tests
ruff check --select TRY004 --exit-zero src bench tests
.venv/bin/python -m pytest -q
```

Required lint follows `pyproject.toml`. `TRY004` is advisory because validation
APIs preserve `ValueError` contracts. Pre-commit and CI display these findings
separately; every other enabled rule remains required.

The README describes optional local-data and pinned-source checks. Report the
commands run, skips, and unresolved diagnostics in the pull request. Do not
suppress diagnostics or change public exception behavior solely to satisfy lint.

Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
Contributions are submitted under the [Apache License 2.0](LICENSE).

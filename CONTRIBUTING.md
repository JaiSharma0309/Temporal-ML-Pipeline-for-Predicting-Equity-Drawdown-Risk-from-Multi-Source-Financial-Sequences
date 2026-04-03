# Contributing

Thank you for your interest in improving this project.

## Workflow

- Fork the repository or create a feature branch from `main`.
- Keep changes focused and well-scoped.
- Use clear commit messages.
- Update documentation when behavior, commands, or outputs change.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Before Opening a PR

- Run the relevant pipeline steps locally.
- Confirm the README instructions still match the repository state.
- Avoid committing large generated datasets unless they are intentionally part of the repo.
- Keep generated prediction dumps and reproducible raw data out of version control unless needed.

## Style

- Prefer readable, well-named functions and variables.
- Keep docstrings up to date.
- Preserve time-aware evaluation assumptions and leakage safeguards.

## Issues and Suggestions

- Bug reports should include the script name, command run, and a short description of the failure.
- Improvement ideas are welcome, especially around data quality, temporal validation, and model interpretability.

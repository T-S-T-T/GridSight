# Tooling files to copy into place

Remote tools cannot write these files, so they are kept here with a `.txt` suffix.
Copy them once (PowerShell, from the project root):

```powershell
New-Item -ItemType Directory -Force .github\workflows | Out-Null
Copy-Item tooling\ci.yml.txt .github\workflows\ci.yml
Copy-Item tooling\pre-commit-config.yaml.txt .pre-commit-config.yaml
Copy-Item tooling\Makefile.txt Makefile
```

| File | Goes to | Purpose |
| --- | --- | --- |
| `ci.yml.txt` | `.github/workflows/ci.yml` | GitHub Actions: lint, tests with coverage, offline end-to-end demo on Ubuntu and Windows |
| `pre-commit-config.yaml.txt` | `.pre-commit-config.yaml` | ruff on commit, fast tests on push (`pre-commit install`) |
| `Makefile.txt` | `Makefile` | Shortcuts for macOS/Linux; on Windows use the `gridsight` commands directly |

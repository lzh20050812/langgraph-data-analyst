# Dependency reproducibility policy

The runtime and development requirement files pin every direct Python dependency to an exact version. These pins match the environment used by the 2026-08-18 regression suite.

## Supported environments

- Docker and CI: Python 3.11 on Linux amd64.
- Verified local development: Python 3.12 on Windows amd64.
- Docker installs `torch==2.13.0` from the official CPU wheel index before installing the project requirements.
- Linux amd64 uses `xgboost-cpu==3.2.0`, the newest CPU wheel line verified here to support Python 3.11. Other platforms use `xgboost==3.4.0`.
- NumPy is pinned by interpreter: `2.4.6` on Python 3.11 and `2.5.2` on Python 3.12 or newer.

## Update procedure

1. Change dependency versions intentionally in `requirements.txt` or `requirements-dev.txt`.
2. Build the Docker image from a clean cache.
3. Run `python -m pip check` and `python -m pytest -q` locally or in Docker.
4. Record any metric-affecting change as a new experiment version; do not silently replace `evaluation/final_results/final_metrics.json`.
5. Commit the requirement change together with its test evidence.

Exact direct pins prevent silent upgrades of the main libraries. Transitive packages are resolved by pip and cached in CI; a production release can add a platform-specific hash lock after building on the target platform.

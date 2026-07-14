# Project instructions

- Treat `configs/*.json` as the source of runtime paths and pipeline settings.
- Keep model architecture and forecasting methodology unchanged unless the user explicitly requests a methodological change.
- Keep notebooks thin: they may select configuration/stage and invoke scripts, but production logic belongs under `src/` or `scripts/`.
- Write runtime files only below the configured `outputs/` locations.
- Every change must run `python -m unittest discover -s tests -v` and `python scripts/smoke_test.py`.
- Do not commit checkpoints, submissions, logs, caches, or generated experiment directories.
- Preserve causal preprocessing and autoregressive inference behavior.

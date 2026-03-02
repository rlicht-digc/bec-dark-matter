# Unified Pipeline Regression Tests

Run:

```bash
python -m pytest analysis/tests/test_unified_pipeline_regression.py
```

What this protects:

- SPARC loader regression: fails if SPARC inputs cannot be resolved from expected data directories, or if loaded SPARC coverage drops below safety floors.
- Unified CSV drift regression: fails if `analysis/results/rar_points_unified.csv` loses `galaxy_key`, or if SPARC point/galaxy counts drop below expected scale.


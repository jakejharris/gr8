# Steering release and qualification tools

GR8 serves prebuilt steering profiles. Profile extraction belongs to the project that publishes the profile; these tools
package a finished profile and qualify its behavior and runtime path.

`release.py` packs `manifest.json` and `directions.f32` into the deterministic safetensors artifact consumed by Hugging
Face discovery. It also validates and unpacks that artifact without requiring NumPy or the safetensors package.

`evaluate.py` produces paired bootstrap confidence intervals, non-inferiority results, behavior rates, McNemar's
secondary statistic, and NLL/top-20 diagnostics from a frozen paired JSONL file.

`trace_audit.py` compares normalized CUDA traces for steered requests against shape-matched stock work and fails on any
kernel, copy, allocation, synchronization, or graph-node count difference.

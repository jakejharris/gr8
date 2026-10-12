# Steering profile schema v1

A profile is a non-executable directory with exactly the two runtime inputs below. Extra files are ignored.

```text
PROFILE/
  manifest.json
  directions.f32
```

`manifest.json` has this shape:

```json
{
  "schema_version": 1,
  "profile_name": "example",
  "model": {
    "name": "glm-5.3",
    "revision": "exact-model-revision",
    "weight_set": "exact-launch-model-set-id",
    "tokenizer_revision": "exact-tokenizer-revision",
    "chat_template_sha256": "64 lowercase hex digits",
    "hidden_size": 6144,
    "layer_count": 78
  },
  "scope": "full",
  "layers": [2, 3, 4],
  "alphas": [1.0, 1.0, 1.0],
  "directions": {
    "file": "directions.f32",
    "dtype": "float32-le",
    "byte_count": 73728,
    "sha256": "64 lowercase hex digits"
  },
  "provenance": {
    "extraction_tool": "publisher-defined extraction tool",
    "extraction_commit": "git commit",
    "dataset_manifests": {
      "train": "64 lowercase hex digits",
      "validation": "64 lowercase hex digits",
      "final": "64 lowercase hex digits"
    }
  }
}
```

The profile name matches `[A-Za-z0-9._-]{1,128}`. `scope` is `shared` or `full`. Layers are unique, strictly
increasing model-layer indices. `alphas` has the same length and every value is finite and in `[0, 4]`.

`directions.f32` stores one little-endian IEEE-754 FP32 vector of width 6144 for each layer, in manifest order. Every
component is finite. Each vector's FP64-computed norm must be in `[0.999, 1.001]`. `byte_count` is exactly
`4 * hidden_size * len(layers)` and `sha256` covers the complete file.

The runtime hashes the exact manifest bytes for the profile identity. This commits to the tensor digest, model,
scope, layer schedule, alphas and provenance. It rejects duplicate JSON keys, unknown schema/scope, model-identity
mismatch, invalid layer order, inconsistent counts, short or trailing tensor bytes, digest mismatch, non-finite data,
zero or off-tolerance norms, and missing provenance before readiness. The S2 boot receipt and replicated-state hash
make rank disagreement fail closed.

Schema v1 loads one profile per process. Requests carry only its name and a scale; they cannot supply paths, tensors,
scopes, layers or alphas.

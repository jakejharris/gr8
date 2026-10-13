# Runtime steering

g6 can load one immutable rank-one steering profile at boot and select it independently for each request. The feature
is opt in. A boot without `STEERING_PROFILE` keeps the original public request fields, graphs and outputs.

For each active layer, the target model applies this operator to the attention writer output and to the configured MLP
writer output:

```text
y' = y - request_scale * profile_alpha * r * (r^T y)
```

`r` is a validated unit FP32 direction. This projects each writer's output, not the residual stream: a direction
published for the post-layer residual (`h <- h - alpha * (h . d) d`) is a different operator, and needs converting
before it is packed as a profile. `full` scope projects the combined shared and routed MLP output. `shared`
scope projects the dense/shared MLP contribution; the routed experts do not enter its dot product. The projection is
fused into launches that already consume or finalize those writer outputs. Row scales use the existing step-control
transfer and profile vectors stay on the device after boot. Steering therefore adds no CUDA launch, copy, allocation,
synchronization or graph recapture to a production step. An exact zero scale returns before direction loads,
projection arithmetic or projection stores for that row.

The target model is steered during ordinary decode, S2 mixed-session decode, prompt tails, prompt-logprob scoring,
native MTP and DFlash verification, and every prefill collective/ownership form. The draft model is unchanged. Prefix
reuse keys include the profile, scope, layer schedule and exact canonical FP32 scale, so state produced under one
configuration cannot serve another.

## Boot

Every rank needs the same profile directory and the exact identity values recorded in its manifest:

```bash
./gr8 glm53 up @cluster.env \
  --steering-profile /opt/g6/profiles/example \
  --steering-model-revision MODEL_REVISION \
  --steering-tokenizer-revision TOKENIZER_REVISION \
  --steering-chat-template-sha256 64_LOWERCASE_HEX_DIGITS
```

For a local profile, the launcher's model-set id supplies `model.weight_set` unless
`--steering-weight-set SHA256` names the stable underlying exlr8 weight set. The engine validates the full schema and
tensor before readiness, copies the directions once, and includes the profile receipt in all-rank boot agreement. A
mismatch in any rank stops the boot. [STEERING-PROFILE-SCHEMA.md](STEERING-PROFILE-SCHEMA.md) defines the files.

An HF release may bundle profiles as deterministic safetensors artifacts under `steering/`, listed in
`steering/index.json`. GR8 loads one only when the operator asks for it: `--steering-profile hf:default` takes the
release's default profile, and `--steering-profile hf:NAME` a named one. GR8 then downloads and verifies the artifact
with the rest of the release, materializes the runtime profile in the model cache, and fills the identity fields from
its embedded manifest. Without `--steering-profile`, a release's `steering/` directory is ignored and nothing from it
is downloaded. A loaded profile stays off for every request that does not select it, unless the operator sets
`--steering-force`. Gated repositories use the existing `HF_TOKEN` authentication.

A profile that cannot be loaded stops the boot before any engine starts, with the reason: a directory or file that
does not exist, an `hf:` name the release does not list, a schema or tensor that fails validation, or identity fields
that do not match the model.

Create or inspect the single-file artifact without NumPy or PyTorch:

```bash
python3 tools/steering/release.py pack PROFILE_DIR steering/PROFILE.safetensors
python3 tools/steering/release.py unpack steering/PROFILE.safetensors OUTPUT_DIR
```

For process-wide qualification, add `--steering-force SCALE`, where `0 < SCALE <= 4`. It selects the same profile and
operator for every request. Request fields that conflict with forced mode are rejected. `GET /health` reports whether
a profile is loaded, whether forced mode is active, its name, digest prefix, scope and active layer count.

## Requests

Both completion endpoints accept two optional vendor fields:

```json
{"steering_profile":"example","steering_scale":1.0}
```

Omitting both turns steering off. A request that names a profile on a server started without one is refused with
400, `steering is not enabled on this server (start it with --steering-profile)`. A profile without a scale uses
`1.0`. A scale without a profile is accepted only
when it is zero, which canonicalizes to off. Values must be finite and in `[0, 4]`; the profile name must equal the
one loaded at boot. Scale is converted to FP32 once at the API boundary, and its exact canonical bits become part of
the request and prefix identity.

## Release and qualification

`tools/steering/` packages a publisher-provided profile, evaluates paired held-out results, and audits normalized CUDA
traces. Profile extraction stays with the project that publishes the profile. See
[tools/steering/README.md](../tools/steering/README.md).

## Profiles and licensing

GR8 ships no steering profile and loads none unless the operator names one. Operators may load profiles of their own
choice. A profile is a derivative of the model it was extracted from: its manifest should record its source artifacts,
and its distribution should carry the license and notices those sources require. The engine and the tools in
`tools/steering/` remain under this repository's MIT license, which a profile does not inherit.

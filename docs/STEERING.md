# Runtime steering

g6 can load one immutable rank-one steering profile at boot and select it independently for each request. The feature
is opt in. A boot without `STEERING_PROFILE` keeps the original public request fields, graphs and outputs.

For each active layer, the target model applies this operator to the attention writer output and to the configured MLP
writer output:

```text
y' = y - request_scale * profile_alpha * r * (r^T y)
```

`r` is a validated unit FP32 direction. `full` scope projects the combined shared and routed MLP output. `shared`
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

An HF release may bundle profiles as deterministic safetensors artifacts under `steering/` and declare one default in
`steering/index.json`. GR8 downloads and verifies that artifact with the rest of the release, materializes the runtime
profile in the model cache, and fills all identity fields from its embedded manifest. The profile is then available to
requests but remains off unless the request selects it (or the operator sets `--steering-force`). An explicit
`--steering-profile DIR` always takes precedence. Gated repositories use the existing `HF_TOKEN` authentication.

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

Omitting both turns steering off. A profile without a scale uses `1.0`. A scale without a profile is accepted only
when it is zero, which canonicalizes to off. Values must be finite and in `[0, 4]`; the profile name must equal the
one loaded at boot. Scale is converted to FP32 once at the API boundary, and its exact canonical bits become part of
the request and prefix identity.

## Release and qualification

`tools/steering/` packages a publisher-provided profile, evaluates paired held-out results, and audits normalized CUDA
traces. Profile extraction stays with the project that publishes the profile. See
[tools/steering/README.md](../tools/steering/README.md).

## Profile provenance and licensing

The `ablit-behavior-v1` profile used to qualify this implementation was independently extracted from weight
differences between `zai-org/GLM-5.3-BF16` and
`drowzeys/keys-GLM-5.3-EXL3-Abliterated`. The latter credits its de-risk projections to Blackfrost Research's
`GLM-5.3-DERISKED-BF16`. The profile is a separately licensed model derivative and is not part of this repository.

A distribution that places this profile in a model repository under `steering/` should put the following text in an
adjacent `NOTICE` file and include the complete GLM-5.3 license:

> This steering profile was independently extracted from weight differences between `zai-org/GLM-5.3-BF16` and
> `drowzeys/keys-GLM-5.3-EXL3-Abliterated`, whose de-risk projections are credited to Blackfrost Research. Distributed
> subject to the GLM-5.3 License.

The g6 engine and extraction tools remain under this repository's MIT license. Other steering profiles must record
their own source artifacts and applicable license rather than inheriting the engine license.

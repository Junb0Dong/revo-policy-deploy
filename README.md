# Revo Policy Deploy

Independent deployment runtime for Revo ACT checkpoints. This repository owns the
policy loader, named observation adapter, checkpoint contract, framed-stdio protocol,
and inferencer process. It does not contain training data, ROS integration, or robot
control code.

## Install

```bash
uv sync --locked

cd /tmp
/mnt/data/junbo/revo-policy-deploy/.venv/bin/python -c \
  "import revo_policy_deploy"
```

The lockfile pins the published LeRobot runtime wheel and the complete PyTorch stack.
The runtime never imports the LeRobot training checkout through `PYTHONPATH` or an
editable path dependency.

## Checkpoint artifact

The training repository must export a portable directory:

```text
checkpoint/
├── params/
│   └── pretrained_model/
├── train_config.yaml
├── policy_contract.json
└── assets/
    ├── resolved_interface.json
    ├── dataset_info.json
    └── tasks.json                 # only for multi-task policies
```

Store large artifacts outside this repository, for example:

```text
/data/policy_artifacts/revo_act/
├── checkpoint_040000/
├── checkpoint_050000/
└── current -> checkpoint_050000
```

The package must contain only relative runtime paths. `LOAD` rejects a modified
contract, mismatched checkpoint ID, path escape, missing runtime file, content hash
mismatch, unsupported loader version, or incompatible LeRobot version.

## Run

```bash
uv run python -m revo_policy_deploy.deploy.policy_inferencer \
  --checkpoint /data/policy_artifacts/revo_act/current \
  --device auto
```

The lifecycle is `DESCRIBE → LOAD → INFER → RESET/CLOSE`. Standard output is reserved
for framed protocol messages; logs and third-party output are redirected to standard
error before model loading.

`INFER` accepts named state groups and RGB `uint8` HWC images. It returns finite
`float32[T,D]` absolute joint targets in this fixed order:

```text
left_arm -> right_arm -> left_hand -> right_hand
```

ACT is the only supported policy family in this initial repository. ActionCodec must
be added as a separate policy adapter without changing the deployment protocol.

## Repository boundaries

- LeRobot training repository: datasets, training, native checkpoints, exporter.
- This repository: artifact validation, ACT loading, preprocessing, inference, protocol.
- `revo_deploy`: ROS, Robot Profile, limits, emergency stop, action playback, monitoring.

Changing to another compatible checkpoint only changes `--checkpoint`; it does not
require editing this repository or reinstalling its environment.

## Verify

```bash
uv run pytest
uv run ruff check .

cd /tmp
/mnt/data/junbo/revo-policy-deploy/.venv/bin/python -c \
  "import revo_policy_deploy"
```

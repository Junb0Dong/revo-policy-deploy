# Vendored LeRobot runtime

`lerobot-0.6.2-py3-none-any.whl` is the runtime wheel used by this deployment
repository. It is intentionally vendored so deployment installation does not
depend on a LeRobot training checkout or an editable source path.

The wheel is only used for model/configuration/processor loading. Training data,
training scripts, exporters, and robot hardware integrations remain outside this
repository.

To replace it, build and validate a new wheel in the training repository, copy it
here under the same pinned version (or update the dependency and lockfile), then
run the full deployment test suite and direct-vs-stdio inference comparison.

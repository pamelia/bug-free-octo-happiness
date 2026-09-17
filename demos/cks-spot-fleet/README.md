# CKS Spot Fleet DDP trainer

This synthetic PyTorch workload demonstrates one regional, two-node training
run selected by the CKS Spot Fleet controller. `torchrun` launches eight ranks
per node; the script initializes a 16-rank NCCL process group and performs real
distributed gradient synchronization.

The Kubernetes demo accepts a public repository URL, commit, and
repository-relative entrypoint. When intentionally testing the latest commit
from a branch such as `main`, resolve that branch once before creating the
TrainJob. Every distributed rank then fetches the same commit, detaches at
`FETCH_HEAD`, and logs the SHA before launching `training.py`.

Tags matching `cks-spot-fleet-v*` also publish `training.py` as a versioned
source archive with a SHA-256 checksum. This gives consumers a
checksum-verifiable download without requiring this repository's GitHub owner
or URL to be embedded in the platform repository.

For branch-latest mode, resolve the branch once and pass the resulting SHA as
`TRAINING_SOURCE_COMMIT`:

```bash
TRAINING_SOURCE_BRANCH=main
TRAINING_SOURCE_COMMIT=$(git ls-remote "$TRAINING_SOURCE_REPOSITORY" \
  "refs/heads/$TRAINING_SOURCE_BRANCH" | awk 'NR == 1 {print $1}')
test -n "$TRAINING_SOURCE_COMMIT"

git init /tmp/training-source
git -C /tmp/training-source remote add origin "$TRAINING_SOURCE_REPOSITORY"
git -C /tmp/training-source fetch --depth=1 origin "$TRAINING_SOURCE_COMMIT"
git -C /tmp/training-source checkout --detach FETCH_HEAD
git -C /tmp/training-source rev-parse HEAD
```

`wait_for_dns.py` remains available as an optional diagnostic for environments
that launch `torchrun` without Trainer and JobSet readiness handling.

This is a scheduling and execution demo, not a performance benchmark. It uses
synthetic data and NCCL over the Pod network.

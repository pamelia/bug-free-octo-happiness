"""Small multi-node PyTorch DDP workload for the CKS Spot Fleet demo.

Each GPU runs a copy of the same model on random inputs and targets. This
exercises GPU compute and cross-worker gradient synchronization; it does not
train a useful model or save checkpoints for recovery after Spot interruption.
"""

import os
import socket
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def main() -> None:
    # The distributed launcher supplies each process's identity: local rank is
    # its GPU index on this node, global rank is its ID across the whole job,
    # world size counts all processes, and node rank identifies this node.
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    node_rank = int(os.environ["NODE_RANK"])

    # Bind one process to one GPU, then join the distributed communication
    # group. NCCL carries GPU-to-GPU communication within and between nodes.
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    device = torch.device("cuda", local_rank)
    # These knobs control model width, samples per GPU per step, and runtime
    # in training steps. The default settings apply unless the job overrides them.
    hidden = int(os.environ.get("HIDDEN_SIZE", "8192"))
    batch = int(os.environ.get("BATCH_SIZE", "32"))
    steps = int(os.environ.get("TRAIN_STEPS", "120"))

    # A small network structure with wide layers keeps the GPUs busy. BF16
    # uses less memory than float32 for model weights and activations.
    model = nn.Sequential(
        nn.Linear(hidden, hidden, bias=False),
        nn.GELU(),
        nn.Linear(hidden, hidden, bias=False),
    ).to(device=device, dtype=torch.bfloat16)
    # DDP starts replicas with matching parameters and synchronizes gradients
    # during backward(), so every GPU applies the same SGD parameter update.
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    print(
        f"TRAINER_READY host={socket.gethostname()} node_rank={node_rank} "
        f"local_rank={local_rank} global_rank={global_rank} "
        f"world_size={world_size} gpu={torch.cuda.get_device_name(local_rank)}",
        flush=True,
    )
    # Wait until every process is ready before timing the training loop.
    dist.barrier()
    started = time.monotonic()

    for step in range(1, steps + 1):
        # Generate a fresh synthetic batch on each GPU, with no dataset or
        # storage dependency. Random targets mean loss need not steadily improve.
        x = torch.randn(batch, hidden, device=device, dtype=torch.bfloat16)
        target = torch.randn(batch, hidden, device=device, dtype=torch.bfloat16)
        # Clear old gradients, predict, measure error in float32, synchronize
        # new gradients via DDP, and update the model weights.
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
        loss.backward()
        optimizer.step()

        # Only global rank 0 prints progress to avoid duplicate job-level logs.
        # Throughput estimates total samples across all GPUs since the loop began.
        if global_rank == 0 and (step == 1 or step % 10 == 0):
            elapsed = time.monotonic() - started
            samples = step * batch * world_size
            print(
                f"TRAINING step={step}/{steps} loss={loss.item():.5f} "
                f"world_size={world_size} "
                f"samples_per_second={samples / elapsed:.1f}",
                flush=True,
            )

    # Finish queued GPU work and wait for every process before reporting
    # completion, then release the distributed communication resources.
    torch.cuda.synchronize()
    dist.barrier()
    if global_rank == 0:
        elapsed = time.monotonic() - started
        print(
            f"TRAINING_COMPLETE steps={steps} world_size={world_size} "
            f"elapsed_seconds={elapsed:.1f}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

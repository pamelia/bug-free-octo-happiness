"""Small multi-node PyTorch DDP workload for the CKS Spot Fleet demo."""

import os
import socket
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    node_rank = int(os.environ["NODE_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    device = torch.device("cuda", local_rank)
    hidden = int(os.environ.get("HIDDEN_SIZE", "8192"))
    batch = int(os.environ.get("BATCH_SIZE", "32"))
    steps = int(os.environ.get("TRAIN_STEPS", "120"))

    model = nn.Sequential(
        nn.Linear(hidden, hidden, bias=False),
        nn.GELU(),
        nn.Linear(hidden, hidden, bias=False),
    ).to(device=device, dtype=torch.bfloat16)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    print(
        f"TRAINER_READY host={socket.gethostname()} node_rank={node_rank} "
        f"local_rank={local_rank} global_rank={global_rank} "
        f"world_size={world_size} gpu={torch.cuda.get_device_name(local_rank)}",
        flush=True,
    )
    dist.barrier()
    started = time.monotonic()

    for step in range(1, steps + 1):
        x = torch.randn(batch, hidden, device=device, dtype=torch.bfloat16)
        target = torch.randn(batch, hidden, device=device, dtype=torch.bfloat16)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
        loss.backward()
        optimizer.step()

        if global_rank == 0 and (step == 1 or step % 10 == 0):
            elapsed = time.monotonic() - started
            samples = step * batch * world_size
            print(
                f"TRAINING step={step}/{steps} loss={loss.item():.5f} "
                f"world_size={world_size} "
                f"samples_per_second={samples / elapsed:.1f}",
                flush=True,
            )

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

# FAQ / environment tuning

glide deliberately sets almost no environment variables for you. The ones below
are common on GPU clusters, but every one of them is a workaround for a specific
machine, so the right value depends on your node — export them yourself (or from
your job script) rather than expecting glide to guess.

The single exception is `NCCL_DEBUG`, which glide defaults to `WARN` when it
self-launches under `torchrun`, so a hanging or failing collective prints
something instead of nothing. Override it like any other variable.

## NCCL transport

### `NCCL_IB_DISABLE=1`

Turns off InfiniBand and falls back to TCP over Ethernet.

Set this **only if your node's IB is broken or absent**. On a cluster with
working IB this silently costs you most of your interconnect bandwidth, so it is
not a safe default.

Symptoms that suggest you need it: `torchrun` hangs at the first collective
(often the first `all_reduce` of step 1) with no error, or NCCL logs repeated
`NET/IB : Got completion with error` lines.

```bash
NCCL_IB_DISABLE=1 glide sft configs/my_sft.yaml
```

### `NCCL_P2P_DISABLE=1`

Turns off direct GPU-to-GPU transfers over PCIe. Needed on hosts where P2P is
advertised but broken — the classic symptom is a hang or garbage gradients on
multi-GPU single-node runs that work fine with `nproc_per_node: 1`.

Check whether P2P actually works before reaching for this:

```bash
python -c "import torch; print(torch.cuda.can_device_access_peer(0, 1))"
```

### `NCCL_DEBUG=INFO`

Raise the log level from glide's `WARN` default when you are actually debugging
a collective. `INFO` prints the chosen transport per rank, which is the fastest
way to confirm whether IB or P2P is in play.

```bash
NCCL_DEBUG=INFO glide sft configs/my_sft.yaml 2>&1 | grep -E "NET/|via P2P"
```

## Allocator

### `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Lets the CUDA caching allocator grow segments instead of fragmenting them.
Often helps when a run OOMs partway through with a lot of *reserved but
unallocated* memory — the tell is a message like
`reserved in total by PyTorch` far exceeding `allocated`.

It is not free: on some driver/PyTorch combinations it is slower or itself
unstable, which is why glide does not turn it on for you.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True glide sft configs/my_sft.yaml
```

## Device selection

### `CUDA_VISIBLE_DEVICES`

glide's `distributed.nproc_per_node: null` resolves to
`torch.cuda.device_count()`, which respects this variable. Restricting visible
devices is therefore the normal way to pin a run to specific GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 glide sft configs/my_sft.yaml
```

An explicit `nproc_per_node` larger than the visible GPU count is clamped, with
a warning.

## Setting these permanently

Put them in the job script or shell profile that launches glide, not in the
library:

```bash
# ~/.bashrc, or your scheduler's job wrapper
export NCCL_IB_DISABLE=1        # this node's IB is broken
export NCCL_P2P_DISABLE=1       # ...and so is its PCIe P2P
```

Because glide uses `setdefault` for `NCCL_DEBUG`, anything you export always
wins.

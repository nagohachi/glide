"""Command-line entry point: ``glide <command> [config.yaml] [--dotted.key value ...]``.

Commands
--------
``sft``   Supervised fine-tuning.
``grpo``  Group Relative Policy Optimization.
``gspo``  Group Sequence Policy Optimization (GRPO with sequence-level IS).
``eval``  Validation-time autoregressive decoding + metrics on a checkpoint.
``test``  Held-out test evaluation on ``data.test`` at the end of training.
``docs``  Generate API documentation from source + docstrings (pdoc).

Any argument after the optional config path is treated as a dotted override, e.g.::

    glide sft configs/asr_sft.yaml --model.name Qwen/Qwen3-ASR-1.7B --training.learning_rate 1e-5
"""

import argparse
import os
import sys

from .. import __version__

_TRAIN_TASKS = {"sft", "grpo", "gspo"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glide", description="TRL-based post-training library.")
    parser.add_argument("--version", action="version", version=f"glide {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd in sorted(_TRAIN_TASKS):
        p = sub.add_parser(cmd, help=f"Run {cmd.upper()} training.")
        p.add_argument("config", nargs="?", default=None, help="Path to the run YAML.")

    pe = sub.add_parser("eval", help="Run AR-decoding evaluation + metrics.")
    pe.add_argument("config", nargs="?", default=None, help="Path to the run YAML.")
    pe.add_argument(
        "--checkpoint",
        "-c",
        default=None,
        help="Model checkpoint path or HF hub id (overrides model.name).",
    )

    pt = sub.add_parser("test", help="Run held-out test evaluation on data.test.")
    pt.add_argument("config", nargs="?", default=None, help="Path to the run YAML.")
    pt.add_argument(
        "--checkpoint",
        "-c",
        default=None,
        help="Model checkpoint path or HF hub id (overrides model.name).",
    )
    pt.add_argument(
        "--output", "-o", default=None, help="Save per-sample predictions to this JSONL file."
    )

    pd = sub.add_parser("docs", help="Generate API docs from docstrings (pdoc).")
    pd.add_argument("--output", "-o", default="docs/api", help="Output directory.")
    pd.add_argument("--serve", action="store_true", help="Serve docs instead of writing HTML.")
    return parser


def _run_training(config, task: str) -> int:
    from transformers import set_seed

    from ..logging_utils import setup_logging, snapshot_config
    from ..trainers import build_trainer

    set_seed(config.seed)
    setup_logging(config)

    trainer = build_trainer(config)
    output_dir = trainer.args.output_dir
    snapshot_config(config, output_dir)
    print(f"[glide] {task} -> output_dir={output_dir}")

    trainer.train()
    trainer.save_model(output_dir)
    if getattr(trainer, "processing_class", None) is not None:
        trainer.processing_class.save_pretrained(output_dir)
    print(f"[glide] done. Model saved to {output_dir}")
    return 0


def _to_eval_device(model):
    """Move a standalone eval/test model to CUDA when available (else leave on CPU)."""
    import torch

    if torch.cuda.is_available():
        model.to("cuda")
    return model


def _run_generation_eval(
    config_path,
    overrides: list[str],
    *,
    split: str,
    checkpoint: str | None = None,
    output: str | None = None,
) -> int:
    """Load a checkpoint and score one data split by autoregressive decoding.

    ``eval`` and ``test`` differ only in which split they read and how much of it
    they score; everything else (config load, plugin init, model load, decoding)
    is identical.

    Args:
        config_path: Path to the run YAML.
        overrides: Dotted ``--key value`` overrides left over from argparse.
        split: ``"eval"`` scores ``data.eval`` and honours
            ``eval.generate.max_eval_samples``; ``"test"`` scores all of
            ``data.test``, uncapped.
        checkpoint: Overrides ``model.name`` when given.
        output: Save per-sample predictions to this JSONL path.

    Returns:
        ``0`` on success, ``1`` if the split is not configured.
    """
    from ..config.loader import load_config
    from ..data.jsonl import read_jsonl
    from ..eval.generate import GenerationEvaluator
    from ..models.loader import load_model_and_processor
    from ..trainers.common import init_plugins

    if checkpoint:
        overrides = [*overrides, f"--model.name={checkpoint}"]
    config = load_config(config_path, overrides, task="sft")
    config.eval.generate.enabled = True
    init_plugins(config)
    loaded = load_model_and_processor(config)
    _to_eval_device(loaded.model)

    paths = getattr(config.data, split)
    if paths is None:
        print(f"[glide] no data.{split} configured; nothing to evaluate.", file=sys.stderr)
        return 1
    if not isinstance(paths, list):
        paths = [paths]
    records = []
    for path in paths:
        records.extend(read_jsonl(path))

    # Test scores the full set; max_eval_samples only caps validation.
    evaluator = GenerationEvaluator(
        config, loaded.processor, records, cap_samples=(split == "eval")
    )
    metrics = evaluator.evaluate(loaded.model, save_path=output, prefix=split)
    print(f"[glide][{split}]", metrics)
    return 0


def _discover_modules() -> list[str]:
    """Return ``glide`` plus every importable submodule (for full pdoc coverage)."""
    import pkgutil

    import glide

    mods = ["glide"]
    for info in pkgutil.walk_packages(glide.__path__, prefix="glide."):
        mods.append(info.name)
    return mods


_GITHUB_REPO = "https://github.com/nagohachi/glide"
_GITHUB_SRC = f"{_GITHUB_REPO}/blob/main/src/glide/"
# docs/templates is four levels above this file: src/glide/cli/main.py -> repo root
_TEMPLATE_DIR = str(__import__("pathlib").Path(__file__).parents[3] / "docs" / "templates")


def _run_docs(output: str, serve: bool) -> int:
    import subprocess

    modules = _discover_modules()
    cmd = [
        "pdoc",
        *modules,
        "--edit-url",
        f"glide={_GITHUB_SRC}",
        "--template-directory",
        _TEMPLATE_DIR,
    ]
    # Modern pdoc (>=14) serves at http://localhost:8080 by default when no output dir
    # is given; the old pdoc3 `--http :8080` flag no longer exists (argparse error).
    if not serve:
        cmd += ["-o", output]
    print(
        f"[glide] generating docs for {len(modules)} modules -> "
        f"{'http://localhost:8080' if serve else output}"
    )
    return subprocess.call(cmd)


def _is_distributed_worker() -> bool:
    """True if we're already running inside a torchrun/torch.distributed worker.

    ``RANK``/``LOCAL_RANK`` are set by ``torch.distributed.run`` itself (not by the
    user), so we use them only to detect "am I a spawned worker?".
    """
    import os

    return any(k in os.environ for k in ("RANK", "LOCAL_RANK", "TORCHELASTIC_RUN_ID"))


def _resolve_nproc(dist_cfg) -> int:
    """Resolve processes-per-node from the YAML ``distributed`` config.

    ``nproc_per_node: null`` -> auto = number of visible GPUs. An explicit value
    larger than the visible GPU count is clamped with a warning.
    """
    try:
        import torch

        ngpu = torch.cuda.device_count()
    except Exception:
        ngpu = 0
    if dist_cfg.nproc_per_node is None:
        return max(1, ngpu)
    n = int(dist_cfg.nproc_per_node)
    if ngpu and n > ngpu:
        print(
            f"[glide] distributed.nproc_per_node={n} > visible GPUs ({ngpu}); clamping to {ngpu}."
        )
        n = ngpu
    return n


def _relaunch_distributed(argv: list[str], dist_cfg, nproc: int) -> int:
    """Re-exec the current command under ``torch.distributed.run`` (DDP).

    Driven entirely by the YAML ``distributed`` config (no env vars required).
    Single-node uses ``--standalone``; multi-node uses the node/master settings.
    """
    import datetime as _dt

    from torch.distributed.run import main as torchrun_main

    # Stamp the versioned output dir ONCE here (parent) so every torchrun worker
    # inherits the same value -> all ranks share one output dir (no per-rank timestamp).
    os.environ.setdefault("GLIDE_OUTPUT_STAMP", _dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # NCCL logging default. Node-specific NCCL/allocator tuning is deliberately NOT
    # set here -- see docs/tutorials/faq.md for the vars you may want to export.
    os.environ.setdefault("NCCL_DEBUG", "WARN")

    launch = [f"--nproc_per_node={nproc}"]
    if dist_cfg.nnodes > 1 or dist_cfg.master_addr:
        launch += [f"--nnodes={dist_cfg.nnodes}", f"--node_rank={dist_cfg.node_rank}"]
        if dist_cfg.master_addr:
            launch.append(f"--master_addr={dist_cfg.master_addr}")
        if dist_cfg.master_port:
            launch.append(f"--master_port={dist_cfg.master_port}")
    else:
        launch.append("--standalone")
    launch += ["--module", "glide.cli.main", *argv]
    print(f"[glide] launching distributed ({nproc} proc/node): torchrun {' '.join(launch)}")
    torchrun_main(launch)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``glide`` console script."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args, overrides = parser.parse_known_args(argv)

    if args.command in _TRAIN_TASKS:
        from ..config.loader import load_config

        config = load_config(args.config, overrides, task=args.command)
        # Self-launch under torchrun when distributed.nproc_per_node resolves to >1
        # (configured in YAML), unless we're already a spawned worker.
        if not _is_distributed_worker():
            nproc = _resolve_nproc(config.distributed)
            # Speech GSPO/GRPO (rl_speech) is single-GPU only; don't self-launch DDP for it
            # (the grad forward calls a custom method DDP can't proxy). Guarded again in
            # SpeechGSPOTrainer.__init__.
            from ..config.schema import Modality

            if config.modality is Modality.SPEECH and args.command in ("grpo", "gspo") and nproc > 1:
                print("[glide] speech RL is single-GPU only; not launching under torchrun "
                      "(pin distributed.nproc_per_node: 1 to silence this).")
                nproc = 1
            if nproc > 1:
                return _relaunch_distributed(argv, config.distributed, nproc)
        return _run_training(config, args.command)
    if args.command == "eval":
        return _run_generation_eval(
            args.config, overrides, split="eval", checkpoint=args.checkpoint
        )
    if args.command == "test":
        return _run_generation_eval(
            args.config, overrides, split="test", checkpoint=args.checkpoint, output=args.output
        )
    if args.command == "docs":
        return _run_docs(args.output, args.serve)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

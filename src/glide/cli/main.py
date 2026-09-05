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

Each command is a :class:`Command` subclass that owns both its argparse flags
and its execution, registered in :data:`COMMANDS`. Adding a command means adding
a subclass and listing it there; :func:`_build_parser` and :func:`main` need no
changes.
"""

import argparse
import os
import sys

from .. import __version__

__all__ = ["Command", "TrainCommand", "EvalCommand", "TestCommand", "DocsCommand",
           "COMMANDS", "main"]


class Command:
    """One ``glide <name>`` subcommand: its flags and how to run it."""

    #: Subcommand name as typed on the command line.
    name: str = ""
    #: One-line help shown in ``glide --help``.
    help: str = ""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Declare this command's flags on its own subparser."""

    def run(self, args: argparse.Namespace, overrides: list[str], argv: list[str]) -> int:
        """Execute the command.

        Args:
            args: Parsed known arguments for this subcommand.
            overrides: Leftover ``--dotted.key value`` tokens for the config loader.
            argv: The full original argument list, needed to re-exec under torchrun.

        Returns:
            The process exit code.
        """
        raise NotImplementedError


class _ConfigCommand(Command):
    """Base for commands whose first positional is an optional run YAML."""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("config", nargs="?", default=None, help="Path to the run YAML.")


class TrainCommand(_ConfigCommand):
    """Training: ``sft``, ``grpo`` or ``gspo``.

    Self-launches under ``torch.distributed.run`` when the YAML ``distributed``
    section resolves to more than one process and we are not already a worker.
    """

    def __init__(self, task: str) -> None:
        self.name = task
        self.help = f"Run {task.upper()} training."

    def run(self, args: argparse.Namespace, overrides: list[str], argv: list[str]) -> int:
        from ..config.loader import load_config

        config = load_config(args.config, overrides, task=self.name)
        if not self._is_distributed_worker():
            nproc = self._resolve_nproc(config.distributed)
            nproc = self._clamp_for_speech_rl(config, nproc)
            if nproc > 1:
                return self._relaunch_distributed(argv, config.distributed, nproc)
        return self._train(config)

    # -- distributed self-launch -------------------------------------------------

    @staticmethod
    def _is_distributed_worker() -> bool:
        """True if we're already running inside a torchrun/torch.distributed worker.

        ``RANK``/``LOCAL_RANK`` are set by ``torch.distributed.run`` itself (not by
        the user), so we use them only to detect "am I a spawned worker?".
        """
        return any(k in os.environ for k in ("RANK", "LOCAL_RANK", "TORCHELASTIC_RUN_ID"))

    @staticmethod
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
                f"[glide] distributed.nproc_per_node={n} > visible GPUs ({ngpu}); "
                f"clamping to {ngpu}."
            )
            n = ngpu
        return n

    def _clamp_for_speech_rl(self, config, nproc: int) -> int:
        """Force single-process for speech RL, which DDP cannot proxy.

        The grad forward calls a custom method DDP can't wrap. Guarded again in
        ``SpeechGSPOTrainer.__init__``.
        """
        from ..config.schema import Modality

        if config.modality is Modality.SPEECH and self.name in ("grpo", "gspo") and nproc > 1:
            print("[glide] speech RL is single-GPU only; not launching under torchrun "
                  "(pin distributed.nproc_per_node: 1 to silence this).")
            return 1
        return nproc

    @staticmethod
    def _relaunch_distributed(argv: list[str], dist_cfg, nproc: int) -> int:
        """Re-exec the current command under ``torch.distributed.run`` (DDP).

        Driven entirely by the YAML ``distributed`` config (no env vars required).
        Single-node uses ``--standalone``; multi-node uses the node/master settings.
        """
        import datetime as _dt

        from torch.distributed.run import main as torchrun_main

        # Stamp the versioned output dir ONCE here (parent) so every torchrun worker
        # inherits the same value -> all ranks share one output dir (no per-rank stamp).
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

    # -- training ----------------------------------------------------------------

    def _train(self, config) -> int:
        from transformers import set_seed

        from ..logging_utils import setup_logging, snapshot_config
        from ..trainers import build_trainer

        set_seed(config.seed)
        setup_logging(config)

        trainer = build_trainer(config)
        output_dir = trainer.args.output_dir
        snapshot_config(config, output_dir)
        print(f"[glide] {self.name} -> output_dir={output_dir}")

        trainer.train()
        trainer.save_model(output_dir)
        if getattr(trainer, "processing_class", None) is not None:
            trainer.processing_class.save_pretrained(output_dir)
        print(f"[glide] done. Model saved to {output_dir}")
        return 0


class _GenerationCommand(_ConfigCommand):
    """Base for ``eval`` and ``test``: decode one data split and score it.

    Subclasses differ only in :attr:`split` -- which ``data.*`` field is read,
    whether ``eval.generate.max_eval_samples`` caps the record count, and the
    metric prefix.
    """

    #: Name of the ``data`` field to score, and the metric prefix.
    split: str = "eval"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument(
            "--checkpoint",
            "-c",
            default=None,
            help="Model checkpoint path or HF hub id (overrides model.name).",
        )

    def run(self, args: argparse.Namespace, overrides: list[str], argv: list[str]) -> int:
        return self._evaluate(
            args.config, overrides, args.checkpoint, getattr(args, "output", None)
        )

    def _evaluate(
        self,
        config_path,
        overrides: list[str],
        checkpoint: str | None,
        output: str | None,
    ) -> int:
        """Load a checkpoint and score :attr:`split` by autoregressive decoding.

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

        import torch

        if torch.cuda.is_available():
            loaded.model.to("cuda")

        paths = getattr(config.data, self.split)
        if paths is None:
            print(
                f"[glide] no data.{self.split} configured; nothing to evaluate.",
                file=sys.stderr,
            )
            return 1
        if not isinstance(paths, list):
            paths = [paths]
        records = []
        for path in paths:
            records.extend(read_jsonl(path))

        # Test scores the full set; max_eval_samples only caps validation.
        evaluator = GenerationEvaluator(
            config, loaded.processor, records, cap_samples=(self.split == "eval")
        )
        metrics = evaluator.evaluate(loaded.model, save_path=output, prefix=self.split)
        print(f"[glide][{self.split}]", metrics)
        return 0


class EvalCommand(_GenerationCommand):
    """Validation decoding + metrics over ``data.eval`` (capped by max_eval_samples)."""

    name = "eval"
    help = "Run AR-decoding evaluation + metrics."
    split = "eval"


class TestCommand(_GenerationCommand):
    """Held-out decoding + metrics over all of ``data.test`` (never capped)."""

    name = "test"
    help = "Run held-out test evaluation on data.test."
    split = "test"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument(
            "--output", "-o", default=None, help="Save per-sample predictions to this JSONL file."
        )


class DocsCommand(Command):
    """Generate the pdoc API reference for every importable ``glide`` module."""

    name = "docs"
    help = "Generate API docs from docstrings (pdoc)."

    _GITHUB_REPO = "https://github.com/nagohachi/glide"
    _GITHUB_SRC = f"{_GITHUB_REPO}/blob/main/src/glide/"
    # docs/templates is four levels above this file: src/glide/cli/main.py -> repo root
    _TEMPLATE_DIR = str(__import__("pathlib").Path(__file__).parents[3] / "docs" / "templates")

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--output", "-o", default="docs/api", help="Output directory.")
        parser.add_argument(
            "--serve", action="store_true", help="Serve docs instead of writing HTML."
        )

    def run(self, args: argparse.Namespace, overrides: list[str], argv: list[str]) -> int:
        import subprocess

        modules = self._discover_modules()
        cmd = [
            "pdoc",
            *modules,
            "--edit-url",
            f"glide={self._GITHUB_SRC}",
            "--template-directory",
            self._TEMPLATE_DIR,
        ]
        # Modern pdoc (>=14) serves at http://localhost:8080 by default when no output
        # dir is given; the old pdoc3 `--http :8080` flag no longer exists.
        if not args.serve:
            cmd += ["-o", args.output]
        print(
            f"[glide] generating docs for {len(modules)} modules -> "
            f"{'http://localhost:8080' if args.serve else args.output}"
        )
        return subprocess.call(cmd)

    @staticmethod
    def _discover_modules() -> list[str]:
        """Return ``glide`` plus every importable submodule (for full pdoc coverage)."""
        import pkgutil

        import glide

        mods = ["glide"]
        for info in pkgutil.walk_packages(glide.__path__, prefix="glide."):
            mods.append(info.name)
        return mods


_TRAIN_TASKS = {"sft", "grpo", "gspo"}

#: Every subcommand, in ``glide --help`` order.
COMMANDS: dict[str, Command] = {
    c.name: c
    for c in [
        *(TrainCommand(task) for task in sorted(_TRAIN_TASKS)),
        EvalCommand(),
        TestCommand(),
        DocsCommand(),
    ]
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glide", description="TRL-based post-training library.")
    parser.add_argument("--version", action="version", version=f"glide {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS.values():
        command.add_arguments(sub.add_parser(command.name, help=command.help))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``glide`` console script."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args, overrides = parser.parse_known_args(argv)

    command = COMMANDS.get(args.command)
    if command is None:
        parser.error(f"Unknown command: {args.command}")
        return 2
    return command.run(args, overrides, argv)


if __name__ == "__main__":
    raise SystemExit(main())

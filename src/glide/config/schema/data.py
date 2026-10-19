"""Dataset locations, chat templating and sequence packing."""

from dataclasses import dataclass, field
from typing import Literal

__all__ = ["DataConfig", "TemplateConfig", "PackingConfig"]


@dataclass
class DataConfig:
    """JSONL dataset locations and field mapping."""

    #: The data root is prepended to every relative ``*_jsonl_path``. Give it
    #: either way, never both -- ``root_key`` wins if you do:
    #:
    #: * ``root_key``: a key into the top-level ``data_roots`` map, so the machine
    #:   -specific absolute path lives in a gitignored ``data_root.yaml`` pulled in
    #:   via ``extends`` (see ``configs/data_root.example.yaml``). Use this when the
    #:   same config runs on several machines.
    #: * ``root_dir``: the absolute path written directly in this config. Use this
    #:   for a one-off run where the indirection buys nothing.
    #:
    #: Leave both unset to treat every ``*_jsonl_path`` as already absolute.
    root_key: str | None = None
    root_dir: str | None = None
    #: Path(s) to training JSONL file(s) (relative to the data root unless absolute).
    train_jsonl_path: str | list[str] | None = None
    #: Path(s) to evaluation/validation JSONL file(s) (relative to the data root).
    eval_jsonl_path: str | list[str] | None = None
    #: Path(s) to held-out test JSONL file(s) evaluated once at the end of training.
    test_jsonl_path: str | list[str] | None = None
    #: Read records lazily by byte offset instead of loading into memory.
    lazy: bool = True
    #: Field names in each JSONL record.
    messages_field: str = "messages"  # chat-style records
    prompt_field: str = "prompt"  # prompt/response or RL records
    response_field: str = "response"
    #: For speech/vision: the field holding the media path / array.
    audio_field: str = "audio"
    image_field: str = "image"
    #: Optional per-record fields carrying the audio length for the length-grouped
    #: sampler, avoiding a filesystem header read per utterance. ``duration_field`` is
    #: in seconds (multiplied by ``speech.sample_rate``); ``num_samples_field`` is the
    #: raw sample count. Either, when present on a record, is used verbatim.
    duration_field: str | None = None
    num_samples_field: str | None = None
    #: Optional reference text field used by validation metrics (WER/CER/BLEU).
    reference_field: str = "reference"
    #: Field holding the assistant reasoning/CoT for thinking-mode SFT. When present
    #: (and ``template.enable_thinking`` is True) it is passed as the chat-template
    #: ``reasoning_content`` so the target becomes ``<think>{reasoning}</think>{answer}``.
    reasoning_field: str = "reasoning"
    #: Cap the number of samples (debugging). ``None`` = all.
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    #: Dataloader worker processes. ``0`` loads in the main process (the PyTorch
    #: default); >0 forks workers, which the lazy JSONL reader is fork-safe for.
    num_workers: int = 0


@dataclass
class TemplateConfig:
    """Prompt/chat-template construction and loss masking."""

    #: Named template in the registry, or ``None`` to use the tokenizer's
    #: built-in chat template.
    name: str | None = None
    #: Train only on the assistant/response tokens (completion-only masking).
    train_on_completions_only: bool = True
    #: When chat templates are used, the substring that marks the start of the
    #: assistant turn (used to locate the response span for masking).
    response_template: str | None = None
    instruction_template: str | None = None
    #: Add a generation prompt suffix when formatting prompts for generation.
    add_generation_prompt: bool = True
    #: System prompt prepended to every conversation when one is not present.
    system_prompt: str | None = None
    max_length: int = 2048
    #: Thinking mode for models with a reasoning toggle (e.g. Qwen3). ``True`` keeps
    #: ``<think>`` reasoning (SFT supervises ``<think>...</think>`` + answer);
    #: ``False`` injects an empty ``<think></think>`` (SFT supervises the answer only);
    #: ``None`` uses the template default. Passed to ``apply_chat_template`` for both
    #: the prompt prefix and the target so completion masking stays aligned.
    enable_thinking: bool | None = None
    #: Name of a registered custom data collator (plugin) to use for the
    #: multimodal SFT path instead of the built-in :class:`MultimodalSFTCollator`.
    #: The builder is called as ``builder(config, processor)``.
    collator: str | None = None


@dataclass
class PackingConfig:
    """Sequence packing settings (forwarded to TRL when applicable)."""

    enabled: bool = False
    #: ``ffd`` (first-fit-decreasing) or ``wrapped``; matches TRL's options.
    strategy: Literal["ffd", "wrapped"] = "ffd"
    max_length: int | None = None

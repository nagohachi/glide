"""Data pipeline: JSONL loading, templating, collators and the speech sampler."""

from .audio import audio_num_samples, load_audio
from .build import (
    build_multimodal_dataset,
    build_rl_text_dataset,
    build_sft_text_dataset,
    compute_audio_lengths,
)
from .collator import MultimodalSFTCollator, find_subsequence
from .jsonl import JsonlDataset, load_hf_dataset, read_jsonl
from .packing import (
    block_diagonal_causal_mask,
    cu_seqlens,
    packed_position_ids,
    segment_ids,
)
from .sampler import LengthGroupedBatchSampler
from .template import (
    build_messages,
    build_prompt_messages,
    extract_media,
    extract_reference,
)

__all__ = [
    "JsonlDataset",
    "load_hf_dataset",
    "read_jsonl",
    "LengthGroupedBatchSampler",
    "MultimodalSFTCollator",
    "find_subsequence",
    "block_diagonal_causal_mask",
    "cu_seqlens",
    "packed_position_ids",
    "segment_ids",
    "build_messages",
    "build_prompt_messages",
    "extract_media",
    "extract_reference",
    "build_sft_text_dataset",
    "build_rl_text_dataset",
    "build_multimodal_dataset",
    "compute_audio_lengths",
    "load_audio",
    "audio_num_samples",
]

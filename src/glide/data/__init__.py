"""Data pipeline: JSONL loading, templating, collators and the speech sampler."""

from .audio import count_audio_samples, load_audio
from .build import (
    build_multimodal_dataset,
    build_rl_text_dataset,
    build_sft_text_dataset,
    compute_audio_lengths,
)
from .collator import MultimodalSFTCollator, find_subsequence, load_media
from .jsonl import JsonlDataset, load_hf_dataset, read_jsonl, read_jsonl_files
from .packing import (
    build_block_diagonal_causal_mask,
    compute_cu_seqlens,
    compute_packed_position_ids,
    compute_segment_ids,
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
    "read_jsonl_files",
    "LengthGroupedBatchSampler",
    "MultimodalSFTCollator",
    "find_subsequence",
    "load_media",
    "build_block_diagonal_causal_mask",
    "compute_cu_seqlens",
    "compute_packed_position_ids",
    "compute_segment_ids",
    "build_messages",
    "build_prompt_messages",
    "extract_media",
    "extract_reference",
    "build_sft_text_dataset",
    "build_rl_text_dataset",
    "build_multimodal_dataset",
    "compute_audio_lengths",
    "load_audio",
    "count_audio_samples",
]

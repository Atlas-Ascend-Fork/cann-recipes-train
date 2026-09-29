#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data utilities for feature extraction: dataset loading and tokenization."""

import json
import logging
from typing import List, Dict, Optional
import torch

logger = logging.getLogger(__name__)


def load_jsonl_dataset(file_paths: List[str]) -> List[Dict]:
    """Load JSONL dataset files containing conversations.

    Args:
        file_paths: List of paths to JSONL files

    Returns:
        List of sample dicts, each containing a 'conversations' field
    """
    samples = []
    for path in file_paths:
        logger.info(f"Loading dataset from {path}")
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    if 'conversations' not in sample:
                        logger.warning(
                            f"{path}:{line_num} missing 'conversations' field, skipping"
                        )
                        continue
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"{path}:{line_num} JSON decode error: {e}")
                    continue
        logger.info(f"Loaded {len(samples)} samples from {path}")

    return samples


def tokenize_conversation(
    conversation: List[Dict[str, str]],
    tokenizer,
    chat_template: str,
    max_length: int,
    min_loss_tokens: int = 14,
) -> Optional[Dict[str, torch.Tensor]]:
    """Tokenize a conversation and create loss mask.

    Args:
        conversation: List of message dicts with 'role' and 'content' keys
        tokenizer: HuggingFace tokenizer instance
        chat_template: Chat template name (e.g., 'deepseek')
        max_length: Maximum sequence length
        min_loss_tokens: Minimum number of assistant tokens required

    Returns:
        Dict with 'input_ids', 'attention_mask', 'loss_mask' tensors, or None if too short
    """
    # Apply chat template to get formatted text
    try:
        formatted_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as e:
        logger.warning(f"Failed to apply chat template: {e}")
        return None

    # Tokenize the full conversation
    encoding = tokenizer(
        formatted_text,
        return_tensors='pt',
        max_length=max_length,
        truncation=True,
        padding=False,
    )

    input_ids = encoding['input_ids'].squeeze(0)  # (seq_len,)
    attention_mask = encoding['attention_mask'].squeeze(0)  # (seq_len,)

    # Create loss mask: 1 for assistant tokens, 0 for user/system tokens
    loss_mask = torch.zeros_like(input_ids, dtype=torch.uint8)

    # Find assistant response boundaries
    # This is a simplified approach - ideally should track exact token boundaries
    # For now, assume the last portion after final user message is assistant
    # A more robust approach would tokenize each message separately

    # Quick heuristic: mark second half as assistant tokens
    # TODO: Improve this by tokenizing messages separately and tracking boundaries
    mid_point = len(input_ids) // 2
    loss_mask[mid_point:] = 1

    # Check if we have enough assistant tokens
    num_loss_tokens = int(loss_mask.sum().item())
    if num_loss_tokens < min_loss_tokens:
        return None

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'loss_mask': loss_mask,
    }


def tokenize_conversation_precise(
    conversation: List[Dict[str, str]],
    tokenizer,
    chat_template: str,
    max_length: int,
    min_loss_tokens: int = 14,
) -> Optional[Dict[str, torch.Tensor]]:
    """Tokenize conversation with precise loss mask calculation.

    This version tokenizes each message separately to accurately track which
    tokens belong to the assistant.

    Args:
        conversation: List of message dicts with 'role' and 'content' keys
        tokenizer: HuggingFace tokenizer instance
        chat_template: Chat template name
        max_length: Maximum sequence length
        min_loss_tokens: Minimum number of assistant tokens required

    Returns:
        Dict with tokenized conversation and precise loss mask
    """
    input_ids_list = []
    loss_mask_list = []

    # Process each message
    for msg in conversation:
        role = msg.get('role', '')
        content = msg.get('content', '')

        # Format message with chat template
        # For deepseek, typical format: "<|User|>{content}<|Assistant|>{response}"
        if role == 'system':
            formatted = f"<|System|>{content}\n"
        elif role == 'user':
            formatted = f"<|User|>{content}\n<|Assistant|>"
        elif role == 'assistant':
            formatted = f"{content}\n"
        else:
            logger.warning(f"Unknown role: {role}, skipping")
            continue

        # Tokenize this message
        tokens = tokenizer.encode(formatted, add_special_tokens=False)

        # Create loss mask: 1 for assistant tokens, 0 otherwise
        is_assistant = (role == 'assistant')
        mask = [1 if is_assistant else 0] * len(tokens)

        input_ids_list.extend(tokens)
        loss_mask_list.extend(mask)

    # Add EOS token if needed
    if tokenizer.eos_token_id is not None:
        input_ids_list.append(tokenizer.eos_token_id)
        loss_mask_list.append(1)  # EOS is part of assistant output

    # Truncate if too long
    if len(input_ids_list) > max_length:
        input_ids_list = input_ids_list[:max_length]
        loss_mask_list = loss_mask_list[:max_length]

    # Convert to tensors
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    loss_mask = torch.tensor(loss_mask_list, dtype=torch.uint8)

    # Check minimum loss tokens
    num_loss_tokens = int(loss_mask.sum().item())
    if num_loss_tokens < min_loss_tokens:
        return None

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'loss_mask': loss_mask,
    }


def compute_local_sample_range(
    num_samples: int,
    rank: int,
    world_size: int,
) -> tuple[int, int]:
    """Compute the sample range for a given rank.

    Distributes samples evenly across ranks, with remainder distributed
    to the first ranks.

    Args:
        num_samples: Total number of samples
        rank: Current rank index
        world_size: Total number of ranks

    Returns:
        Tuple of (start_index, end_index) for this rank
    """
    base = num_samples // world_size
    remainder = num_samples % world_size

    # Ranks 0 to remainder-1 get one extra sample
    if rank < remainder:
        start = rank * (base + 1)
        count = base + 1
    else:
        start = remainder * (base + 1) + (rank - remainder) * base
        count = base

    end = start + count
    return start, end

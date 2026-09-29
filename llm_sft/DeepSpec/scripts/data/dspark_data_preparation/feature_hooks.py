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

"""Hook manager for capturing intermediate features from DeepSeek-V4 model."""

from typing import Dict, List, Optional
import torch
import logging

logger = logging.getLogger(__name__)


class FeatureHookManager:
    """Manages forward hooks for capturing intermediate hidden states.

    This class registers hooks on specified decoder layers and captures their
    outputs during forward passes. Designed to work with the existing executor
    inference pipeline without modifying the model code.

    Attributes:
        model: DeepseekV3ForCausalLM instance from executor
        target_layer_ids: List of decoder layer indices to capture
        captured: Dict storing captured tensors by layer
        handles: List of hook handles for cleanup
    """

    def __init__(self, model, target_layer_ids: List[int]):
        """Initialize hook manager.

        Args:
            model: The loaded DeepseekV3ForCausalLM model instance
            target_layer_ids: List of layer indices to capture (e.g., [40, 41, 42])
        """
        self.model = model
        self.target_layer_ids = sorted(target_layer_ids)
        self.captured: Dict[str, torch.Tensor] = {}
        self.handles = []

    def register_hooks(self):
        """Register forward hooks on target layers and model output."""
        # Access the model backbone (DeepseekV3Model)
        # model.model is the DeepseekV3Model instance
        backbone = self.model.model

        logger.info(f"Registering hooks on layers: {self.target_layer_ids}")

        # Register hooks on specified decoder layers
        for layer_id in self.target_layer_ids:
            if layer_id < 0 or layer_id >= len(backbone.layers):
                raise ValueError(
                    f"Invalid layer_id {layer_id}. Model has {len(backbone.layers)} layers."
                )

            layer = backbone.layers[layer_id]
            handle = layer.register_forward_hook(self._make_layer_hook(layer_id))
            self.handles.append(handle)
            logger.debug(f"Registered hook on layer {layer_id}")

        # Register hook for final model output (before lm_head)
        handle = backbone.register_forward_hook(self._capture_last_hidden)
        self.handles.append(handle)
        logger.debug("Registered hook for last hidden states")

        logger.info(f"Successfully registered {len(self.handles)} hooks")

    def _make_layer_hook(self, layer_id: int):
        """Create a hook function for a specific layer.

        Args:
            layer_id: The decoder layer index

        Returns:
            Hook function that captures layer output
        """
        def hook(module, inputs, output):
            # Extract tensor from layer output
            tensor = self._extract_tensor(output)
            # Store on CPU to avoid GPU memory buildup
            self.captured[f"layer_{layer_id}"] = tensor.detach().cpu()
        return hook

    def _capture_last_hidden(self, module, inputs, output):
        """Hook to capture final model output (before lm_head).

        Args:
            module: The DeepseekV3Model module
            inputs: Forward pass inputs (unused)
            output: Model output tensor
        """
        # For DeepseekV3Model, output is the hidden states before lm_head
        tensor = self._extract_tensor(output)
        self.captured["last_hidden"] = tensor.detach().cpu()

    def _extract_tensor(self, output) -> torch.Tensor:
        """Extract hidden state tensor from layer output.

        DeepSeek-V4 layers may return:
            - Packed TND: (num_tokens, hc_mult, hidden_size)  [layer hook]
            - Packed TND: (num_tokens, hidden_size)           [backbone output, post hc_head]
            - A tuple: (hidden_states, ...)

            The raw tensor is returned as-is; no folding/squeezing is applied so
            downstream consumers receive the original hyper-connection features.

            Args:
                output: Layer output (tensor or tuple)

            Returns:
                Extracted hidden state tensor, shape unchanged.
        """
        if isinstance(output, torch.Tensor):
            tensor = output
        elif isinstance(output, (tuple, list)) and len(output) > 0:
            # Take first element of tuple
            first = output[0]
            if isinstance(first, torch.Tensor):
                tensor = first
            else:
                raise TypeError(
                    f"Expected tensor in tuple output, got {type(first)}"
                )
        else:
            raise TypeError(f"Unsupported layer output type: {type(output)}")

        return tensor

    def get_captured_features(self) -> Dict[str, torch.Tensor]:
        """Retrieve captured features and concatenate target layers.

        Returns:
            Dict with keys:
                - 'target_hidden': Concatenated target layer outputs
                  Shape: (batch, seq_len, num_layers * hidden_size)
                - 'last_hidden': Final model output
                  Shape: (batch, seq_len, hidden_size)
        """
        if "last_hidden" not in self.captured:
            raise RuntimeError(
                "No features captured. Did you run a forward pass after "
                "registering hooks?"
            )

        # Collect target layer features in order
        target_hidden_list = []
        for layer_id in self.target_layer_ids:
            key = f"layer_{layer_id}"
            if key not in self.captured:
                raise RuntimeError(
                    f"Layer {layer_id} was not captured. "
                    f"Available keys: {list(self.captured.keys())}"
                )
            target_hidden_list.append(self.captured[key])

        # Concatenate along hidden dimension
        # Each tensor: (batch, seq_len, hidden_size)
        # Result: (batch, seq_len, num_layers * hidden_size)
        target_hidden = torch.cat(target_hidden_list, dim=-1)

        return {
            "target_hidden": target_hidden,
            "last_hidden": self.captured["last_hidden"],
        }

    def clear_captured(self):
        """Clear captured features to free memory."""
        self.captured.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        logger.info("Removed all hooks")

    def __del__(self):
        """Cleanup hooks on deletion."""
        if self.handles:
            self.remove_hooks()

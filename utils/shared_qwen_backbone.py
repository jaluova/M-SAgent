from __future__ import annotations

import numpy as np
import torch
from PIL import Image


class SharedQwenBackbone:
    """Expose the subset of Qwen2.5-VL needed by GridGround without loading a second model."""

    def __init__(self, qwen_model, qwen_processor):
        self.qwen_model = qwen_model
        self.qwen_processor = qwen_processor
        self.tokenizer = getattr(qwen_processor, "tokenizer", None)
        self.visual_dim = qwen_model.config.hidden_size
        self.text_dim = qwen_model.config.hidden_size
        self.merge_size = getattr(qwen_processor.image_processor, "merge_size", 1)

    def _module_device(self, module):
        try:
            return next(module.parameters()).device
        except StopIteration:
            return next(self.qwen_model.parameters()).device

    def visual_device(self):
        if hasattr(self.qwen_model, "visual"):
            return self._module_device(self.qwen_model.visual)
        return self._module_device(self.qwen_model)

    def text_device(self):
        if hasattr(self.qwen_model, "model") and hasattr(self.qwen_model.model, "embed_tokens"):
            return self._module_device(self.qwen_model.model.embed_tokens)
        if hasattr(self.qwen_model, "get_input_embeddings"):
            return self._module_device(self.qwen_model.get_input_embeddings())
        return self._module_device(self.qwen_model)

    def _tensor_to_pil(self, image_tensor):
        image = image_tensor.detach().float().cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = image.clamp(0, 1)
        image = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(image)

    def _pack_visual_outputs(self, outputs, image_grid_thw):
        token_counts = (
            image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
        ) // (self.merge_size ** 2)
        token_counts = token_counts.tolist()

        chunks = []
        start = 0
        for count in token_counts:
            end = start + count
            chunks.append(outputs[start:end])
            start = end

        max_tokens = max(token_counts)
        padded = outputs.new_zeros((len(chunks), max_tokens, outputs.shape[-1]))
        for idx, chunk in enumerate(chunks):
            padded[idx, :chunk.shape[0]] = chunk
        return padded

    def encode_image(self, images):
        if not hasattr(self.qwen_model, "visual"):
            raise AttributeError("Qwen model does not expose a supported visual encoder")

        pil_images = [self._tensor_to_pil(image) for image in images]
        image_inputs = self.qwen_processor.image_processor(
            pil_images,
            return_tensors="pt",
        )
        device = self.visual_device()
        pixel_values = image_inputs["pixel_values"].to(device)
        image_grid_thw = image_inputs["image_grid_thw"].to(device)

        with torch.no_grad():
            outputs = self.qwen_model.visual(pixel_values, image_grid_thw)
        return self._pack_visual_outputs(outputs, image_grid_thw)

    def encode_text(self, input_ids, attention_mask=None):
        device = self.text_device()
        input_ids = input_ids.to(device)
        if hasattr(self.qwen_model, "model") and hasattr(self.qwen_model.model, "embed_tokens"):
            with torch.no_grad():
                return self.qwen_model.model.embed_tokens(input_ids)
        if hasattr(self.qwen_model, "get_input_embeddings"):
            with torch.no_grad():
                return self.qwen_model.get_input_embeddings()(input_ids)
        raise AttributeError("Qwen model does not expose a supported text embedding layer")

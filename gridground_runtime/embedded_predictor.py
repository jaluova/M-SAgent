from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from config import Config
from utils.localization import LocalizationResult

from .adapter import CoordinateAdapter, LightweightCoordinateAdapter


@dataclass
class EmbeddedGridGroundConfig:
    adapter_type: str = "lightweight"
    visual_dim: int = 768
    grid_feature_dim: int = 256
    hidden_dim: int = 256
    num_heads: int = 4
    num_grid_tokens: int = 25
    num_output_points: int = 4
    dropout: float = 0.1
    output_mode: str = "grid_logits"
    grid_size: int = 11
    max_length: int = 512

    @classmethod
    def from_json_file(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = payload.get("model", {})
        data = payload.get("data", {})
        return cls(
            adapter_type=model.get("adapter_type", cls.adapter_type),
            visual_dim=int(model.get("visual_dim", cls.visual_dim)),
            grid_feature_dim=int(model.get("grid_feature_dim", cls.grid_feature_dim)),
            hidden_dim=int(model.get("hidden_dim", cls.hidden_dim)),
            num_heads=int(model.get("num_heads", cls.num_heads)),
            num_grid_tokens=int(model.get("num_grid_tokens", cls.num_grid_tokens)),
            num_output_points=int(model.get("num_output_points", cls.num_output_points)),
            dropout=float(model.get("dropout", cls.dropout)),
            output_mode=model.get("output_mode", cls.output_mode),
            grid_size=int(model.get("grid_size", cls.grid_size)),
            max_length=int(data.get("max_length", cls.max_length)),
        )


class EmbeddedGridGroundPredictor:
    def __init__(self, shared_backbone, config=Config):
        self.config = config
        self.shared_backbone = shared_backbone
        self.model_dir = Path(config.GRIDGROUND_MODEL_DIR).expanduser() if config.GRIDGROUND_MODEL_DIR else None
        self.config_path = Path(config.GRIDGROUND_CONFIG_PATH).expanduser()
        self.adapter_path = Path(config.GRIDGROUND_ADAPTER_PATH).expanduser()
        self.runtime_config = EmbeddedGridGroundConfig.from_json_file(self.config_path)
        self.device = shared_backbone.visual_device()
        self.tokenizer = shared_backbone.tokenizer
        if self.tokenizer is None:
            raise RuntimeError("Shared Qwen backbone does not expose a tokenizer")
        self.adapter = self._build_adapter().to(self.device)
        self._load_adapter_weights()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _build_adapter(self):
        kwargs = dict(
            visual_dim=self.runtime_config.visual_dim,
            grid_feature_dim=self.runtime_config.grid_feature_dim,
            hidden_dim=self.runtime_config.hidden_dim,
            num_heads=self.runtime_config.num_heads,
            num_grid_tokens=self.runtime_config.num_grid_tokens,
            num_output_points=self.runtime_config.num_output_points,
            dropout=self.runtime_config.dropout,
            output_mode=self.runtime_config.output_mode,
            grid_size=self.runtime_config.grid_size,
        )
        if self.runtime_config.adapter_type == "lightweight":
            return LightweightCoordinateAdapter(**kwargs)
        return CoordinateAdapter(**kwargs)

    def _load_adapter_weights(self):
        checkpoint = torch.load(self.adapter_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        missing_keys, unexpected_keys = self.adapter.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Embedded GridGround missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Embedded GridGround unexpected keys: {unexpected_keys}")
        self.adapter.eval()

    def describe_runtime(self):
        return {
            "ok": True,
            "backend": "embedded",
            "device": str(self.device),
            "shared_qwen": True,
            "model_dir": str(self.model_dir) if self.model_dir else "",
            "config_path": str(self.config_path),
            "adapter_path": str(self.adapter_path),
        }

    @staticmethod
    def _load_font(size, bold=False):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
        for path in candidates:
            if Path(path).exists():
                try:
                    return ImageFont.truetype(path, size)
                except OSError:
                    pass
        return ImageFont.load_default()

    def build_grid_image(self, image):
        image = image.convert("RGB")
        width, height = image.size
        border_size = 28
        new_width = width + border_size * 2
        new_height = height + border_size * 2
        grid_image = Image.new("RGB", (new_width, new_height), "white")
        grid_image.paste(image, (border_size, border_size))
        draw = ImageDraw.Draw(grid_image)
        grid_font = self._load_font(15, bold=False)

        for index in range(11):
            x = border_size + index * (width / 10.0)
            y = border_size + index * (height / 10.0)
            draw.line([(x, border_size), (x, border_size + height)], fill="black", width=1)
            draw.line([(border_size, y), (border_size + width, y)], fill="black", width=1)
            label = str(index)
            bbox = draw.textbbox((0, 0), label, font=grid_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.text((x - text_w / 2, border_size - text_h - 5), label, fill="black", font=grid_font)
            draw.text((border_size - text_w - 5, y - text_h / 2), label, fill="black", font=grid_font)

        return grid_image

    def preprocess_pil_image(self, image, grid_image=None):
        image = image.convert("RGB")
        original_size = image.size
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        if grid_image is None:
            grid_image = self.build_grid_image(image)
        grid_image_tensor = self.transform(grid_image.convert("RGB")).unsqueeze(0).to(self.device)
        return image_tensor, grid_image_tensor, original_size

    def _build_instruction(self, query):
        return (
            "Given the grid coordinate system, locate the referent described as "
            f"'{query}' in the image and predict the most likely target points."
        )

    def _encode_text(self, instruction):
        encoding = self.tokenizer(
            instruction,
            padding="max_length",
            truncation=True,
            max_length=self.runtime_config.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.shared_backbone.text_device())
        attention_mask = encoding["attention_mask"].to(self.shared_backbone.text_device())
        return input_ids, attention_mask

    def _to_absolute_points(self, normalized_points, image_size):
        width, height = image_size
        return [
            [round(float(x) * width, 2), round(float(y) * height, 2)]
            for x, y in normalized_points
        ]

    def render_prediction_overlay(self, image, absolute_points, scores, query):
        canvas = image.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)
        title_font = self._load_font(20, bold=True)
        score_font = self._load_font(16)

        for idx, ((x, y), score) in enumerate(zip(absolute_points, scores), start=1):
            radius = 10
            bbox = [x - radius, y - radius, x + radius, y + radius]
            draw.ellipse(bbox, outline="#dc2626", width=3)
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill="#dc2626")
            draw.text((x + 14, y - 12), f"P{idx}", fill="#dc2626", font=title_font)
            draw.text((x + 14, y + 10), f"{float(score):.2f}", fill="#991b1b", font=score_font)

        draw.rectangle([0, 0, canvas.width, 34], fill=(245, 247, 251))
        draw.text((12, 7), f"Query: {query}", fill="#111827", font=score_font)
        return canvas

    def localize_with_metadata(self, image, query):
        image_tensor, grid_image_tensor, original_size = self.preprocess_pil_image(image)
        instruction = self._build_instruction(query)
        input_ids, attention_mask = self._encode_text(instruction)

        with torch.no_grad():
            visual_features = self.shared_backbone.encode_image(image_tensor)
            adapter_dtype = next(self.adapter.parameters()).dtype
            if visual_features.dtype != adapter_dtype:
                visual_features = visual_features.to(dtype=adapter_dtype)
            text_embeddings = self.shared_backbone.encode_text(input_ids, attention_mask)
            if text_embeddings.dtype != adapter_dtype:
                text_embeddings = text_embeddings.to(dtype=adapter_dtype)

            enhanced = self.adapter(
                image_tensor,
                grid_image_tensor,
                visual_features,
                text_features=text_embeddings,
            )
            grid_logits = self.adapter.predict_grid_logits(
                enhanced,
                text_features=text_embeddings,
                attention_mask=attention_mask,
            )
            selected = self.adapter.decode_grid_logits_dynamic(
                grid_logits,
                abs_threshold=self.config.TRAIN_ADAPTER_ABS_THRESHOLD,
                rel_ratio=self.config.TRAIN_ADAPTER_REL_RATIO,
                min_k=self.config.TRAIN_ADAPTER_MIN_K,
                max_k=self.config.TRAIN_ADAPTER_MAX_K,
            )

        pred_points = selected["selected_points"][0].detach().cpu()
        pred_logits = selected["selected_logits"][0].detach().cpu()
        normalized_points = pred_points.tolist()
        scores = torch.sigmoid(pred_logits).tolist()
        absolute_points = self._to_absolute_points(normalized_points, original_size)
        localization = LocalizationResult(
            absolute_points=absolute_points,
            normalized_points=[[round(float(x), 4), round(float(y), 4)] for x, y in normalized_points],
            scores=[round(float(score), 4) for score in scores],
            selection_mode="dynamic_topk",
            selected_k=int(selected["selected_ks"][0]),
            source="gridground_embedded",
            annotated_image=self.render_prediction_overlay(image, absolute_points, scores, query),
        )
        metadata = {
            "backend": "embedded",
            "service_device": str(self.device),
            "slow_path": False,
            "shared_qwen": True,
            "attempts": [
                {
                    "attempt": 1,
                    "timeout_s": None,
                    "elapsed_ms": None,
                    "ok": True,
                }
            ],
            "retry_used": False,
            "model_dir": str(self.model_dir) if self.model_dir else "",
        }
        return localization, metadata

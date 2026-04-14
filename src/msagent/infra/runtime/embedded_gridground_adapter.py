"""embedded GridGround 所需的最小 adapter 组件。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = F.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        output += self.shortcut(inputs)
        return F.relu(output)


class GridEncoder(nn.Module):
    def __init__(self, input_channels: int = 3, feature_dim: int = 512) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, feature_dim, 2, stride=2)
        self._initialize_weights()

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = F.relu(self.bn1(self.conv1(inputs)))
        output = self.maxpool(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return output


class FeatureProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_tokens: int = 64) -> None:
        super().__init__()
        side = int(num_tokens**0.5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((side, side))
        self.projection = nn.Linear(input_dim, output_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, output_dim))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = inputs.shape
        output = self.adaptive_pool(inputs)
        output = output.view(batch_size, channels, -1).transpose(1, 2)
        output = self.projection(output)
        output = output + self.pos_embedding
        return self.norm(output)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        visual_features: torch.Tensor,
        context_features: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_visual_tokens, dim = visual_features.shape
        _, num_context_tokens, _ = context_features.shape
        residual = visual_features
        query = self.q_proj(visual_features)
        key = self.k_proj(context_features)
        value = self.v_proj(context_features)

        query = query.view(batch_size, num_visual_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, num_context_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, num_context_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        if attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool).unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        context = torch.matmul(attention, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_visual_tokens, dim)
        output = self.out_proj(context)
        return self.norm(residual + output)


class TextGuidedCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.grid_cross_attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.text_cross_attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.merge_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(
        self,
        visual_features: torch.Tensor,
        grid_features: torch.Tensor,
        *,
        text_features: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grid_enhanced = self.grid_cross_attn(visual_features, grid_features)
        if text_features is None:
            return grid_enhanced
        text_enhanced = self.text_cross_attn(
            visual_features,
            text_features,
            attention_mask=text_attention_mask,
        )
        gate = self.merge_gate(torch.cat([grid_enhanced, text_enhanced], dim=-1))
        return gate * text_enhanced + (1 - gate) * grid_enhanced


class GatedFusion(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim * 2, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        original_features: torch.Tensor,
        enhanced_features: torch.Tensor,
    ) -> torch.Tensor:
        concat_features = torch.cat([original_features, enhanced_features], dim=-1)
        gate = torch.sigmoid(self.gate_proj(concat_features))
        fused = gate * enhanced_features + (1 - gate) * original_features
        fused = self.out_proj(fused)
        fused = self.dropout(fused)
        return self.norm(original_features + fused)


class ResidualFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.ffn(inputs)


class AttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score_proj = nn.Linear(dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self.score_proj(features).squeeze(-1)
        if attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(features * weights.unsqueeze(-1), dim=1)


class BaseCoordinateAdapter(nn.Module):
    def __init__(
        self,
        *,
        visual_dim: int = 768,
        grid_feature_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_grid_tokens: int = 64,
        num_output_points: int = 4,
        dropout: float = 0.1,
        grid_size: int = 11,
        ffn_hidden_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.num_output_points = num_output_points
        self.grid_size = grid_size
        self.num_grid_logits = grid_size * grid_size
        self.grid_encoder = GridEncoder(input_channels=3, feature_dim=grid_feature_dim)
        self.grid_projector = FeatureProjector(
            input_dim=grid_feature_dim,
            output_dim=visual_dim,
            num_tokens=num_grid_tokens,
        )
        self.cross_attention = TextGuidedCrossAttention(dim=visual_dim, num_heads=num_heads, dropout=dropout)
        self.gated_fusion = GatedFusion(dim=visual_dim, dropout=dropout)
        self.residual_ffn = ResidualFFN(
            dim=visual_dim,
            hidden_dim=hidden_dim * ffn_hidden_multiplier,
            dropout=dropout,
        )
        self.text_pool = AttentionPool(visual_dim)
        self.visual_condition_proj = nn.Linear(visual_dim, visual_dim)
        self.text_condition_proj = nn.Linear(visual_dim, visual_dim)
        self.visual_query_proj = nn.Linear(visual_dim, visual_dim)
        self.text_key_proj = nn.Linear(visual_dim, visual_dim)
        self.text_value_proj = nn.Linear(visual_dim, visual_dim)
        self.text_context_proj = nn.Linear(visual_dim, visual_dim)
        self.token_modulation = nn.Sequential(
            nn.LayerNorm(visual_dim * 3),
            nn.Linear(visual_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, visual_dim),
            nn.Sigmoid(),
        )
        self.token_text_gate = nn.Sequential(
            nn.LayerNorm(visual_dim * 2),
            nn.Linear(visual_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, visual_dim * 2),
        )
        self.token_score = nn.Linear(visual_dim, 1)
        self.token_text_score = nn.Sequential(
            nn.LayerNorm(visual_dim * 2),
            nn.Linear(visual_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.grid_position_embedding = nn.Parameter(
            self._build_2d_sinusoidal_embedding(grid_size, visual_dim)
        )
        self.grid_classifier = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_grid_logits),
        )
        self._initialize_weights()

    @staticmethod
    def _build_2d_sinusoidal_embedding(grid_size: int, dim: int) -> torch.Tensor:
        num_positions = grid_size * grid_size
        embedding = torch.zeros(num_positions, dim)
        quarter = dim // 4
        for idx in range(num_positions):
            y = idx // grid_size
            x = idx % grid_size
            for channel in range(quarter):
                freq = 1.0 / (10000.0 ** (2.0 * channel / dim))
                embedding[idx, 4 * channel] = math.sin(x * freq)
                embedding[idx, 4 * channel + 1] = math.cos(x * freq)
                embedding[idx, 4 * channel + 2] = math.sin(y * freq)
                embedding[idx, 4 * channel + 3] = math.cos(y * freq)
        return embedding.unsqueeze(0)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        images: torch.Tensor,
        grid_images: torch.Tensor,
        visual_features: torch.Tensor,
        *,
        text_features: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del images
        grid_features_map = self.grid_encoder(grid_images)
        grid_tokens = self.grid_projector(grid_features_map)
        enhanced_features = self.cross_attention(
            visual_features=visual_features,
            grid_features=grid_tokens,
            text_features=text_features,
            text_attention_mask=text_attention_mask,
        )
        fused_features = self.gated_fusion(visual_features, enhanced_features)
        return self.residual_ffn(fused_features)

    def _pool_text_features(
        self,
        *,
        text_features: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        visual_summary: torch.Tensor,
    ) -> torch.Tensor:
        if text_features is None:
            return torch.zeros_like(visual_summary)
        return self.text_pool(text_features, attention_mask=attention_mask)

    def _build_text_conditioning(
        self,
        visual_features: torch.Tensor,
        *,
        text_features: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if text_features is None:
            return torch.zeros_like(visual_features)
        query = self.visual_query_proj(visual_features)
        keys = self.text_key_proj(text_features)
        values = self.text_value_proj(text_features)
        scale = float(self.visual_dim) ** -0.5
        attention_scores = torch.matmul(query, keys.transpose(-1, -2)) * scale
        if attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool).unsqueeze(1)
            attention_scores = attention_scores.masked_fill(~mask, torch.finfo(attention_scores.dtype).min)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        text_context = torch.matmul(attention_weights, values)
        return self.text_context_proj(text_context)

    def predict_grid_logits(
        self,
        visual_features: torch.Tensor,
        *,
        text_features: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual_summary = visual_features.mean(dim=1)
        text_summary = self._pool_text_features(
            text_features=text_features,
            attention_mask=attention_mask,
            visual_summary=visual_summary,
        )
        text_condition = self._build_text_conditioning(
            visual_features,
            text_features=text_features,
            attention_mask=attention_mask,
        )
        conditioned_tokens = (
            self.visual_condition_proj(visual_features)
            + self.text_condition_proj(text_condition)
            + text_condition
        )
        token_modulation = self.token_modulation(
            torch.cat([visual_features, text_condition, text_condition], dim=-1)
        )
        text_gate = self.token_text_gate(torch.cat([text_condition, text_condition], dim=-1))
        gate_scale, gate_bias = torch.chunk(text_gate, chunks=2, dim=-1)
        gate_scale = 0.5 * torch.tanh(gate_scale)
        gate_bias = 0.25 * torch.tanh(gate_bias)
        conditioned_tokens = F.gelu(conditioned_tokens) * (1.0 + token_modulation)
        conditioned_tokens = conditioned_tokens * (1.0 + gate_scale) + gate_bias
        token_logits = self.grid_classifier(conditioned_tokens)
        token_weight_logits = self.token_score(conditioned_tokens).squeeze(-1)
        token_weight_logits = token_weight_logits + self.token_text_score(
            torch.cat([conditioned_tokens, text_condition], dim=-1)
        ).squeeze(-1)
        token_weights = torch.softmax(token_weight_logits, dim=1)
        grid_logits = torch.sum(token_logits * token_weights.unsqueeze(-1), dim=1)
        pos_bias = torch.matmul(
            text_summary.unsqueeze(1),
            self.grid_position_embedding.transpose(-1, -2),
        ).squeeze(1)
        return grid_logits + pos_bias

    def decode_grid_logits(
        self,
        grid_logits: torch.Tensor,
        *,
        top_k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        top_k = top_k or self.num_output_points
        top_k = max(1, min(top_k, self.num_grid_logits))
        values, indices = torch.topk(grid_logits, k=top_k, dim=-1)
        ys = torch.div(indices, self.grid_size, rounding_mode="floor")
        xs = indices % self.grid_size
        denom = float(max(self.grid_size - 1, 1))
        points = torch.stack(
            [xs.to(grid_logits.dtype) / denom, ys.to(grid_logits.dtype) / denom],
            dim=-1,
        )
        return points, values

    @staticmethod
    def select_dynamic_topk(
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
        *,
        abs_threshold: float = 0.35,
        rel_ratio: float = 0.75,
        min_k: int = 1,
        max_k: int = 6,
    ) -> dict[str, torch.Tensor | int]:
        if pred_logits.ndim != 1:
            raise ValueError("pred_logits must be a 1D tensor")
        if pred_points.ndim != 2:
            raise ValueError("pred_points must be a 2D tensor")
        if pred_points.shape[0] != pred_logits.shape[0]:
            raise ValueError("pred_points and pred_logits must contain the same number of candidates")
        if pred_logits.shape[0] == 0:
            return {
                "selected_points": pred_points.new_zeros((0, 2)),
                "selected_logits": pred_logits.new_zeros((0,)),
                "selected_scores": pred_logits.new_zeros((0,)),
                "selected_indices": torch.zeros((0,), dtype=torch.long, device=pred_logits.device),
                "selected_k": 0,
                "candidate_scores": pred_logits.new_zeros((0,)),
            }
        max_k = max(1, min(int(max_k), pred_logits.shape[0]))
        min_k = max(1, min(int(min_k), max_k))
        sorted_logits, sorted_indices = torch.sort(pred_logits, descending=True)
        sorted_points = pred_points[sorted_indices]
        sorted_scores = torch.sigmoid(sorted_logits)
        threshold = torch.maximum(
            sorted_scores.new_tensor(float(abs_threshold)),
            sorted_scores[0] * float(rel_ratio),
        )
        keep_mask = sorted_scores >= threshold
        keep_count = int(keep_mask.sum().item())
        selected_k = max(min_k, min(max_k, keep_count if keep_count > 0 else 1))
        return {
            "selected_points": sorted_points[:selected_k],
            "selected_logits": sorted_logits[:selected_k],
            "selected_scores": sorted_scores[:selected_k],
            "selected_indices": sorted_indices[:selected_k],
            "selected_k": selected_k,
            "candidate_scores": sorted_scores,
        }

    def decode_grid_logits_dynamic(
        self,
        grid_logits: torch.Tensor,
        *,
        abs_threshold: float = 0.35,
        rel_ratio: float = 0.75,
        min_k: int = 1,
        max_k: int = 6,
    ) -> dict[str, list[torch.Tensor] | list[int] | torch.Tensor]:
        candidate_points, candidate_logits = self.decode_grid_logits(
            grid_logits,
            top_k=self.num_grid_logits,
        )
        selected_points: list[torch.Tensor] = []
        selected_logits: list[torch.Tensor] = []
        selected_scores: list[torch.Tensor] = []
        selected_indices: list[torch.Tensor] = []
        selected_ks: list[int] = []
        candidate_scores: list[torch.Tensor] = []
        for points, logits in zip(candidate_points, candidate_logits):
            selected = self.select_dynamic_topk(
                points,
                logits,
                abs_threshold=abs_threshold,
                rel_ratio=rel_ratio,
                min_k=min_k,
                max_k=max_k,
            )
            selected_points.append(selected["selected_points"])
            selected_logits.append(selected["selected_logits"])
            selected_scores.append(selected["selected_scores"])
            selected_indices.append(selected["selected_indices"])
            selected_ks.append(int(selected["selected_k"]))
            candidate_scores.append(selected["candidate_scores"])
        return {
            "selected_points": selected_points,
            "selected_logits": selected_logits,
            "selected_scores": selected_scores,
            "selected_indices": selected_indices,
            "selected_ks": selected_ks,
            "candidate_points": candidate_points,
            "candidate_logits": candidate_logits,
            "candidate_scores": candidate_scores,
        }


class CoordinateAdapter(BaseCoordinateAdapter):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(ffn_hidden_multiplier=4, **kwargs)


class LightweightCoordinateAdapter(BaseCoordinateAdapter):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(ffn_hidden_multiplier=2, **kwargs)

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_attention import GatedFusion, ResidualFFN, TextGuidedCrossAttention
from .grid_encoder import FeatureProjector, GridEncoder


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score_proj = nn.Linear(dim, 1)

    def forward(self, features, attention_mask=None):
        scores = self.score_proj(features).squeeze(-1)
        if attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(features * weights.unsqueeze(-1), dim=1)


class BaseCoordinateAdapter(nn.Module):
    def __init__(
        self,
        visual_dim=768,
        grid_feature_dim=512,
        hidden_dim=512,
        num_heads=8,
        num_grid_tokens=64,
        num_output_points=4,
        dropout=0.1,
        output_mode="grid_logits",
        grid_size=11,
        ffn_hidden_multiplier=4,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.num_output_points = num_output_points
        self.output_mode = output_mode
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
        self.point_head = nn.Sequential(
            nn.Linear(visual_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_output_points * 3),
        )
        self._initialize_weights()

    @staticmethod
    def _build_2d_sinusoidal_embedding(grid_size, dim):
        import math

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

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

    def forward(self, images, grid_images, visual_features, text_features=None):
        grid_features_map = self.grid_encoder(grid_images)
        grid_tokens = self.grid_projector(grid_features_map)
        enhanced_features = self.cross_attention(
            visual_features=visual_features,
            grid_features=grid_tokens,
            text_features=text_features,
        )
        fused_features = self.gated_fusion(visual_features, enhanced_features)
        return self.residual_ffn(fused_features)

    def _pool_text_features(self, text_features=None, attention_mask=None, visual_summary=None):
        if text_features is None:
            if visual_summary is None:
                raise ValueError("visual_summary is required when text_features is None")
            return torch.zeros_like(visual_summary)
        return self.text_pool(text_features, attention_mask=attention_mask)

    def _build_text_conditioning(self, visual_features, text_features=None, attention_mask=None):
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

    def predict_grid_logits(self, visual_features, text_features=None, attention_mask=None):
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

    def decode_grid_logits(self, grid_logits, top_k=None):
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
    def select_dynamic_topk(pred_points, pred_logits, abs_threshold=0.35, rel_ratio=0.75, min_k=1, max_k=6):
        if pred_logits.ndim != 1:
            raise ValueError("pred_logits must be a 1D tensor")
        if pred_points.ndim != 2:
            raise ValueError("pred_points must be a 2D tensor")
        if pred_points.shape[0] != pred_logits.shape[0]:
            raise ValueError("pred_points and pred_logits must contain the same number of candidates")
        num_candidates = pred_logits.shape[0]
        if num_candidates == 0:
            empty_points = pred_points.new_zeros((0, 2))
            empty_scores = pred_logits.new_zeros((0,))
            empty_indices = torch.zeros((0,), dtype=torch.long, device=pred_logits.device)
            return {
                "selected_points": empty_points,
                "selected_logits": empty_scores,
                "selected_scores": empty_scores,
                "selected_indices": empty_indices,
                "selected_k": 0,
                "candidate_scores": empty_scores,
            }
        max_k = max(1, min(int(max_k), num_candidates))
        min_k = max(1, min(int(min_k), max_k))
        sorted_logits, sorted_indices = torch.sort(pred_logits, descending=True)
        sorted_points = pred_points[sorted_indices]
        sorted_scores = torch.sigmoid(sorted_logits)
        top1_score = sorted_scores[0]
        threshold = torch.maximum(
            sorted_scores.new_tensor(float(abs_threshold)),
            top1_score * float(rel_ratio),
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

    def decode_grid_logits_dynamic(self, grid_logits, abs_threshold=0.35, rel_ratio=0.75, min_k=1, max_k=6):
        candidate_points, candidate_logits = self.decode_grid_logits(grid_logits, top_k=self.num_grid_logits)
        selected_points = []
        selected_logits = []
        selected_scores = []
        selected_indices = []
        selected_ks = []
        candidate_scores = []
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
            selected_ks.append(selected["selected_k"])
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
    def __init__(self, **kwargs):
        super().__init__(ffn_hidden_multiplier=4, **kwargs)


class LightweightCoordinateAdapter(BaseCoordinateAdapter):
    def __init__(self, **kwargs):
        super().__init__(ffn_hidden_multiplier=2, **kwargs)

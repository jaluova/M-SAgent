import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
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

    def forward(self, visual_features, grid_features, attention_mask=None):
        batch_size, num_visual_tokens, dim = visual_features.shape
        _, num_grid_tokens, _ = grid_features.shape
        residual = visual_features
        query = self.q_proj(visual_features)
        key = self.k_proj(grid_features)
        value = self.v_proj(grid_features)

        query = query.view(batch_size, num_visual_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, num_grid_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, num_grid_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        context = torch.matmul(attention, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_visual_tokens, dim)
        output = self.out_proj(context)
        return self.norm(residual + output)


class TextGuidedCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.grid_cross_attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.text_cross_attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.merge_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, visual_features, grid_features, text_features=None, text_attention_mask=None):
        grid_enhanced = self.grid_cross_attn(visual_features, grid_features)
        if text_features is None:
            return grid_enhanced
        text_enhanced = self.text_cross_attn(visual_features, text_features)
        gate = self.merge_gate(torch.cat([grid_enhanced, text_enhanced], dim=-1))
        return gate * text_enhanced + (1 - gate) * grid_enhanced


class GatedFusion(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.gate_proj = nn.Linear(dim * 2, dim)
        self.gate_activation = nn.Sigmoid()
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, original_features, enhanced_features):
        concat_features = torch.cat([original_features, enhanced_features], dim=-1)
        gate = self.gate_activation(self.gate_proj(concat_features))
        fused = gate * enhanced_features + (1 - gate) * original_features
        fused = self.out_proj(fused)
        fused = self.dropout(fused)
        return self.norm(original_features + fused)


class ResidualFFN(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
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

    def forward(self, inputs):
        return inputs + self.ffn(inputs)

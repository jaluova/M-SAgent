"""最小前端可视化页面加载器。"""

from __future__ import annotations

from pathlib import Path


def load_visualizer_html() -> str:
    """读取内置可视化页面。"""
    html_path = Path(__file__).with_name("web_visualizer.html")
    return html_path.read_text(encoding="utf-8")


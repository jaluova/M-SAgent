import numpy as np
import json
from PIL import ImageDraw
from config import Config
from datetime import datetime

from tools.train_adapter_client import TrainAdapterClient
from utils.localization import LocalizationResult

class ObjectLocator:
    """目标定位工具"""
    
    def __init__(self, mllm_processor, train_adapter_client=None, config=Config):
        self.mllm = mllm_processor
        self.config = config
        self.grid_rows = config.GRID_ROWS
        self.grid_cols = config.GRID_COLS
        self.train_adapter_client = train_adapter_client or TrainAdapterClient(config=config)
        
    def locate_referent(self, tool_params, image, sam_processor, query=None):
        """优先通过 TrainAdapter 服务定位，再回退到 MLLM 生成的网格点。"""
        print("定位目标...")
        localization, labels, fallback_reason = self._resolve_point_prompts(
            tool_params=tool_params,
            image=image,
            query=query,
        )

        if localization is None or not localization.absolute_points:
            print("未得到有效定位结果")
            return {"success": False, "message": fallback_reason or "No valid localization points"}

        points = localization.absolute_points
        print(f"使用定位来源: {localization.source}, 点数: {len(points)}, top score: {localization.top_score():.3f}")

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            points_img = self.visualize_points(
                image,
                points,
                labels
            )
            points_path = self.config.TOOL_CALLS_LOG / f"object_locator_points_{timestamp}.jpg"
            points_img.save(points_path)
            print(f"点可视化已保存: {points_path}")

            if localization.annotated_image is not None:
                overlay_path = self.config.TOOL_CALLS_LOG / f"train_adapter_overlay_{timestamp}.jpg"
                localization.annotated_image.save(overlay_path)
                print(f"TrainAdapter overlay 已保存: {overlay_path}")

            meta_path = self.config.TOOL_CALLS_LOG / f"object_locator_meta_{timestamp}.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(localization.as_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"点可视化失败: {e}")
            
        # 尝试调用 sam 进行分割
        if sam_processor:
            print("调用 sam 进行分割...")
            result = sam_processor.segment_with_points(
                image,
                points=points,
                labels=labels,
                multimask_output=True
            )
            
            # 保存分割结果可视化
            if result.get("success") and result.get("best_result"):
                try:
                    best_mask = result["best_result"]["mask"]
                    # 应用MASK
                    vis_image = sam_processor.apply_mask_to_image(image, best_mask)
                    # 叠加点
                    vis_image = self.visualize_points(vis_image, points, labels)
                    
                    vis_path = self.config.TOOL_CALLS_LOG / f"object_locator_result_{timestamp}.jpg"
                    vis_image.save(vis_path)
                    print(f"定位分割结果已保存: {vis_path}")
                except Exception as e:
                    print(f"保存分割结果可视化失败: {e}")

            result["localization"] = localization.as_dict()
            if fallback_reason:
                result["fallback_reason"] = fallback_reason
            
            return result

        # 返回点信息
        return {
            "success": True,
            "points": points,
            "labels": labels,
            "localization": localization.as_dict(),
            "fallback_reason": fallback_reason,
        }

    def locate_object_with_points(self, tool_params, image, sam_processor, query=None):
        """兼容旧调用方式。"""
        return self.locate_referent(tool_params, image, sam_processor, query=query)

    def _resolve_point_prompts(self, tool_params, image, query=None):
        fallback_reason = None
        localization = None
        labels = None

        if self.config.TRAIN_ADAPTER_ENABLED and query:
            try:
                localization = self.train_adapter_client.localize(image, query)
            except Exception as exc:
                fallback_reason = f"TrainAdapter unavailable: {exc}"
                print(fallback_reason)
            else:
                if localization.has_usable_points(self.config.TRAIN_ADAPTER_MIN_SCORE_FOR_USE):
                    labels = [1] * len(localization.absolute_points)
                    return localization, labels, None
                fallback_reason = (
                    "TrainAdapter returned low-confidence localization "
                    f"(top_score={localization.top_score():.3f}, selected_k={localization.selected_k})"
                )
                print(fallback_reason)

        points, labels = self._grid_points_to_pixel_points(
            tool_params.get("points"),
            tool_params.get("labels"),
            image.size,
        )
        if not points:
            return None, None, fallback_reason or "No valid TrainAdapter result or fallback hints"

        localization = self._build_hint_localization(points, labels, image.size)
        return localization, labels, fallback_reason

    def _build_hint_localization(self, points, labels, image_size):
        width, height = image_size
        normalized_points = []
        for x, y in points:
            normalized_points.append([
                float(x) / width if width else 0.0,
                float(y) / height if height else 0.0,
            ])
        scores = [1.0 if int(label) == 1 else 0.0 for label in labels]
        return LocalizationResult(
            absolute_points=[[float(x), float(y)] for x, y in points],
            normalized_points=normalized_points,
            scores=scores,
            selection_mode="mllm_grid_hints",
            selected_k=len(points),
            source="mllm_hints",
        )

    def _grid_points_to_pixel_points(self, grid_points, labels, image_size):
        if not grid_points or not labels or len(grid_points) != len(labels):
            print("未提供可用的回退点提示")
            return [], []

        print(f"收到回退网格点参数: {grid_points}, 标签: {labels}")

        width, height = image_size
        row_positions = np.linspace(0, height, self.grid_rows + 1, dtype=int)
        col_positions = np.linspace(0, width, self.grid_cols + 1, dtype=int)

        points = []
        valid_labels = []

        for i, p in enumerate(grid_points):
            try:
                col_idx = int(p[0])
                row_idx = int(p[1])
                if row_idx < 0 or row_idx > self.grid_rows or col_idx < 0 or col_idx > self.grid_cols:
                    print(f"警告: 点 {p} (Col: {col_idx}, Row: {row_idx}) 超出网格范围，已忽略")
                    continue

                y = row_positions[row_idx]
                x = col_positions[col_idx]

                points.append([float(x), float(y)])
                valid_labels.append(int(labels[i]))
                print(f"解析点 {i}: Grid(Col={col_idx}, Row={row_idx}) -> Pixel(x={x:.1f}, y={y:.1f})")
            except Exception as e:
                print(f"坐标转换失败 {p}: {e}")

        if points:
            print(f"转换后的像素点({len(points)}个): {points}")
        return points, valid_labels

    def visualize_points(self, image, points, labels):
        """
        可视化点和标签
        
        Args:
            image: PIL Image
            points: 点列表
            labels: 标签列表
            
        Returns:
            PIL Image: 带点的图像
        """
        image_with_points = image.copy()
        draw = ImageDraw.Draw(image_with_points)
        
        for point, label in zip(points, labels):
            x, y = point
            
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            radius = 8
            
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=color,
                outline=(0, 0, 0)
            )
            
            label_text = "F" if label == 1 else "B"
            draw.text((x - 3, y - 10), label_text, fill=(255, 255, 255))
        
        return image_with_points

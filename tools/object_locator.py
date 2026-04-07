import numpy as np
import json
import time
import cv2
from PIL import ImageDraw
from config import Config
from datetime import datetime

from gridground_runtime.embedded_predictor import EmbeddedGridGroundPredictor
from tools.train_adapter_client import TrainAdapterClient
from utils.localization import LocalizationResult

class ObjectLocator:
    """目标定位工具"""
    
    def __init__(self, mllm_processor, train_adapter_client=None, embedded_localizer=None, config=Config):
        self.mllm = mllm_processor
        self.config = config
        self.grid_rows = config.GRID_ROWS
        self.grid_cols = config.GRID_COLS
        self.localization_backend = (config.GRIDGROUND_BACKEND or "embedded").lower()
        self.embedded_localizer_init_error = None
        self.embedded_localizer = embedded_localizer
        if self.localization_backend == "embedded" and self.embedded_localizer is None and mllm_processor is not None:
            try:
                self.embedded_localizer = EmbeddedGridGroundPredictor(
                    mllm_processor.shared_backbone,
                    config=config,
                )
            except Exception as exc:
                self.embedded_localizer_init_error = str(exc)
                print(f"Embedded GridGround init failed: {exc}")

        if self.localization_backend == "embedded" and self.embedded_localizer is None and train_adapter_client is not None:
            self.localization_backend = "http"

        self.train_adapter_client = train_adapter_client or TrainAdapterClient(config=config)

    def check_service_health(self):
        if self.localization_backend == "embedded":
            if self.embedded_localizer is None:
                return {
                    "ok": False,
                    "backend": "embedded",
                    "error": self.embedded_localizer_init_error or "embedded runtime unavailable",
                    "error_kind": "init",
                }
            return self.embedded_localizer.describe_runtime()

        if not self.config.TRAIN_ADAPTER_ENABLED:
            return {
                "ok": False,
                "skipped": True,
                "reason": "TRAIN_ADAPTER_ENABLED is false",
            }

        status = self.train_adapter_client.describe_service()
        expected_device = (self.config.TRAIN_ADAPTER_EXPECTED_DEVICE or "").lower()
        if status.get("ok") and expected_device:
            actual_device = str(status.get("device", "")).lower()
            status["expected_device"] = expected_device
            status["device_matches_expected"] = actual_device == expected_device
        return status
        
    def locate_referent(self, tool_params, image, sam_processor, query=None):
        """优先通过 TrainAdapter 服务定位，再回退到 MLLM 生成的网格点。"""
        print("定位目标...")
        localization, labels, fallback_reason, service_meta = self._resolve_point_prompts(
            tool_params=tool_params,
            image=image,
            query=query,
        )

        if localization is None or not localization.absolute_points:
            print("未得到有效定位结果")
            return {"success": False, "message": fallback_reason or "No valid localization points"}

        points = localization.absolute_points
        if service_meta:
            service_device = service_meta.get("service_device", "unknown")
            slow_path = " (slow path)" if service_meta.get("slow_path") else ""
            latest_attempt = (service_meta.get("attempts") or [{}])[-1]
            elapsed_ms = latest_attempt.get("elapsed_ms")
            elapsed_suffix = f", latency: {elapsed_ms:.2f} ms" if isinstance(elapsed_ms, (float, int)) else ""
            print(f"TrainAdapter service device: {service_device}{slow_path}{elapsed_suffix}")

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
                json.dump(
                    {
                        "localization": localization.as_dict(),
                        "train_adapter_service": service_meta,
                        "fallback_reason": fallback_reason,
                        "runtime_profile": self.config.runtime_profile_summary(),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
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

            # 连通域过滤：只保留包含最高置信度点的连通区域
            if result.get("success") and result.get("best_result"):
                top_point = localization.absolute_points[0]  # 按置信度排序，第一个最高
                result_items = result.get("results") or [result["best_result"]]

                for res in result_items:
                    res["mask"] = self._keep_component_at_point(
                        res["mask"], top_point,
                    )

                # 重新计算 mask_area 和 best_result
                for res in result_items:
                    res["mask_area"] = int(np.sum(res["mask"] > 0.5))

                result["results"] = result_items
                result["best_result"] = max(
                    result_items, key=lambda x: x["score"],
                )

            # 保存分割结果可视化
            if result.get("success") and result.get("best_result"):
                try:
                    best_mask = result["best_result"]["mask"]
                    vis_image = sam_processor.apply_mask_to_image(image, best_mask)
                    vis_image = self.visualize_points(vis_image, points, labels)

                    vis_path = self.config.TOOL_CALLS_LOG / f"object_locator_result_{timestamp}.jpg"
                    vis_image.save(vis_path)
                    print(f"定位分割结果已保存: {vis_path}")
                except Exception as e:
                    print(f"保存分割结果可视化失败: {e}")

            result["localization"] = localization.as_dict()
            result["train_adapter_service"] = service_meta
            if fallback_reason:
                result["fallback_reason"] = fallback_reason
            
            return result

        # 返回点信息
        return {
            "success": True,
            "points": points,
            "labels": labels,
            "localization": localization.as_dict(),
            "train_adapter_service": service_meta,
            "fallback_reason": fallback_reason,
        }

    def locate_object_with_points(self, tool_params, image, sam_processor, query=None):
        """兼容旧调用方式。"""
        return self.locate_referent(tool_params, image, sam_processor, query=query)

    def _resolve_point_prompts(self, tool_params, image, query=None):
        fallback_reason = None
        localization = None
        labels = None
        service_meta = None

        if self.localization_backend == "embedded" and query:
            if self.embedded_localizer is None:
                fallback_reason = (
                    "Embedded GridGround unavailable: "
                    f"{self.embedded_localizer_init_error or 'runtime not initialized'}"
                )
                print(fallback_reason)
            else:
                started = time.perf_counter()
                try:
                    localization, service_meta = self.embedded_localizer.localize_with_metadata(image, query)
                except Exception as exc:
                    fallback_reason = f"Embedded GridGround unavailable: {exc}"
                    print(fallback_reason)
                else:
                    if service_meta and service_meta.get("attempts"):
                        service_meta["attempts"][0]["elapsed_ms"] = round(
                            (time.perf_counter() - started) * 1000.0,
                            2,
                        )
                    if localization.has_usable_points(self.config.TRAIN_ADAPTER_MIN_SCORE_FOR_USE):
                        labels = [1] * len(localization.absolute_points)
                        return localization, labels, None, service_meta
                    fallback_reason = (
                        "Embedded GridGround returned low-confidence localization "
                        f"(top_score={localization.top_score():.3f}, selected_k={localization.selected_k})"
                    )
                    print(fallback_reason)

        if self.config.TRAIN_ADAPTER_ENABLED and query and self.localization_backend == "http":
            try:
                if hasattr(self.train_adapter_client, "localize_with_metadata"):
                    localization, service_meta = self.train_adapter_client.localize_with_metadata(image, query)
                else:
                    localization = self.train_adapter_client.localize(image, query)
                    service_meta = {}
            except Exception as exc:
                fallback_reason = f"TrainAdapter unavailable: {exc}"
                print(fallback_reason)
            else:
                if service_meta and service_meta.get("slow_path"):
                    print("TrainAdapter is serving requests on CPU slow path")
                if localization.has_usable_points(self.config.TRAIN_ADAPTER_MIN_SCORE_FOR_USE):
                    labels = [1] * len(localization.absolute_points)
                    return localization, labels, None, service_meta
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
            return None, None, fallback_reason or "No valid TrainAdapter result or fallback hints", service_meta

        localization = self._build_hint_localization(points, labels, image.size)
        return localization, labels, fallback_reason, service_meta

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

    @staticmethod
    def _keep_component_at_point(mask, point):
        """保留包含指定点的连通区域，去掉被杂点拉进来的分离区域。

        Args:
            mask: numpy array (H, W), 分割结果
            point: [x, y] 最高置信度点的绝对坐标

        Returns:
            numpy array (H, W), 过滤后的 mask
        """
        mask_array = np.asarray(np.squeeze(mask))
        mask_binary = (mask_array > 0.5).astype(np.uint8)
        num_labels, labels_map = cv2.connectedComponents(mask_binary)
        if num_labels <= 2:
            # 只有背景 + 1 个连通域，无需过滤
            return mask_binary.astype(mask_array.dtype, copy=False)

        px, py = int(round(point[0])), int(round(point[1]))
        h, w = labels_map.shape
        py = max(0, min(py, h - 1))
        px = max(0, min(px, w - 1))

        target_label = labels_map[py, px]
        if target_label == 0:
            # 点恰好在背景上（mask 边缘），找最近的前景连通域
            fg_ys, fg_xs = np.where(mask_binary > 0)
            if len(fg_ys) == 0:
                return mask_binary.astype(mask_array.dtype, copy=False)
            dists = (fg_xs - px) ** 2 + (fg_ys - py) ** 2
            nearest_idx = np.argmin(dists)
            target_label = labels_map[fg_ys[nearest_idx], fg_xs[nearest_idx]]

        filtered = (labels_map == target_label).astype(mask_array.dtype, copy=False)
        removed_pixels = int(np.sum(mask_binary) - np.sum(filtered))
        if removed_pixels > 0:
            print(f"连通域过滤: 移除了 {removed_pixels} 个离散像素 (共 {num_labels - 1} 个连通域)")
        return filtered

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

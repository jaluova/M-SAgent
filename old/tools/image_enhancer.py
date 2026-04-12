import numpy as np
from PIL import Image
from config import Config
from utils.image_utils import resize_image
import os
from datetime import datetime

class ImageEnhancer:
    """图像增强工具"""

    EDGE_CONTACT_BORDER_PX = 3
    EDGE_CONTACT_MIN_FRACTION = 0.05
    
    def __init__(self):
        self.grid_rows = Config.GRID_ROWS
        self.grid_cols = Config.GRID_COLS
        
    def enhance_image(self, image, text_prompt, sam_processor, tool_params):
        """
        增强图像处理流程
        
        Args:
           image: PIL Image (原始图像)
           text_prompt: str (文本提示)
           sam_processor: SAMProcessor 实例
           tool_params: dict (工具参数)
           
        Returns:
            dict: 分割结果
        """
        print("执行图像增强处理...")
        
        # 1. 解析参数并获取裁剪区域
        crop_box = self._get_crop_box(image, tool_params)
        if crop_box is None:
            print("无法确定裁剪区域")
            return None
            
        x1, y1, x2, y2 = crop_box
        print(f"裁剪区域: {crop_box}, 大小: {x2-x1}x{y2-y1}")
        
        # 2. 裁剪图像
        cropped_image = image.crop((x1, y1, x2, y2))
        
        # 3. 放大图像
        # 将裁剪部分放大到较大尺寸以便于SAM识别细节
        enhanced_image = resize_image(cropped_image, max_size=1024)
        
        # 保存中间结果方便调试
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            save_path = Config.TOOL_CALLS_LOG / f"image_enhancer_crop_{timestamp}.jpg"
            enhanced_image.save(save_path)
            print(f"增强后的裁剪图像已保存: {save_path}")
        except Exception as e:
            print(f"保存中间图像失败: {e}")
            
        # 4. SAM3 分割
        print(f"使用增强图像进行SAM3分割: {text_prompt}")
        seg_result = sam_processor.segment_with_text(enhanced_image, text_prompt)
        
        if not seg_result or not seg_result.get("success"):
            print("SAM3增强图像分割失败")
            return None
        
        best_result = seg_result.get("best_result")
        if not best_result:
             return None
             
        mask_relative = best_result["mask"] # mask relative to enhanced_image

        truncated, edge_fractions = self._mask_looks_truncated(
            mask_relative,
            crop_box,
            image.size,
        )
        if truncated:
            print(
                "增强分割结果疑似被裁剪框截断，拒绝当前 image_enhancer 结果: "
                f"{edge_fractions}"
            )
            return {
                "success": False,
                "message": "Enhanced crop likely truncates the target object",
                "results": [],
            }
        
        # 5. 还原 Mask 到原图尺寸坐标
        full_mask = self._restore_mask(mask_relative, crop_box, image.size)
        full_mask_area = np.sum(full_mask > 0.5)

        # 保存最终结果可视化
        try:
            vis_image = sam_processor.apply_mask_to_image(image, full_mask)
            vis_path = Config.TOOL_CALLS_LOG / f"image_enhancer_result_{timestamp}.jpg"
            vis_image.save(vis_path)
            print(f"增强分割结果已保存: {vis_path}")
        except Exception as e:
            print(f"保存增强分割结果可视化失败: {e}")

        # 构造符合 Pipeline 期望的返回结果
        final_result = {
            "success": True,
            "results": [],
            "best_result": {
                "mask": full_mask,
                "score": best_result["score"],
                "method": "image_enhancer",
                "text_prompt": text_prompt,
                "mask_area": full_mask_area
            }
        }
        final_result["results"].append(final_result["best_result"])
        
        return final_result

    def _get_crop_box(self, image, params):
        """根据网格参数计算裁剪区域"""
        width, height = image.size
        
        row_idx_start = 0
        row_idx_end = 0
        col_idx_start = 0
        col_idx_end = 0

        # 网格通常显示为 1-based (1..N)，这里假设输入也是 1-based，转换为 0-based处理
        
        if "rectangular area" in params:
             # [[c1, r1], [c2, r2]]  ([x1, y1], [x2, y2])
             # 坐标是网格线的交点
             # Col (X轴): 0在左侧 (顶部数字)
             # Row (Y轴): 0在顶部 (左侧数字)
             try:
                 p1, p2 = params["rectangular area"]
                 
                 # 解析 [col(x), row(y)]
                 c1, r1 = p1
                 c2, r2 = p2
                 
                 print(f"收到裁剪区域坐标: p1=[x={c1}, y={r1}], p2=[x={c2}, y={r2}]")
                 
                 # 计算像素坐标 X
                 # x = (c / cols) * width
                 x1 = int((c1 / self.grid_cols) * width)
                 x2 = int((c2 / self.grid_cols) * width)
                 
                 # 计算像素坐标 Y
                 # y = (r / rows) * height
                 y1 = int((r1 / self.grid_rows) * height)
                 y2 = int((r2 / self.grid_rows) * height)
                 
                 # 整理坐标 (min, max)
                 xmin = min(x1, x2)
                 xmax = max(x1, x2)
                 ymin = min(y1, y2)
                 ymax = max(y1, y2)
                 
                 print(f"转换像素坐标: x=[{xmin}, {xmax}], y=[{ymin}, {ymax}]")
                 
                 # 边界检查
                 xmin = max(0, xmin)
                 ymin = max(0, ymin)
                 xmax = min(width, xmax)
                 ymax = min(height, ymax)
                 
                 if xmax <= xmin or ymax <= ymin:
                     print(f"无效的裁剪区域: {xmin}, {ymin}, {xmax}, {ymax}")
                     return None
                     
                 return (xmin, ymin, xmax, ymax)
                 
             except (ValueError, TypeError) as e:
                 print(f"解析参数失败: {e}")
                 return None

        return None

    def _mask_looks_truncated(self, mask, crop_box, full_size):
        """检测 mask 是否明显贴着非图像边界的 crop 边缘。"""
        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array.squeeze()
        mask_bool = mask_array > 0.5
        if mask_bool.ndim != 2 or not np.any(mask_bool):
            return False, {}

        border = min(
            self.EDGE_CONTACT_BORDER_PX,
            max(1, mask_bool.shape[0] // 4),
            max(1, mask_bool.shape[1] // 4),
        )

        edge_fractions = {
            "left": float(np.any(mask_bool[:, :border], axis=1).mean()),
            "right": float(np.any(mask_bool[:, -border:], axis=1).mean()),
            "top": float(np.any(mask_bool[:border, :], axis=0).mean()),
            "bottom": float(np.any(mask_bool[-border:, :], axis=0).mean()),
        }

        x1, y1, x2, y2 = crop_box
        full_width, full_height = full_size
        edge_is_internal = {
            "left": x1 > 0,
            "right": x2 < full_width,
            "top": y1 > 0,
            "bottom": y2 < full_height,
        }

        truncated_edges = [
            edge
            for edge, fraction in edge_fractions.items()
            if edge_is_internal[edge] and fraction >= self.EDGE_CONTACT_MIN_FRACTION
        ]
        return bool(truncated_edges), edge_fractions

    def _restore_mask(self, mask, crop_box, full_size):
        """将局部mask还原到全图尺寸"""
        x1, y1, x2, y2 = crop_box
        crop_width = x2 - x1
        crop_height = y2 - y1
        full_width, full_height = full_size
        
        # mask 可能是 (H, W) 或 (1, H, W)
        if hasattr(mask, 'shape') and len(mask.shape) == 3:
            mask = mask.squeeze()
            
        # 1. 将 mask 调整回裁剪区域的大小
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8)
        
        # 使用 Nearest 保持二值性质
        mask_resized_pil = mask_pil.resize((crop_width, crop_height), Image.Resampling.NEAREST)
        mask_resized_np = np.array(mask_resized_pil) > 128
        
        # 2. 放置到全图 Mask 中
        full_mask = np.zeros((full_height, full_width), dtype=bool)
        
        # 确保尺寸匹配
        target_h = y2 - y1
        target_w = x2 - x1
        h, w = mask_resized_np.shape[:2]
        
        if h != target_h or w != target_w:
             mask_resized_pil = mask_pil.resize((target_w, target_h), Image.Resampling.NEAREST)
             mask_resized_np = np.array(mask_resized_pil) > 128
        
        full_mask[y1:y2, x1:x2] = mask_resized_np
        
        return full_mask
    

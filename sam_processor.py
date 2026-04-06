import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
import warnings
warnings.filterwarnings("ignore")

import sys
import os

from config import Config

# 添加SAM3目录到Python路径
sam3_root = os.path.join(Config.SAM3_MODEL_PATH, "sam3")
if sam3_root not in sys.path:
    sys.path.insert(0, sam3_root)

try:
    # 现在可以直接导入
    from model_builder import build_sam3_image_model
    from model.sam3_image_processor import Sam3Processor
    from visualization_utils import plot_results
    SAM3_AVAILABLE = True
    print("✓ SAM3导入成功")
except ImportError as e:
    print(f"导入失败: {e}")
    SAM3_AVAILABLE = False

class SAMProcessor:
    def __init__(self):
        self.device = Config.DEVICE
        self.model = None
        self.processor = None
        
        if SAM3_AVAILABLE:
            self._load_model()
        else:
            print("警告: SAM3不可用")
        
    def _load_model(self):
        """加载SAM3模型"""
        print("加载SAM3模型...")
        try:
            # 尝试加载模型
            self.model = build_sam3_image_model(
                bpe_path=Config.SAM3_BPE_PATH,
                checkpoint_path=Config.SAM3_CHECKPOINT_PATH,
                enable_inst_interactivity=True,
            )
            print("✓ 模型构建成功")
            
            # 创建处理器
            self.processor = Sam3Processor(self.model)
            print("✓ SAM3处理器创建成功")

            # 测试模型是否可用
            print("测试SAM3模型...")
            test_image = Image.new('RGB', (640, 480), color='white')
            try:
                inference_state = self.processor.set_image(test_image)
                print("✓ 模型测试通过")
            except Exception as e:
                print(f"模型测试失败: {e}")
                self.model = None
                self.processor = None

            # 默认先放到CPU节省显存
            self.to_cpu()

        except Exception as e:
            print(f"SAM3加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.processor = None

    def to_cpu(self):
        """将模型移动到CPU"""
        if self.model is not None:
            self.model.cpu()
            if hasattr(self.model, "image_encoder"):
                 self.model.image_encoder.cpu() # 确保子模块也移动
            torch.cuda.empty_cache()
            print("SAM3已移至CPU")

    def to_gpu(self):
        """将模型移动到GPU"""
        if self.model is not None and Config.DEVICE == "cuda":
            self.model.cuda()
            print("SAM3已移至GPU")
    
    def is_available(self):
        """检查SAM3是否可用"""
        return self.model is not None and self.processor is not None
    
    def segment_with_text(self, image, text_prompt=None, multimask_output=True):
        """使用文本提示分割图像"""
        # 检查模型是否可用
        if not self.is_available():
            print("SAM3不可用，无法执行文本提示分割")
            return {
                "success": False,
                "message": "SAM3 unavailable",
                "results": []
            }

        try:
            # 移至GPU
            self.to_gpu()

            # 设置图像
            print("设置图像...")
            inference_state = self.processor.set_image(image)

            # 重置所有提示
            self.processor.reset_all_prompts(inference_state)

            # 设置文本提示
            print(f"使用文本提示: {text_prompt}")
            inference_state = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)

            # 可视化结果
            # plot_results(image, inference_state)

            # 提取分割结果
            masks = inference_state["masks"]
            scores = inference_state["scores"]
            
            if multimask_output and len(masks) > 0:
                try:
                    masks_data = [m.squeeze().cpu().numpy() for m in masks]
                    scores_data = [float(s) for s in scores]
                    self.save_multimask_visualization(image, masks_data, scores_data, "text_prompt")
                except Exception as e:
                    print(f"保存可视化结果失败: {e}")

            results = []
            for i, (mask, score) in enumerate(zip(masks, scores)):
                mask_np = mask.squeeze().cpu().numpy()
                mask_area = np.sum(mask_np > 0.5)

                results.append({
                    "mask": mask_np,
                    "score": float(score),
                    "method": "text_prompt",
                    "text_prompt": text_prompt,
                    "mask_area": mask_area
                })

            # 按分数排序
            results.sort(key=lambda x: x["score"], reverse=True)

            if not results:
                print("未找到任何分割结果")
                return {
                    "success": False,
                    "message": "No masks found",
                    "results": []
                }

            print(f"最佳分割分数: {results[0]['score']:.3f}")

            return {
                "success": True,
                "results": results,
                "best_result": results[0]
            }

        except Exception as e:
            print(f"文本提示分割失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "message": f"分割失败: {str(e)}",
                "results": []
            }
        finally:
            self.to_cpu()
    
    def segment_with_points(self, image, points, labels, multimask_output=True):
        """使用点提示分割图像"""
        if not self.is_available():
            print("SAM3不可用，无法执行点提示分割")
            return {
                "success": False,
                "message": "SAM3 unavailable",
                "results": [],
                "best_result": None,
            }

        results = []
        try:
            self.to_gpu() # 移至GPU
            
            # 设置图像
            print("设置图像...")
            inference_state = self.processor.set_image(image)
            
            # 转换为numpy数组
            points_np = np.array(points, dtype=np.float32)
            labels_np = np.array(labels, dtype=np.int32)

            print(f"使用点提示: {len(points)}个点")

            # 如果有之前的mask，作为输入
            mask_input = None
            if results:
                best_idx = np.argmax([r["score"] for r in results])
                best_mask = results[best_idx]["mask"]
                mask_input = torch.from_numpy(best_mask).float().unsqueeze(0).unsqueeze(0)
                if self.device == "cuda":
                    mask_input = mask_input.cuda()

            # 预测实例 - 确保inference_state正确传递
            print("调用predict_inst...")
            masks, scores, _ = self.model.predict_inst(
                inference_state,
                point_coords=points_np,
                point_labels=labels_np,
                mask_input=mask_input,
                multimask_output=multimask_output,
            )

            print(f"点提示找到 {len(scores)} 个分割结果")

            # 处理结果
            for i, (score, mask) in enumerate(zip(scores, masks)):
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.squeeze().cpu().numpy()
                else:
                    mask_np = mask.squeeze()
                mask_area = np.sum(mask_np > 0.5)

                results.append({
                    "mask": mask_np,
                    "score": float(score),
                    "points": points,
                    "labels": labels,
                    "method": "points",
                    "mask_area": mask_area
                })

            # 可视化结果（可选）
            if len(results) > 0:
                masks_data = [r["mask"] for r in results]
                scores_data = [r["score"] for r in results]
                try:
                    self.save_multimask_visualization(
                        image, 
                        masks_data, 
                        scores_data, 
                        "points_prompt", 
                        points_np, 
                        labels_np
                    )
                except Exception as e:
                    print(f"保存可视化结果失败: {e}")

        except Exception as e:
            print(f"点提示分割失败: {e}")
            return {
                "success": False,
                "message": f"分割失败: {str(e)}",
                "results": [],
                "best_result": None,
            }
        finally:
            self.to_cpu() # 移至CPU

        return {
            "success": len(results) > 0,
            "results": results,
            "best_result": max(results, key=lambda x: x["score"]) if results else None
        }

    def save_multimask_visualization(self, image, masks, scores, method_name, point_coords=None, input_labels=None):
        """保存多mask分割的可视化结果"""
        from datetime import datetime
        import matplotlib.pyplot as plt
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Config.TOOL_CALLS_LOG / f"{method_name}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"保存中间可视化结果到: {save_dir}")
        
        # 1. 保存所有mask叠加的结果
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        
        # 排个序，分低的在下，高的在上? 或者反过来
        # 这里简单遍历
        for i, (mask, score) in enumerate(zip(masks, scores)):
            self.show_mask(mask, plt.gca(), random_color=True)
            
        if point_coords is not None and input_labels is not None:
             self.show_points(point_coords, input_labels, plt.gca())
             
        plt.axis('off')
        plt.title(f"All Masks Overlay ({len(masks)})")
        plt.savefig(save_dir / "all_masks_overlay.jpg", bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # 2. 保存单独的mask
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            self.show_mask(mask, plt.gca(), random_color=False)
            
            if point_coords is not None and input_labels is not None:
                 self.show_points(point_coords, input_labels, plt.gca())
                 
            plt.title(f"Mask {i} Score: {score:.3f}")
            plt.axis('off')
            plt.savefig(save_dir / f"mask_{i}_score_{score:.3f}.jpg", bbox_inches='tight', pad_inches=0)
            plt.close()
    
    def visualize_masks_with_numbers(self, image, masks_data, draw_numbers=True):
        """
        Visualize multiple masks with numbers and colors.
        
        Args:
            image: PIL Image
            masks_data: List of dicts, each containing:
                - mask: boolean numpy array or tensor
                - color: tuple (r, g, b) (0-255)
                - id: int (number to display)
            draw_numbers: bool, whether to draw ID numbers
                
        Returns:
            PIL Image: Image with masks and numbers
        """
        if not masks_data:
            return image.copy()
            
        # Ensure image is RGB and numpy array
        if isinstance(image, Image.Image):
            image = image.convert('RGB')
            # Work with copy to avoid modifying original
            img_np = np.array(image)
        else:
            img_np = image.copy()
            
        height, width = img_np.shape[:2]
        
        # Create an overlay layer for masks (using float for blending)
        overlay_np = img_np.astype(float)
        
        # 1. Draw Masks
        for data in masks_data:
            mask = data['mask']
            # Default color (Red) if not provided
            color = data.get('color', (255, 0, 0))
            
            # Remove alpha from color if present
            if len(color) == 4:
                rgb = color[:3]
            else:
                rgb = color
            
            alpha = 0.5
            
            # Handle Mask Format
            # mask should be (H, W) boolean or 0/1
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            
            mask = np.squeeze(mask)
            
            # Resize mask if needed
            if mask.shape[:2] != (height, width):
                mask_uint8 = (mask * 255).astype(np.uint8)
                mask_resized = cv2.resize(mask_uint8, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_bool = mask_resized > 128
            else:
                mask_bool = mask > 0.5
            
            if not mask_bool.any():
                continue
                
            # Apply color blending manually
            roi = overlay_np[mask_bool]
            # Broadcasting rgb to roi shape
            blended = roi * (1 - alpha) + np.array(rgb) * alpha
            overlay_np[mask_bool] = blended
            
            # Draw contour (Black outline for the mask itself)
            # Find contours on voltage mask
            contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_np, contours, -1, (0, 0, 0), 2) 

        # Convert back to uint8
        result_np = np.clip(overlay_np, 0, 255).astype(np.uint8)
        
        # 2. Draw Numbers
        if draw_numbers:
            for data in masks_data:
                mask = data['mask']
                label_id = str(data.get('id', ''))
                color = data.get('color', (255, 0, 0))
                if len(color) == 4: color = color[:3]
                    
                # Resize mask again (or better to store processed masks, but this is fine only a few masks)
                # ... (Same resize logic)
                mask = np.squeeze(mask)
                if mask.shape[:2] != (height, width):
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    mask_resized = cv2.resize(mask_uint8, (width, height), interpolation=cv2.INTER_NEAREST)
                    mask_bool = mask_resized > 128
                else:
                    mask_bool = mask > 0.5

                if not mask_bool.any():
                    continue
                
                # Find center
                mask_uint8 = mask_bool.astype(np.uint8)
                M = cv2.moments(mask_uint8)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    y_indices, x_indices = np.where(mask_bool)
                    if len(x_indices) > 0:
                        cX = int(np.mean(x_indices))
                        cY = int(np.mean(y_indices))
                    else:
                        continue
                
                # Calculate adaptive font scale based on mask size
                x, y, w, h = cv2.boundingRect(mask_bool.astype(np.uint8))
                min_dim = min(w, h)
                
                # Heuristic: roughly 1.0 scale for 40px dimension
                # Cap between 0.4 and 2.0
                font_scale = max(0.4, min(min_dim / 40.0, 2.0))
                thickness = max(1, int(font_scale * 2))
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # Calculate text size to center it
                (text_width, text_height), baseline = cv2.getTextSize(label_id, font, font_scale, thickness)
                
                # Adjust center position
                text_x = cX - text_width // 2
                text_y = cY + text_height // 2
                
                # Ensure text is within image bounds
                text_x = max(0, min(text_x, width - text_width))
                text_y = max(text_height, min(text_y, height))

                # Black outline 
                cv2.putText(result_np, label_id, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
                # Colored text (matching mask color)
                cv2.putText(result_np, label_id, (text_x, text_y), font, font_scale, tuple(int(c) for c in color), thickness)

        return Image.fromarray(result_np)

    def apply_mask_to_image(self, image, mask, color=(255, 0, 0, 128)):
        """将mask应用到图像上"""
        try:
            # 确保图像是RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 创建叠加图像
            overlay = image.convert('RGBA')
            
            # 将mask缩放到图像大小
            if mask.shape[:2] != image.size[::-1]:
                mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(
                    image.size, Image.Resampling.BILINEAR
                )
                mask_array = np.array(mask_resized) > 128
            else:
                mask_array = mask > 0.5
            
            # 创建颜色层
            color_array = np.zeros((image.size[1], image.size[0], 4), dtype=np.uint8)
            color_array[mask_array] = color
            
            # 合成图像
            color_image = Image.fromarray(color_array, 'RGBA')
            result = Image.alpha_composite(overlay, color_image)
            
            return result.convert('RGB')
            
        except Exception as e:
            print(f"应用mask失败: {e}")
            return image
    
    @staticmethod
    def show_mask(mask, ax, random_color=False, borders=True):
        """显示分割mask"""
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            import cv2
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # 尝试平滑轮廓
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(
            pos_points[:, 0],
            pos_points[:, 1],
            color="green",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )
        ax.scatter(
            neg_points[:, 0],
            neg_points[:, 1],
            color="red",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )

    def show_masks(self, image, masks, scores, point_coords=None, input_labels=None, box_coords=None, borders=True):
        """显示多个mask"""
        from matplotlib import pyplot as plt
        
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            self.show_mask(mask, plt.gca(), borders=borders)
            if point_coords is not None:
                assert input_labels is not None
                self.show_points(point_coords, input_labels, plt.gca())
            
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()

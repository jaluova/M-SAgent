import json
import time
from datetime import datetime
import numpy as np
from PIL import Image
from config import Config
from mllm_processor import MLLMProcessor
from sam_processor import SAMProcessor
from tools.object_locator import ObjectLocator
from tools.concept_generator import ConceptGenerator
from tools.image_enhancer import ImageEnhancer
from utils.image_utils import add_grid_to_image, resize_image, ensure_rgb

class MLLMSAMPipeline:
    """MLLM和SAM3的集成管道"""
    
    COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
        (0, 128, 0),    # Dark Green
        (128, 0, 0),    # Maroon
    ]

    def __init__(self):
        Config.setup_dirs()
        self._log_runtime_profile()

        # 初始化组件
        print("初始化组件...")
        self.mllm = MLLMProcessor()
        self.sam = SAMProcessor()
        self.locator = ObjectLocator(self.mllm)
        self._run_startup_healthchecks()
        self.concept_gen = ConceptGenerator(self.mllm)
        self.enhancer = ImageEnhancer()
        
        # 状态跟踪
        self.state = {
            "original_image": None, #原始输入图像
            "current_image": None,
            "original_text": "",
            "current_text": "",
            "final_image":None,
            "grid_info": None,
            "iteration": 0,
            "results": [],
            "accepted_masks": [], # List of {'mask': ..., 'score': ..., 'method': ...}
            "tool_history": [],   # List of {'tool': ..., 'verdict': ..., 'iteration': ...}
        }
        
        print("初始化完成")

    def _log_runtime_profile(self):
        print("运行配置摘要:")
        for key, value in Config.runtime_profile_summary().items():
            print(f"  - {key}: {value}")

    def _run_startup_healthchecks(self):
        if not Config.TRAIN_ADAPTER_STARTUP_HEALTHCHECK:
            print("Localization backend startup health check disabled")
            return

        status = self.locator.check_service_health()
        if status.get("skipped"):
            print(f"Localization backend startup check skipped: {status.get('reason')}")
            return

        if not status.get("ok"):
            print(
                "Localization backend startup check failed: "
                f"{status.get('error')} (kind={status.get('error_kind', 'unknown')})"
            )
            return

        backend = status.get("backend", Config.GRIDGROUND_BACKEND)
        device = status.get("device", "unknown")
        elapsed_ms = status.get("elapsed_ms")
        device_suffix = f", latency={elapsed_ms:.2f} ms" if isinstance(elapsed_ms, (float, int)) else ""
        print(f"Localization backend startup check passed: backend={backend}, device={device}{device_suffix}")

        if "device_matches_expected" in status and not status["device_matches_expected"]:
            print(
                "TrainAdapter device mismatch: "
                f"expected {status.get('expected_device')}, got {device}"
            )

    def get_color(self, index):
        return self.COLORS[index % len(self.COLORS)]

    def _normalize_segmentation_evaluation_result(self, evaluation_result):
        if isinstance(evaluation_result, (tuple, list)) and len(evaluation_result) >= 2:
            verdict, rejected_indices = evaluation_result[0], evaluation_result[1]
        elif isinstance(evaluation_result, str):
            verdict, rejected_indices = evaluation_result, []
        else:
            verdict, rejected_indices = "Reject", []

        verdict = str(verdict).capitalize()
        if verdict not in {"Accept", "Reject"}:
            verdict = "Reject"

        if isinstance(rejected_indices, (list, tuple, set)):
            rejected_indices = list(rejected_indices)
        else:
            rejected_indices = []

        return verdict, rejected_indices
    
    def run(self, image_path, text_prompt, max_iterations=None):
        """
        运行完整流程
        
        Args:
            image_path: 图像路径
            text_prompt: 文本提示
            max_iterations: 最大迭代次数
            
        Returns:
            dict: 最终结果
        """
        if max_iterations is None:
            max_iterations = Config.MAX_TOOL_CALLS
        
        print(f"\n开始处理流程")
        print(f"图像: {image_path}")
        print(f"目标: {text_prompt}")
        
        # 1. 加载并预处理图像
        self.load_image(image_path)
        self.state["original_text"] = text_prompt
        self.state["current_text"] = text_prompt
        
        # 2 迭代处理
        while self.state["iteration"] < max_iterations:
            self.state["iteration"] += 1
            print(f"\n--- 迭代 {self.state['iteration']}/{max_iterations} ---")
            
            # 2.1 MLLM分析决策
            print("MLLM分析中...")
            mllm_response, raw_response = self.mllm.process(
                self.state["original_image"],
                self.state["original_text"],
                self.state["grid_info"],
                image_path,
                self.state["iteration"],
                self.state["current_image"],
                tool_history=self.state["tool_history"],
            )
            
            print(f"MLLM决策: {mllm_response.get('name')}")
            
            # 2.2 执行相应操作
            action = mllm_response.get("name")
            tool_params = mllm_response.get("parameters", {})
            
            if action == "report_no_mask":
                print("经过仔细查看图片，未找到与查询相匹配的内容，结束流程")
                self.state["tool_history"].append({
                    "tool": "report_no_mask",
                    "verdict": "NoMask",
                    "iteration": self.state["iteration"],
                })
                break

            elif action == "concept_generator":
                # 生成概念，得到生成的每个concept对应的带有mask的分割结果图片
                result = self.concept_gen.segment_with_concept(
                    tool_params, self.sam, self.state["original_image"],
                    query=self.state["original_text"],
                )
                verdict = self._process_segmentation_result(result, "concept_generator")
                self.state["tool_history"].append({
                    "tool": "concept_generator",
                    "verdict": verdict,
                    "iteration": self.state["iteration"],
                })
                if verdict == "Accept":
                    print("MLLM接受当前结果，结束流程")
                    break

            elif action == "object_locator":
                # 定位目标，得到使用点提示的带有mask的分割结果图片
                result = self.locator.locate_referent(
                    tool_params,
                    self.state["original_image"],
                    self.sam,
                    query=self.state["original_text"],
                )
                verdict = self._process_segmentation_result(result, "object_locator")
                self.state["tool_history"].append({
                    "tool": "object_locator",
                    "verdict": verdict,
                    "iteration": self.state["iteration"],
                })
                if verdict == "Accept":
                    print("MLLM接受当前结果，结束流程")
                    break

            elif action == "image_enhancer":
                # 增强图像，放大图片和提示词交给分割模型生成mask，这个mask按照相应缩放比例再还原加到当前图像上
                result = self.enhancer.enhance_image(self.state["original_image"], self.state["original_text"], self.sam, tool_params)
                verdict = self._process_segmentation_result(result, "image_enhancer")
                self.state["tool_history"].append({
                    "tool": "image_enhancer",
                    "verdict": verdict,
                    "iteration": self.state["iteration"],
                })
                if verdict == "Accept":
                    print("MLLM接受当前结果，结束流程")
                    break

            else:
                print(f"未知操作: {action}, 尝试直接分割")
                # 直接将当前图和文本提示交给分割模型处理，得到带有mask的分割结果图片
                result = self.sam.segment_with_text(self.state["original_image"], self.state["original_text"])
                verdict = self._process_segmentation_result(result, "direct_segmentation")
                self.state["tool_history"].append({
                    "tool": f"unknown({action})",
                    "verdict": verdict,
                    "iteration": self.state["iteration"],
                })
                if verdict == "Accept":
                    print("MLLM接受当前结果，结束流程")
                    break
            
            # 如果没有Accept，继续下一轮循环
        
        # 4. 返回最佳结果，最终的带有mask的图片
        return self.get_final_result()
    
    def create_high_res_visuals(self, original_image, masks_data, draw_numbers=True):
        """生成高清的可视化图像（放大后绘制，确保清晰）"""
        target_min_pixels = Config.MLLM_MIN_PIXELS
        
        # 1. 计算放大倍数
        current_pixels = original_image.width * original_image.height
        scale_factor = 1.0
        
        if current_pixels < target_min_pixels:
            scale_factor = (target_min_pixels / current_pixels) ** 0.5
            scale_factor = min(scale_factor, 4.0) # 限制最大放大倍数
        
        if scale_factor <= 1.05:
            # 不需要放大，直接绘制
            return self.sam.visualize_masks_with_numbers(original_image, masks_data, draw_numbers)
            
        # 2. 放大图像
        new_size = (int(original_image.width * scale_factor), int(original_image.height * scale_factor))
        # print(f"可视化时放大图像到: {new_size}")
        high_res_image = original_image.resize(new_size, Image.Resampling.BICUBIC)
        
        # 3. 放大Mask
        high_res_masks_data = []
        for item in masks_data:
            mask = item["mask"]
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_resized_img = mask_img.resize(new_size, Image.Resampling.NEAREST)
            mask_resized = np.array(mask_resized_img) > 128
            
            new_item = item.copy()
            new_item["mask"] = mask_resized
            high_res_masks_data.append(new_item)
            
        # 4. 在放大图上绘制
        return self.sam.visualize_masks_with_numbers(high_res_image, high_res_masks_data, draw_numbers)

    def _process_segmentation_result(self, result, method):
        """处理分割结果(公共逻辑)"""
        if not result or not result.get("success"):
            print(f"{method} 分割失败或未返回结果")
            return "Reject"
            
        best_seg = result["best_result"]
        print(f"{method} 成功，分数: {best_seg['score']:.3f}")
        
        try:
            # 构建候选掩码列表 (现有 + 新增)
            candidates = self.state["accepted_masks"].copy()
            candidates.append({
                "mask": best_seg["mask"],
                "score": best_seg["score"],
                "method": method
            })
            
            # 准备检查用的可视化数据
            masks_for_vis = []
            for i, item in enumerate(candidates):
                masks_for_vis.append({
                    "mask": item["mask"],
                    "color": self.get_color(i),
                    "id": i + 1
                })
            
            # 生成检查图像 (高清)
            check_image = self.create_high_res_visuals(
                self.state["original_image"],
                masks_for_vis
            )
            
            # 保存中间结果用于检查
            timestamp = datetime.now().strftime("%H%M%S")
            save_path = Config.OUTPUT_DIR / f"iter{self.state['iteration']}_{method}_{timestamp}_check.jpg"
            check_image.save(save_path)
            print(f"检查图像已保存: {save_path}")
            
            # 评估结果
            print("调用MLLM评估分割质量...")
            # 注意: 评估时传入的 original_image 也应该是高清对应的版本，或者直接传原图让 MLLM 内部去放大？
            # MLLM 内部已经有了 smart_resize。如果我们传给它的是 check_image (已经是放大的)，和 original_image (未放大)。
            # 为了对比方便，最好让 MLLM 看到的两张图尺寸一致。
            # 但 mllm_processor.segmentation_evaluation 内部有 image_resolution_params，会强制 MLLM把它们都看作大图。
            # 所以这里可以直接传原图和高清check图。Qwen 会把原图放大匹配。
            
            evaluation_result = self.mllm.segmentation_evaluation(
                self.state["original_image"],
                check_image,
                self.state["original_text"]
            )
            verdict, rejected_indices = self._normalize_segmentation_evaluation_result(
                evaluation_result
            )
            
            print(f"评估结论: {verdict}")
            if verdict == "Reject":
                # 确保 rejected_indices 都是整数
                if rejected_indices:
                    rejected_indices = [int(x) for x in rejected_indices]
                else:
                    # 如果拒绝但没有指定序号，默认拒绝最后一个（新增的那个）
                    rejected_indices = [len(candidates)]
                    print(f"警告: 拒绝但未检测到序号，默认为最后一个: {rejected_indices}")
                
                print(f"拒绝的序号: {rejected_indices}")
            
            # 根据评估更新状态
            if verdict == "Accept":
                self.state["accepted_masks"] = candidates
                
                # 更新 current_image 用于下一轮 (下一轮 MLLM 思考用)
                # 使用高清图作为 current_image
                self.state["current_image"] = check_image
                self.state["final_image"] = check_image
                
                return "Accept"
            else:
                # Reject - 过滤掉拒绝的Mask，保留其他的
                kept_candidates = []
                for i, item in enumerate(candidates):
                    idx = i + 1
                    if idx not in rejected_indices:
                        kept_candidates.append(item)
                
                self.state["accepted_masks"] = kept_candidates
                print(f"保留了 {len(kept_candidates)} 个掩码 (删除了 {len(rejected_indices)} 个)")
                
                # 更新 current_image
                masks_for_vis_kept = []
                for i, item in enumerate(kept_candidates):
                    masks_for_vis_kept.append({
                        "mask": item["mask"],
                        "color": self.get_color(i),
                        "id": i + 1
                    })
                
                if kept_candidates:
                    self.state["current_image"] = self.create_high_res_visuals(
                        self.state["original_image"],
                        masks_for_vis_kept
                    )
                    self.state["final_image"] = self.state["current_image"]
                else:
                    self.state["current_image"] = None # 回退到原图
                    self.state["final_image"] = self.state["original_image"].copy()
                
                return "Reject"
            
        except Exception as e:
            print(f"处理分割结果出错: {e}")
            import traceback
            traceback.print_exc()
            return "Reject"

    def load_image(self, image_path):
        """加载图像"""
        try:
            image = Image.open(image_path)
            image = ensure_rgb(image)
            image = resize_image(image, max_size=1024)
            
            self.state["original_image"] = image.copy()
            self.state["final_image"] = image.copy()
            
            print(f"图像加载成功: {image.size}")
        except Exception as e:
            raise ValueError(f"无法加载图像: {e}")
    
    
    
    def get_final_result(self):
        """获取最终结果"""
        print(f"\n=== 流程结束 ===")
        print(f"总迭代次数: {self.state['iteration']}")
        
        accepted = self.state["accepted_masks"]
        
        if accepted:
            print(f"最终接受掩码数量: {len(accepted)}")
            
            try:
                # 准备可视化数据
                masks_for_vis = []
                for i, item in enumerate(accepted):
                    masks_for_vis.append({
                        "mask": item["mask"],
                        "color": self.get_color(i),
                        "id": i + 1
                    })
                
                # 生成最终图像
                final_image = self.sam.visualize_masks_with_numbers(
                    self.state["original_image"],
                    masks_for_vis,
                    draw_numbers=False # 最终结果不要带数字
                )
                
                # 保存最终结果
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = Config.OUTPUT_DIR / f"final_result_{timestamp}.jpg"
                final_image.save(save_path)
                
                # 保存Mask
                mask_path = Config.OUTPUT_DIR / f"final_mask_{timestamp}.npy"
                # Save all masks
                masks_array = [item["mask"] for item in accepted]
                try: 
                    np.save(mask_path, masks_array)
                except:
                    pass
                
                return {
                    "success": True,
                    "final_image_path": str(save_path),
                    "mask_path": str(mask_path),
                    "best_score": max([m["score"] for m in accepted]) if accepted else 0.0,
                    "iterations": self.state["iteration"],
                    "mask_count": len(accepted),
                    "accepted_masks": accepted
                }
                
            except Exception as e:
                print(f"保存最终结果出错: {e}")
                import traceback
                traceback.print_exc()
                return {"success": False, "message": str(e)}
        else:
            print("未找到有效的结果")
            return {
                "success": False, 
                "message": "No accepted masks found",
                "iterations": self.state["iteration"]
            }

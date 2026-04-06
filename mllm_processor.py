import torch
import json
import re
import base64
from io import BytesIO
import re
import json
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from config import Config
from utils.image_utils import add_grid_to_image, resize_image, ensure_rgb, smart_resize_for_mllm
from utils.shared_qwen_backbone import SharedQwenBackbone

class MLLMProcessor:
    def __init__(self):
        self.device = Config.DEVICE
        self.model, self.processor = self._load_model()
        self.shared_backbone = SharedQwenBackbone(self.model, self.processor)
        print("MLLM处理器初始化完成")
        
    def _load_model(self):
        """加载Qwen2.5-VL模型 - 使用官方推荐的方式"""
        print(f"加载Qwen2.5-VL模型: {Config.QWEN_MODEL_PATH}")
        
        try:
            # 加载处理器
            processor = AutoProcessor.from_pretrained(
                Config.QWEN_MODEL_PATH,
                trust_remote_code=True
            )
            
            # 加载模型
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                Config.QWEN_MODEL_PATH,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            model.eval()
            print("✓ Qwen2.5-VL模型加载成功")
            return model, processor
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise
    
    def process(self, image, text_prompt,grid_info,image_path,iteration=1, processed_image=None):
        """
        处理图像和文本输入
        
        Args:
            image: PIL Image
            text_prompt: 文本提示
            iteration: 当前迭代次数
            processed_image: 处理后的图像
            processed_image如果是None的话，代表是第一次迭代
            
        Returns:
            dict: 解析后的响应
            str: 原始文本
        """
        try:
            # 1. 准备并调整图像分辨率 (smart_resize_for_mllm)
            # 确保原始图像根据 Qwen2.5-VL 要求对齐 (28的倍数) 并尽可能放大
            if iteration > 1 and processed_image is not None:
                # 如果是后续迭代，但 image_path 指向原始文件，我们需要加载并处理它
                 # 注意：这里假设 process 被调用时 image_path 始终指向那个"Context Image"
                try:
                    original_pil = Image.open(image_path)
                except:
                    original_pil = image
            else:
                original_pil = image
                
            aligned_original_image = smart_resize_for_mllm(original_pil)
            temp_original_image_path = Config.BASE_DIR / "temp_original_image_aligned.jpg"
            aligned_original_image.save(temp_original_image_path)

            # 2. 准备用于画网格的底图 (同样进行放大和对齐处理)
            if processed_image is not None:
                base_image_for_grid = smart_resize_for_mllm(processed_image)
            else:
                base_image_for_grid = smart_resize_for_mllm(image)
                
            # 3. 绘制高清网格
            # 计算适合当前分辨率的动态Padding，并确保是对其的
            w, h = base_image_for_grid.size
            max_dim = max(w, h)
            # 目标padding: ~4-5% 图像尺寸，保证能放下自动缩放的字体
            target_padding = max(28, int(max_dim * 0.05))
            # 向上取整到14的倍数，确保 2*padding 是28的倍数
            padding = ((target_padding + 13) // 14) * 14
            
            grid_processed_image, grid_processed_image_info = add_grid_to_image(
                base_image_for_grid,
                Config.GRID_ROWS,
                Config.GRID_COLS,
                line_thickness=None, # 让其自动根据分辨率计算
                padding=padding 
            )
            
            # !!! 再次对齐 !!! 
            # 理论上如果 padding是14的倍数，这里不需要再次 resize，但保留作为保险
            # 由于尺寸已经对其，这一步应该不会改变图像内容
            grid_processed_image = smart_resize_for_mllm(grid_processed_image)
            
            if processed_image is None:
                grid_info = grid_processed_image_info
            
            grid_image_path = Config.BASE_DIR / "temp_grid_image.jpg"
            grid_processed_image.save(grid_image_path)
            
            # 图像分辨率控制参数

            image_resolution_params = {
                "min_pixels": Config.MLLM_MIN_PIXELS,
                "max_pixels": Config.MLLM_MAX_PIXELS,
            }

            # 构建消息 - 按照官方文档格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": self.get_prompt_text(text_prompt)
                        },
                        {
                            "type": "image",
                            "image": str(temp_original_image_path),
                            **image_resolution_params
                        },
                        {
                            "type": "image",
                            "image": str(grid_image_path),
                            **image_resolution_params
                        },
                    ],
                }
            ]
            
            # 准备推理 - 按照官方文档
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 处理视觉信息
            image_inputs, video_inputs = process_vision_info(messages)
            
            # 准备模型输入
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            # 推理：生成输出
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            
            # 去除输入部分
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # 解码为文本
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            # 清理临时文件
            # if temp_image_path.exists():
            #     temp_image_path.unlink()
            
            print(f"MLLM原始输出: {output_text}")
            
            # 解析响应
            if isinstance(output_text, list):
                output_text = output_text[0] if output_text else ""
            
            response = self._parse_response(output_text)
            
            return response, output_text
            
        except Exception as e:
            print(f"MLLM处理出错: {e}")
            import traceback
            traceback.print_exc()
            return {
                "action": "segment",
                "reason": f"处理错误: {str(e)[:100]}",
                "tool_params": {}
            }, ""
    
    def get_prompt_text(self, text_prompt):
        """获取完整的提示词文本"""
        with open(Config.SYSTEM_PROMPT, "r", encoding="utf-8") as f:
            system_prompt = f.read()
        user_prompt = f"""
Initial user input query: {text_prompt}
Below are the original image and the image  with grid respectively"""
        
        return system_prompt+"\n\n"+user_prompt
    
    def get_check_prompt(self,text_prompt):
        """获取用于迭代检查的提示词文本"""
        with open(Config.SYSTEM_PROMPT_ITERATIVE_CHECKING, "r", encoding="utf-8") as f:
            system_prompt = f.read()
        
        user_prompt = f"""Initial user input query: {text_prompt}
    Below are the original image and the image overlaid with the predicted segmentation mask respectively:"""
        return system_prompt+"\n\n"+user_prompt
        
    def segmentation_evaluation(self, original_image, masked_image, text_prompt):
        """评估分割结果，返回Accept或Reject"""
        try:
            # 保存临时图像
            eval_orig_path = Config.BASE_DIR / "temp_eval_orig.jpg"
            eval_mask_path = Config.BASE_DIR / "temp_eval_mask.jpg"
            
            original_image.save(eval_orig_path)
            masked_image.save(eval_mask_path)
            
            # 构建消息 - 按照官方文档格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": self.get_check_prompt(text_prompt)
                        },
                        {"type": "image","image": str(eval_orig_path),},
                        {"type": "image","image": str(eval_mask_path),},
                    ],
                }
            ]
            
            # 准备推理 - 按照官方文档
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 处理视觉信息
            image_inputs, video_inputs = process_vision_info(messages)
            
            # 准备模型输入
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            # 推理：生成输出
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            
            # 去除输入部分
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # 解码为文本
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            if isinstance(output_text, list):
                output_text = output_text[0]

            print(f"MLLM评估输出: {output_text}")
            
            # 清理临时文件
            if eval_orig_path.exists():
                eval_orig_path.unlink()
            if eval_mask_path.exists():
                eval_mask_path.unlink()
            
            # 解析 Verdict
            verdict_match = re.search(r"<verdict>\s*(Accept|Reject)\s*</verdict>", output_text, re.IGNORECASE)
            rejected_indices = []
            
            if verdict_match:
                verdict = verdict_match.group(1).capitalize()
                print(f"解析Verdict结果: {verdict}")
                
                if verdict == "Reject":
                    # 解析 Reject 的 indices
                    index_match = re.search(r"<index>(.*?)</index>", output_text, re.DOTALL)
                    if index_match:
                        try:
                            index_json = index_match.group(1).strip()
                            data = json.loads(index_json)
                            if "mark" in data:
                                rejected_indices = data["mark"]
                        except Exception as e:
                            print(f"解析 Index JSON 失败: {e}")
                            # 备用解析
                            rejected_indices = [int(n) for n in re.findall(r"\d+", index_match.group(1))]
                    print(f"拒绝的掩码索引: {rejected_indices}")

                return verdict, rejected_indices
            
            print("未找到有效的Verdict标签，默认返回Reject")
            return "Reject", []
            
        except Exception as e:
            print(f"MLLM评估出错: {e}")
            import traceback
            traceback.print_exc()
            return ""
        
    
    
    def _parse_response(self, text):
        """解析模型响应，提取<tool>标签内的JSON"""
        try:
            # 使用正则表达式提取<tool>标签中的内容
            match = re.search(r'<tool>(.*?)</tool>', text, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                # 移除可能的markdown代码块标记
                if json_str.startswith("```"):
                    json_str = json_str.strip("`").strip("json").strip()
                return json.loads(json_str)
            
            print("未找到<tool>标签")
            return self._get_default_response()
            
        except Exception as e:
            print(f"解析响应失败: {e}")
            return self._get_default_response()

    def _get_default_response(self):
        """获取默认响应"""
        print("未找到有效操作，使用默认响应")
        return {
            "name": "report_no_mask",
            "parameters": {}
        }
    

    

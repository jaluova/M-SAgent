import torch
import json
import re
import base64
import html
from io import BytesIO
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from config import Config
from utils.image_utils import add_grid_to_image, resize_image, ensure_rgb, smart_resize_for_mllm
from utils.shared_qwen_backbone import SharedQwenBackbone

class MLLMProcessor:
    SUPPORTED_TOOL_NAMES = {
        "object_locator",
        "concept_generator",
        "image_enhancer",
        "report_no_mask",
    }

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
    
    def process(self, image, text_prompt, grid_info, image_path, iteration=1, processed_image=None, tool_history=None):
        """
        处理图像和文本输入

        Args:
            image: PIL Image
            text_prompt: 文本提示
            iteration: 当前迭代次数
            processed_image: 处理后的图像（None 代表第一次迭代）
            tool_history: 之前迭代的工具调用历史

        Returns:
            dict: 解析后的响应
            str: 原始文本
        """
        temp_paths = []
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
                
            aligned_original_image = smart_resize_for_mllm(ensure_rgb(original_pil))
            temp_original_image_path = Config.BASE_DIR / "temp_original_image_aligned.jpg"
            aligned_original_image.save(temp_original_image_path)
            temp_paths.append(temp_original_image_path)

            # 2. 准备用于画网格的底图。
            # 后续迭代优先在已有 mask overlay 上叠加网格，帮助模型把候选 mask 和坐标对应起来。
            has_mask_overlay = processed_image is not None
            if has_mask_overlay:
                base_image_for_grid = smart_resize_for_mllm(ensure_rgb(processed_image))
            else:
                base_image_for_grid = aligned_original_image.copy()
                
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
            temp_paths.append(grid_image_path)
            
            # 图像分辨率控制参数

            image_resolution_params = {
                "min_pixels": Config.MLLM_MIN_PIXELS,
                "max_pixels": Config.MLLM_MAX_PIXELS,
            }

            # 构建消息 - system prompt 放在 system role
            system_prompt_text = self._read_system_prompt()
            user_prompt_text = self._build_user_prompt(
                text_prompt,
                tool_history,
                has_mask_overlay=has_mask_overlay,
            )

            user_content = [
                {"type": "text", "text": user_prompt_text},
                {
                    "type": "image",
                    "image": str(temp_original_image_path),
                    **image_resolution_params,
                },
                {
                    "type": "image",
                    "image": str(grid_image_path),
                    **image_resolution_params,
                },
            ]

            if has_mask_overlay:
                mask_overlay_image_path = Config.BASE_DIR / "temp_mask_overlay_image.jpg"
                smart_resize_for_mllm(ensure_rgb(processed_image)).save(mask_overlay_image_path)
                temp_paths.append(mask_overlay_image_path)
                user_content.append(
                    {
                        "type": "image",
                        "image": str(mask_overlay_image_path),
                        **image_resolution_params,
                    }
                )

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt_text}],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
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
        finally:
            for temp_path in temp_paths:
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except OSError as cleanup_error:
                    print(f"清理临时文件失败: {temp_path} -> {cleanup_error}")
    
    def _read_system_prompt(self):
        """读取 system prompt 文件内容"""
        with open(Config.SYSTEM_PROMPT, "r", encoding="utf-8") as f:
            return f.read()

    def _build_user_prompt(self, text_prompt, tool_history=None, has_mask_overlay=False):
        """构建 user prompt，含查询文本和历史摘要"""
        parts = [f"Initial user input query: {text_prompt}"]

        if tool_history:
            parts.append("")
            parts.append("=== IMPORTANT: PREVIOUS ATTEMPTS ===")
            for entry in tool_history:
                details = []
                score = entry.get("score")
                if isinstance(score, (int, float)):
                    details.append(f"score={float(score):.3f}")
                note = entry.get("note")
                if note:
                    details.append(f"note={note}")
                details_suffix = f" ({'; '.join(details)})" if details else ""
                parts.append(
                    f"- Iteration {entry['iteration']}: Called \"{entry['tool']}\" -> Result: {entry['verdict']}{details_suffix}"
                )
            parts.append(
                "WARNING: If a previous attempt was rejected, you MUST NOT repeat the same "
                "tool with near-identical parameters. Choose a DIFFERENT tool or make a "
                "substantial change in points, concepts, or crop region."
            )
            parts.append(
                "Treat previous tool outputs as hypotheses, not proof. Re-check the original image carefully."
            )
            parts.append("===")

        parts.append("")
        if has_mask_overlay:
            parts.append(
                "You will receive three images in order:"
            )
            parts.append(
                "1. the original image;"
            )
            parts.append(
                "2. the current context image with a grid overlaid on it (this grid image may already include previous mask overlays);"
            )
            parts.append(
                "3. the current accepted-mask overlay image without the grid."
            )
            parts.append(
                "Treat the accepted-mask overlay as context from previous iterations, not proof that those masks are correct."
            )
        else:
            parts.append("You will receive two images in order:")
            parts.append("1. the original image;")
            parts.append("2. the original image with the grid overlaid.")
        return "\n".join(parts)

    def get_prompt_text(self, text_prompt):
        """获取完整的提示词文本（保留向后兼容）"""
        system_prompt = self._read_system_prompt()
        user_prompt = self._build_user_prompt(text_prompt)
        return system_prompt + "\n\n" + user_prompt
    
    def _read_check_system_prompt(self):
        """读取迭代检查的 system prompt"""
        with open(Config.SYSTEM_PROMPT_ITERATIVE_CHECKING, "r", encoding="utf-8") as f:
            return f.read()

    def get_check_prompt(self, text_prompt):
        """获取用于迭代检查的提示词文本（保留向后兼容）"""
        system_prompt = self._read_check_system_prompt()
        user_prompt = (
            f"Initial user input query: {text_prompt}\n"
            "You will receive three images in order:\n"
            "1. the original image;\n"
            "2. the whole-image mask overlay;\n"
            "3. the zoomed-in review image, which contains two side-by-side panels: "
            "a tight mask-focused crop and a larger context crop around the same candidate."
        )
        return system_prompt + "\n\n" + user_prompt
        
    def segmentation_evaluation(self, original_image, masked_image, text_prompt, zoomed_image=None):
        """评估分割结果，返回Accept或Reject"""
        eval_orig_path = Config.BASE_DIR / "temp_eval_orig.jpg"
        eval_mask_path = Config.BASE_DIR / "temp_eval_mask.jpg"
        eval_zoom_path = Config.BASE_DIR / "temp_eval_zoom.jpg"
        try:
            # 保存临时图像
            original_image.save(eval_orig_path)
            masked_image.save(eval_mask_path)
            zoom_source = zoomed_image if zoomed_image is not None else masked_image
            zoom_source.save(eval_zoom_path)

            image_resolution_params = {
                "min_pixels": Config.MLLM_MIN_PIXELS,
                "max_pixels": Config.MLLM_MAX_PIXELS,
            }

            # 构建消息 - system prompt 放在 system role
            check_system_prompt = self._read_check_system_prompt()
            check_user_prompt = (
                f"Initial user input query: {text_prompt}\n"
                "You will receive three images in order:\n"
                "1. the original image;\n"
                "2. the whole-image mask overlay;\n"
                "3. the zoomed-in review image, which contains two side-by-side panels: "
                "a tight mask-focused crop and a larger context crop around the same candidate."
            )

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": check_system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": check_user_prompt},
                        {
                            "type": "image",
                            "image": str(eval_orig_path),
                            **image_resolution_params,
                        },
                        {
                            "type": "image",
                            "image": str(eval_mask_path),
                            **image_resolution_params,
                        },
                        {
                            "type": "image",
                            "image": str(eval_zoom_path),
                            **image_resolution_params,
                        },
                    ],
                },
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
            
            if isinstance(output_text, list):
                output_text = output_text[0]

            print(f"MLLM评估输出: {output_text}")

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
            return "Reject", []
        finally:
            for temp_path in (eval_orig_path, eval_mask_path, eval_zoom_path):
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except OSError as cleanup_error:
                    print(f"清理临时文件失败: {temp_path} -> {cleanup_error}")
        
    
    
    def _parse_response(self, text):
        """解析模型响应，兼容标准 <tool> 和直接工具标签格式。"""
        try:
            tag_pattern = re.compile(
                r"<(?P<tag>[a-zA-Z_][\w-]*)\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?P=tag)>",
                re.DOTALL,
            )
            for match in tag_pattern.finditer(text):
                tag = (match.group("tag") or "").strip()
                attrs = (match.group("attrs") or "").strip()
                body = (match.group("body") or "").strip()

                if tag == "tool":
                    parsed = self._parse_standard_tool_tag(attrs, body)
                elif tag in self.SUPPORTED_TOOL_NAMES:
                    parsed = self._parse_direct_tool_tag(tag, attrs, body)
                else:
                    parsed = None

                if parsed is not None:
                    return parsed

            print("未找到<tool>标签")
            return self._get_default_response()
            
        except Exception as e:
            print(f"解析响应失败: {e}")
            return self._get_default_response()

    def _parse_standard_tool_tag(self, attrs, body):
        if body:
            parsed = self._parse_tool_json_body(body)
            if parsed is not None:
                return parsed

        if attrs:
            return self._parse_tool_attributes(attrs)

        return None

    def _parse_direct_tool_tag(self, tag, attrs, body):
        parameters = self._parse_named_attributes(attrs)
        if parameters is None:
            return None

        if body:
            body_parameters = self._parse_tool_json_body(body)
            if isinstance(body_parameters, dict):
                parameters.update(body_parameters)

        return {
            "name": tag,
            "parameters": parameters,
        }

    def _parse_tool_json_body(self, body):
        json_str = html.unescape(body.strip())
        if json_str.startswith("```"):
            json_str = re.sub(r"^```(?:json)?\s*", "", json_str, flags=re.IGNORECASE)
            json_str = re.sub(r"\s*```$", "", json_str)
        json_str = json_str.rstrip(" \t\r\n;")
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    def _parse_tool_attributes(self, attrs):
        name_match = re.search(r"""name\s*=\s*(['"])(.*?)\1""", attrs, re.DOTALL)
        params_match = re.search(r"""parameters\s*=\s*(['"])(.*?)\1""", attrs, re.DOTALL)
        if not name_match:
            return None

        name = html.unescape(name_match.group(2).strip())
        params_text = html.unescape(params_match.group(2).strip()) if params_match else "{}"
        try:
            parameters = json.loads(params_text) if params_text else {}
        except json.JSONDecodeError:
            return None

        return {
            "name": name,
            "parameters": parameters,
        }

    def _parse_named_attributes(self, attrs):
        parameters = {}
        for key, quote, raw_value in re.findall(r"""([a-zA-Z_][\w-]*)\s*=\s*(['"])(.*?)\2""", attrs, re.DOTALL):
            if key in {"name", "parameters"}:
                continue

            value_text = html.unescape(raw_value.strip())
            try:
                parameters[key] = json.loads(value_text)
            except json.JSONDecodeError:
                parameters[key] = value_text

        return parameters

    def _get_default_response(self):
        """解析失败时的默认响应：用 concept_generator 重试，而非直接终止"""
        print("未找到有效操作，回退到 concept_generator 重试")
        return {
            "name": "concept_generator",
            "parameters": {
                "new_concepts": [],
                "num_concepts": 0,
            },
        }
    

    

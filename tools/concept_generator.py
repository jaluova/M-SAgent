from config import Config
import json
from datetime import datetime

class ConceptGenerator:
    """概念生成工具"""
    
    def __init__(self, mllm_processor):
        self.mllm = mllm_processor
        
    def segment_with_concept(self, tool_params, sam_processor, image, query=None):
        """使用生成的概念进行分割
        Args:
            tool_params (dict): 来自MLLM的工具参数，包含概念列表等
            sam_processor: SAM处理器
            image: 待分割图像
            query: 原始用户查询文本，作为最终 fallback concept
        Returns:
            dict: 概念分割结果
        """
        new_concepts = tool_params.get("new_concepts", [])

        # 参数容错：MLLM 可能用了错误的 key 名
        if not new_concepts:
            for fallback_key in ("text", "concept", "text_prompt", "query", "prompt"):
                fallback_val = tool_params.get(fallback_key)
                if fallback_val:
                    if isinstance(fallback_val, str):
                        new_concepts = [fallback_val]
                    elif isinstance(fallback_val, list):
                        new_concepts = [str(v) for v in fallback_val if v]
                    if new_concepts:
                        print(f"concept_generator: 从参数 '{fallback_key}' 提取到概念: {new_concepts}")
                        break

        # 最终兜底：用原始查询作为 concept
        if not new_concepts and query:
            new_concepts = [query]
            print(f"concept_generator: 使用原始查询作为 fallback concept: {query}")

        num_concepts = tool_params.get("num_concepts", len(new_concepts))

        print(f"提取到的概念列表: {new_concepts}")
        print(f"概念数量: {num_concepts}")

        # 测试每个概念
        best_result = None
        best_score = -1
        results_list = []
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, concept in enumerate(new_concepts[:num_concepts]):
            print(f"测试概念: {concept}")

            seg_result = sam_processor.segment_with_text(
                image,
                text_prompt=concept
            )

            if seg_result and seg_result.get("success") and seg_result.get("results"):
                current_best = seg_result["best_result"]
                current_best["concept"] = concept
                current_best["method"] = "concept_generator"
                
                # 保存可视化结果
                try:
                    mask = current_best["mask"]
                    vis_image = sam_processor.apply_mask_to_image(image, mask)
                    safe_concept = "".join([c if c.isalnum() else "_" for c in concept])
                    vis_path = Config.TOOL_CALLS_LOG / f"concept_generator_{timestamp}_{i}_{safe_concept}.jpg"
                    vis_image.save(vis_path)
                    print(f"概念 '{concept}' 分割结果已保存: {vis_path}")
                except Exception as e:
                    print(f"保存概念分割结果失败: {e}")
                
                results_list.append(current_best)
                
                if current_best["score"] > best_score:
                    best_score = current_best["score"]
                    best_result = current_best

        return {
            "success": best_result is not None,
            "results": results_list,
            "best_result": best_result
        }
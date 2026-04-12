import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_results(original_image, grid_image, segmentation_result, save_path=None):
    """
    可视化结果
    
    Args:
        original_image: 原始图像
        grid_image: 带网格的图像
        segmentation_result: 分割结果
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始图像
    axes[0].imshow(original_image)
    axes[0].set_title("原始图像")
    axes[0].axis('off')
    
    # 网格图像
    axes[1].imshow(grid_image)
    axes[1].set_title("带网格的图像")
    axes[1].axis('off')
    
    # 分割结果
    if segmentation_result and "visualization" in segmentation_result:
        axes[2].imshow(segmentation_result["visualization"])
        conf = segmentation_result.get("confidence", 0)
        axes[2].set_title(f"分割结果 (置信度: {conf:.2f})")
    else:
        axes[2].imshow(original_image)
        axes[2].set_title("分割结果 (无)")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    plt.show()

def create_comparison_grid(segmentation_history):
    """
    创建分割历史比较图
    
    Args:
        segmentation_history: 分割历史列表
        
    Returns:
        PIL Image: 比较图
    """
    if not segmentation_history:
        return None
    
    # 收集所有可视化结果
    vis_images = []
    captions = []
    
    for history in segmentation_history:
        result = history.get("result", {})
        if result.get("success") and "visualization" in result:
            vis_images.append(result["visualization"])
            caption = f"迭代{history['iteration']}: {history['text_prompt'][:20]}..."
            captions.append(caption)
    
    if not vis_images:
        return None
    
    # 创建网格
    n = len(vis_images)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    # 计算每个子图的大小
    sample_size = vis_images[0].size
    grid_width = sample_size[0] * cols
    grid_height = sample_size[1] * rows + 30 * rows  # 为标题留空间
    
    # 创建画布
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # 绘制每个结果
    for i, (img, caption) in enumerate(zip(vis_images, captions)):
        row = i // cols
        col = i % cols
        
        x = col * sample_size[0]
        y = row * (sample_size[1] + 30)
        
        # 粘贴图像
        grid_image.paste(img, (x, y))
        
        # 添加标题（简化版，没有PIL字体）
        from PIL import ImageDraw
        draw = ImageDraw.Draw(grid_image)
        draw.text((x + 10, y + sample_size[1] + 5), caption, fill='black')
    
    return grid_image
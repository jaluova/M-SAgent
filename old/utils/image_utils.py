import cv2
import numpy as np
from PIL import Image
import os
from config import Config

def add_grid_to_image(image, num_rows=10, num_cols=10, line_color=(255, 0, 0), 
                     text_color=(0, 0, 0), line_thickness=None, padding=30):
    """
    为图像添加网格，并在外围添加白边显示坐标
    
    Args:
        image: PIL Image 或 numpy array
        num_rows: 行数
        num_cols: 列数
        line_color: 网格线颜色 (R, G, B)
        text_color: 文本颜色 (R, G, B)
        line_thickness: 线宽 (如果为None则根据图像大小自动计算)
        padding: 白边宽度
        
    Returns:
        PIL Image: 带网格和坐标的图像
        dict: 网格信息
    """
    # 转换为numpy array
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image.copy()
    
    height, width = image_np.shape[:2]
    max_dim = max(height, width)

    # --- 动态视觉参数计算 ---
    # 这里的参数经过调优，专为MLLM视觉理解设计
    
    # 1. 线宽: 每500像素增加1px，最细2px, 最粗6px
    if line_thickness is None:
        line_thickness = max(2, min(6, int(max_dim / 500) + 1))
        
    # 2. 字体大小: 基础0.6，随尺寸线性增加
    font_scale = max(0.6, max_dim / 800.0)
    font_thickness = max(1, int(font_scale * 2))
    
    # 3. 锚点半径: 比线稍粗
    dot_radius = int(line_thickness * 1.5)
    dot_color = (0, 255, 0) # 亮绿色锚点用于辅助定位交汇处

    # 创建带白边的画布
    padded_height = height + 2 * padding
    padded_width = width + 2 * padding
    
    # 默认白色背景
    padded_image = np.ones((padded_height, padded_width, 3), dtype=np.uint8) * 255
    
    # 将原图复制到中心位置
    padded_image[padding:padding+height, padding:padding+width] = image_np
    
    # 在原图区域绘制网格线
    row_positions = np.linspace(0, height, num_rows + 1, dtype=int)
    col_positions = np.linspace(0, width, num_cols + 1, dtype=int)
    
    # 确保边界正确
    row_positions[-1] = height - 1
    col_positions[-1] = width - 1
    
    # 绘制水平线 (注意加上padding偏移)
    for y in row_positions:
        cv2.line(padded_image, (padding, y + padding), (width + padding, y + padding), 
                 line_color, line_thickness)
    
    # 绘制垂直线
    for x in col_positions:
        cv2.line(padded_image, (x + padding, padding), (x + padding, height + padding), 
                 line_color, line_thickness)

    # 绘制行与列的交汇点
    for y in row_positions:
        for x in col_positions:
            cv2.circle(padded_image, (x + padding, y + padding), dot_radius, dot_color, -1)

        
    # 在左侧显示行号 (0到num_rows, 从上到下)
    for i in range(num_rows + 1):
        # 行号从上到下递增，与线对齐
        y = row_positions[i]
        text = f"{i}"
        
        # 跳过0，单独绘制
        if text == "0":
            continue

        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        # 让数字贴近线（靠右对齐，留一点间距，自适应间距）
        text_margin = int(5 * font_scale)
        text_x = padding - text_size[0] - text_margin
        text_y = y + padding + text_size[1] // 2
        
        cv2.putText(padded_image, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
    
    # 在顶部显示列号 (0到num_cols)
    for j in range(num_cols + 1):
        # 列号从左到右递增，与线对齐
        x = col_positions[j]
        text = f"{j}"
        
        # 跳过0，单独绘制
        if text == "0":
            continue

        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = x + padding - text_size[0] // 2
        # 让数字贴近线（靠下对齐，留一点间距）
        text_margin = int(5 * font_scale)
        text_y = padding - text_margin
        
        cv2.putText(padded_image, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    # 绘制原点 0
    text = "0"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
    text_margin = int(5 * font_scale)
    # 放置在左上角
    text_x = padding - text_size[0] - text_margin
    text_y = padding - text_margin
    cv2.putText(padded_image, text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
    
    # 添加标签
    grid_info = {
        'rows': num_rows,
        'cols': num_cols,
        'cell_width': width / num_cols,
        'cell_height': height / num_rows,
        'row_positions': row_positions,
        'col_positions': col_positions,
        'padding': padding  # 记录padding信息
    }
    
    # 转换为PIL Image
    grid_pil = Image.fromarray(padded_image)
    
    return grid_pil, grid_info

def smart_resize_for_mllm(image, min_pixels=None, max_pixels=None):
    """
    智能调整图像大小以适应MLLM输入要求：
    1. 确保图像尺寸是28的整数倍（Qwen2.5-VL要求）
    2. 尽可能放大图像以提升细节识别能力（不低于min_pixels）
    """
    if min_pixels is None:
        min_pixels = Config.MLLM_MIN_PIXELS
    if max_pixels is None:
        max_pixels = Config.MLLM_MAX_PIXELS
        
    w, h = image.size
    current_pixels = w * h
    
    scale_factor = 1.0
    
    # 策略：尽可能放大，达到min_pixels以上
    if current_pixels < min_pixels:
        scale_factor = (min_pixels / current_pixels) ** 0.5
        # 允许更大的放大倍数，但设置合理上限以防显存爆炸或无效放大
        # Qwen2.5-VL 处理动态分辨率，但过大图像会增加推理成本
        # 既然用户要求“尽可能放大”，我们放宽限制
        scale_factor = max(scale_factor, 1.0)
        
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    # 确保是28的整数倍 (向上取整)
    def align_to_28(n):
        return ((n + 27) // 28) * 28
    
    final_w = align_to_28(new_w)
    final_h = align_to_28(new_h)
    
    # 检查是否超过最大像素限制，如果超过则按比例缩小至限制内，并保持28倍数
    if final_w * final_h > max_pixels:
        downscale = (max_pixels / (final_w * final_h)) ** 0.5
        final_w = int(final_w * downscale)
        final_h = int(final_h * downscale)
        final_w = align_to_28(final_w)
        final_h = align_to_28(final_h)
        # 确保至少是28x28
        final_w = max(28, final_w)
        final_h = max(28, final_h)
    
    if final_w != w or final_h != h:
        # print(f"Resizing image from {w}x{h} to {final_w}x{final_h} (aligned to 28)")
        return image.resize((final_w, final_h), Image.Resampling.BICUBIC)
        
    return image

def get_grid_cell(image, grid_info, row_idx, col_idx):
    """
    获取指定网格单元格
    
    Args:
        image: 原始图像
        grid_info: 网格信息
        row_idx: 行索引 (0-based)
        col_idx: 列索引 (0-based)
        
    Returns:
        PIL Image: 裁剪后的单元格图像
        tuple: 单元格坐标 (x1, y1, x2, y2)
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image.copy()
    
    height, width = image_np.shape[:2]
    
    # 计算单元格边界
    cell_width = width // grid_info['cols']
    cell_height = height // grid_info['rows']
    
    x1 = col_idx * cell_width
    y1 = row_idx * cell_height
    x2 = min(x1 + cell_width, width)
    y2 = min(y1 + cell_height, height)
    
    # 裁剪图像
    cell_image = image_np[y1:y2, x1:x2]
    cell_pil = Image.fromarray(cell_image)
    
    return cell_pil, (x1, y1, x2, y2)

def resize_image(image, max_size=1024):
    """
    调整图像大小，保持宽高比
    
    Args:
        image: PIL Image
        max_size: 最大边长
        
    Returns:
        PIL Image: 调整后的图像
    """
    width, height = image.size
    
    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image

def ensure_rgb(image):
    """
    确保图像为RGB格式
    
    Args:
        image: PIL Image
        
    Returns:
        PIL Image: RGB格式图像
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image
import argparse
import sys
import torch
from pathlib import Path
from pipeline import MLLMSAMPipeline

def main():
    parser = argparse.ArgumentParser(description="MLLM + SAM3 图像分割系统")
    parser.add_argument("--image", type=str, required=True, 
                       help="输入图像路径")
    parser.add_argument("--text", type=str, required=True,
                       help="目标文本描述")
    parser.add_argument("--max_iter", type=int, default=5,
                       help="最大迭代次数 (默认: 5)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录 (默认: /root/autodl-tmp/mllm_sam_project/outputs)")
    
    args = parser.parse_args()
    
    # 检查图像是否存在
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"错误: 图像文件不存在: {args.image}")
        sys.exit(1)
    
    # 设置输出目录
    if args.output_dir:
        from config import Config
        Config.OUTPUT_DIR = Path(args.output_dir)
        Config.setup_dirs()
    
    try:
        # 显存监控初始化
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            print(f"初始显存: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")

        # 初始化并运行管道
        pipeline = MLLMSAMPipeline()
        result = pipeline.run(args.image, args.text, args.max_iter)

        # 打印显存使用情况
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"\n{'='*20} 显存使用报告 {'='*20}")
            print(f"峰值显存占用: {max_memory:.2f} MB ({max_memory/1024:.2f} GB)")
            print(f"当前显存占用: {current_memory:.2f} MB ({current_memory/1024:.2f} GB)")
            print("="*54)
        
        # 输出结果
        print("\n" + "="*50)
        if result["success"]:
            print("✓ 分割成功!")
            print(f"最佳分数: {result['best_score']:.3f}")
            print(f"最终图像: {result['final_image_path']}")
            print(f"Mask文件: {result['mask_path']}")
            print(f"总迭代次数: {result['iterations']}")
        else:
            print("✗ 分割失败")
            print(f"错误: {result.get('message', '未知错误')}")
        
        print("="*50)
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
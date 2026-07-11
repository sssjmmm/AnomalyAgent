#!/usr/bin/env python3
"""
自动生成缺失的mask图片

功能：
1. 扫描数据集目录，找出缺失的mask图片
2. 使用mask_gen.py工具根据ori原图和image异常图生成对应的mask
3. 将生成的mask保存到对应的mask路径中
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict
import traceback

# 导入mask_gen工具
try:
    from agentic_rl.tools.mask_gen import mask_gen, get_mask_gen_tool
    print("✓ 成功导入 mask_gen 工具")
except ImportError as e:
    print(f"✗ 无法导入 mask_gen 工具: {e}")
    raise RuntimeError("Run this script from the repository root after installing inference dependencies") from e


def find_missing_masks(dataset_path: str) -> List[Tuple[str, str, str]]:
    """
    找出所有缺失的mask图片
    
    Args:
        dataset_path: 数据集根目录路径
        
    Returns:
        List of (sample_name, anomaly_name, image_filename) tuples
    """
    missing_masks = []
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"✗ 数据集路径不存在: {dataset_path}")
        return missing_masks
    
    # 遍历所有sample_name目录
    for sample_dir in sorted(dataset_path.iterdir()):
        if not sample_dir.is_dir():
            continue
        
        sample_name = sample_dir.name
        
        # 遍历所有anomaly_name目录
        for anomaly_dir in sorted(sample_dir.iterdir()):
            if not anomaly_dir.is_dir():
                continue
            
            anomaly_name = anomaly_dir.name
            
            # 检查必要的目录是否存在
            ori_dir = anomaly_dir / "ori"
            image_dir = anomaly_dir / "image"
            mask_dir = anomaly_dir / "mask"
            
            if not ori_dir.exists():
                print(f"⚠ 警告: {ori_dir} 不存在，跳过")
                continue
            
            if not image_dir.exists():
                print(f"⚠ 警告: {image_dir} 不存在，跳过")
                continue
            
            # 确保mask目录存在
            mask_dir.mkdir(parents=True, exist_ok=True)
            
            # 获取image目录中的所有图片文件
            image_files = set()
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                image_files.update([f.name for f in image_dir.glob(f'*{ext}')])
            
            # 获取mask目录中已有的图片文件
            mask_files = set()
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                mask_files.update([f.name for f in mask_dir.glob(f'*{ext}')])
            
            # 找出缺失的mask
            missing_files = image_files - mask_files
            
            for filename in sorted(missing_files):
                missing_masks.append((sample_name, anomaly_name, filename))
    
    return missing_masks


def generate_mask_for_image(
    dataset_path: str,
    sample_name: str,
    anomaly_name: str,
    image_filename: str,
    threshold: float = 0.5
) -> Dict:
    """
    为单个图片生成mask
    
    Args:
        dataset_path: 数据集根目录路径
        sample_name: 物体类别名称
        anomaly_name: 异常类型名称
        image_filename: 图片文件名
        threshold: mask生成阈值
        
    Returns:
        生成结果的字典
    """
    dataset_path = Path(dataset_path)
    
    # 构建路径
    ori_path = dataset_path / sample_name / anomaly_name / "ori" / image_filename
    image_path = dataset_path / sample_name / anomaly_name / "image" / image_filename
    mask_path = dataset_path / sample_name / anomaly_name / "mask" / image_filename
    
    # 检查文件是否存在
    if not ori_path.exists():
        return {
            "status": "error",
            "error": f"ori文件不存在: {ori_path}",
            "mask_path": str(mask_path)
        }
    
    if not image_path.exists():
        return {
            "status": "error",
            "error": f"image文件不存在: {image_path}",
            "mask_path": str(mask_path)
        }
    
    # 如果mask已存在，跳过
    if mask_path.exists():
        return {
            "status": "skipped",
            "message": f"mask已存在: {mask_path}",
            "mask_path": str(mask_path)
        }
    
    try:
        # 调用mask_gen工具生成mask
        # 注意：mask_gen会自动根据anomaly_image路径生成mask路径
        # 但我们需要确保mask保存到正确的位置
        result = mask_gen(
            normal_image=str(ori_path),
            anomaly_image=str(image_path),
            threshold=threshold,
            original_image_path=str(image_path),
            output_base_dir=None  # 使用默认路径生成逻辑
        )
        
        # 检查生成的mask路径是否正确
        generated_mask_path = result.get("mask_url", "")
        if generated_mask_path and Path(generated_mask_path).exists():
            # 如果生成的路径与目标路径不同，移动文件
            if generated_mask_path != str(mask_path):
                import shutil
                shutil.move(generated_mask_path, mask_path)
                print(f"  → 移动mask: {generated_mask_path} -> {mask_path}")
            
            result["status"] = "success"
            result["mask_path"] = str(mask_path)
            return result
        else:
            return {
                "status": "error",
                "error": f"mask生成失败，未找到生成的文件",
                "mask_path": str(mask_path),
                "generated_path": generated_mask_path
            }
    
    except Exception as e:
        error_msg = f"生成mask时出错: {str(e)}"
        print(f"  ✗ {error_msg}")
        traceback.print_exc()
        return {
            "status": "error",
            "error": error_msg,
            "mask_path": str(mask_path)
        }


def main():
    parser = argparse.ArgumentParser(
        description="自动生成缺失的mask图片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 扫描整个数据集并生成所有缺失的mask
  python generate_missing_masks.py --dataset_path /path/to/mvtec_benchmark
  
  # 只处理特定物体类别
  python generate_missing_masks.py --dataset_path /path/to/mvtec_benchmark --sample_names screw toothbrush
  
  # 指定mask生成阈值
  python generate_missing_masks.py --dataset_path /path/to/mvtec_benchmark --threshold 0.6
        """
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="数据集根目录路径"
    )
    
    parser.add_argument(
        "--sample_names",
        type=str,
        nargs="+",
        default=None,
        help="指定要处理的物体类别（例如: screw toothbrush），如果不指定则处理所有类别"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="mask生成阈值（默认: 0.5）"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅扫描缺失的mask，不实际生成（用于预览）"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("自动生成缺失的mask图片")
    print("=" * 80)
    print(f"数据集路径: {args.dataset_path}")
    print(f"物体类别: {args.sample_names if args.sample_names else '全部'}")
    print(f"Mask阈值: {args.threshold}")
    print(f"模式: {'预览模式（不生成）' if args.dry_run else '生成模式'}")
    print("=" * 80)
    print()
    
    # 扫描缺失的mask
    print("正在扫描缺失的mask...")
    missing_masks = find_missing_masks(args.dataset_path)
    
    # 如果指定了sample_names，进行过滤
    if args.sample_names:
        missing_masks = [
            (s, a, f) for s, a, f in missing_masks
            if s in args.sample_names
        ]
    
    if not missing_masks:
        print("✓ 没有发现缺失的mask，所有图片都有对应的mask！")
        return
    
    print(f"发现 {len(missing_masks)} 个缺失的mask")
    print()
    
    # 按sample_name和anomaly_name分组统计
    stats = {}
    for sample_name, anomaly_name, filename in missing_masks:
        key = (sample_name, anomaly_name)
        stats[key] = stats.get(key, 0) + 1
    
    print("缺失mask统计:")
    for (sample_name, anomaly_name), count in sorted(stats.items()):
        print(f"  {sample_name}/{anomaly_name}: {count} 个")
    print()
    
    if args.dry_run:
        print("预览模式：以下mask将被生成（前10个）:")
        for i, (sample_name, anomaly_name, filename) in enumerate(missing_masks[:10]):
            print(f"  {i+1}. {sample_name}/{anomaly_name}/{filename}")
        if len(missing_masks) > 10:
            print(f"  ... 还有 {len(missing_masks) - 10} 个")
        return
    
    # 初始化mask生成工具（单例模式，只初始化一次）
    print("正在初始化mask生成工具...")
    try:
        tool = get_mask_gen_tool()
        print(f"✓ mask生成工具初始化完成，设备: {tool.device}")
        print()
    except Exception as e:
        print(f"✗ 初始化mask生成工具失败: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # 生成缺失的mask
    print("开始生成缺失的mask...")
    print()
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    # 使用tqdm显示进度
    for sample_name, anomaly_name, filename in tqdm(missing_masks, desc="生成mask"):
        result = generate_mask_for_image(
            dataset_path=args.dataset_path,
            sample_name=sample_name,
            anomaly_name=anomaly_name,
            image_filename=filename,
            threshold=args.threshold
        )
        
        status = result.get("status", "unknown")
        if status == "success":
            success_count += 1
        elif status == "skipped":
            skipped_count += 1
        else:
            error_count += 1
            # 显示错误信息（但不中断处理）
            error_msg = result.get("error", "未知错误")
            tqdm.write(f"✗ {sample_name}/{anomaly_name}/{filename}: {error_msg}")
    
    print()
    print("=" * 80)
    print("生成完成！")
    print("=" * 80)
    print(f"成功: {success_count} 个")
    print(f"跳过: {skipped_count} 个（已存在）")
    print(f"失败: {error_count} 个")
    print(f"总计: {len(missing_masks)} 个")
    print("=" * 80)


if __name__ == "__main__":
    main()

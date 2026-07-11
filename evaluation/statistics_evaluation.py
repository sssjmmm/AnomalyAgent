#!/usr/bin/env python3
"""
统计批处理日志中 evaluation_passed 的 true 和 false 数量
"""
import json
import argparse
from pathlib import Path


def statistics_evaluation(log_file: str):
    """
    统计批处理日志中 evaluation_passed 字段的分布
    
    Args:
        log_file: 批处理日志文件路径
    """
    # 读取JSON文件
    with open(log_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    total = data.get('total', len(results))
    
    # 统计 evaluation_passed 的值
    true_count = 0
    false_count = 0
    null_count = 0
    other_count = 0
    
    for result in results:
        evaluation_passed = result.get('evaluation_passed')
        if evaluation_passed is True:
            true_count += 1
        elif evaluation_passed is False:
            false_count += 1
        elif evaluation_passed is None:
            null_count += 1
        else:
            other_count += 1
    
    # 打印统计结果
    print("=" * 60)
    print(f"批处理日志统计: {Path(log_file).name}")
    print("=" * 60)
    print(f"总记录数: {total}")
    print(f"成功记录数: {data.get('success', 0)}")
    print(f"失败记录数: {data.get('failed', 0)}")
    print()
    print("evaluation_passed 统计:")
    print(f"  True  (评估通过): {true_count:4d} ({true_count/total*100:.2f}%)")
    print(f"  False (评估未通过): {false_count:4d} ({false_count/total*100:.2f}%)")
    if null_count > 0:
        print(f"  None  (未设置): {null_count:4d} ({null_count/total*100:.2f}%)")
    if other_count > 0:
        print(f"  其他值: {other_count:4d} ({other_count/total*100:.2f}%)")
    print("=" * 60)
    
    return {
        'total': total,
        'true': true_count,
        'false': false_count,
        'null': null_count,
        'other': other_count
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计批处理日志中 evaluation_passed 的分布")
    parser.add_argument(
        "log_file",
        type=str,
        help="批处理日志JSON文件路径"
    )
    
    args = parser.parse_args()
    
    if not Path(args.log_file).exists():
        print(f"错误: 文件不存在: {args.log_file}")
        exit(1)
    
    statistics_evaluation(args.log_file)


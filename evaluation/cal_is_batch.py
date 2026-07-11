import argparse
import subprocess
import csv
import os
import re
import ssl
from collections import defaultdict

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=""):
        return iterable

def parse_fidelity_output(output):
    """解析fidelity命令的输出，提取IS值"""
    is_mean = None
    is_std = None
    
    # 查找 Inception Score 行
    match = re.search(r'Inception Score: ([\d.]+) ± ([\d.]+)', output)
    if match:
        is_mean = float(match.group(1))
        is_std = float(match.group(2))
    
    # 如果上面的正则没匹配到，尝试查找其他格式
    if is_mean is None:
        match = re.search(r'inception_score_mean: ([\d.]+)', output)
        if match:
            is_mean = float(match.group(1))
    
    if is_std is None:
        match = re.search(r'inception_score_std: ([\d.]+)', output)
        if match:
            is_std = float(match.group(1))
    
    return is_mean, is_std

def find_fidelity_command():
    """查找fidelity命令的路径"""
    import shutil
    # 方法1: 使用which查找
    fidelity_path = shutil.which('fidelity')
    if fidelity_path:
        return fidelity_path
    
    # 方法2: 尝试常见的conda环境路径
    conda_base = os.environ.get('CONDA_PREFIX', '')
    if conda_base:
        conda_fidelity = os.path.join(conda_base, 'bin', 'fidelity')
        if os.path.exists(conda_fidelity):
            return conda_fidelity
    
    # 方法3: 尝试用户conda路径
    user_conda_base = os.environ.get('CONDA_BASE', '')
    env_name = os.environ.get('CONDA_DEFAULT_ENV', 'Anomalydiffusion')
    conda_fidelity = os.path.join(user_conda_base, 'envs', env_name, 'bin', 'fidelity')
    if os.path.exists(conda_fidelity):
        return conda_fidelity
    
    # 如果都找不到，返回None
    return None


def calculate_is_for_anomaly(gen_path, sample_name, anomaly_name, gpu=0):
    """计算单个物体+缺陷组合的IS值"""
    image_path = os.path.join(gen_path, sample_name, anomaly_name, 'image')
    # 转换为绝对路径
    image_path = os.path.abspath(image_path)
    
    # 检查路径是否存在
    if not os.path.exists(image_path):
        print(f"警告: 路径不存在 {image_path}")
        return None, None
    
    # 查找fidelity命令
    fidelity_cmd = find_fidelity_command()
    if not fidelity_cmd:
        print(f"错误: 找不到fidelity命令，请确保已安装fidelity工具")
        print(f"安装方法: pip install pytorch-fid 或 conda install -c conda-forge pytorch-fid")
        return None, None
    
    # 构建fidelity命令
    cmd = [
        fidelity_cmd,
        '--gpu', str(gpu),
        '--isc',
        '--input1', image_path
    ]
    
    try:
        # 执行命令并捕获输出
        # 确保使用当前环境的环境变量（包括PATH）
        env = os.environ.copy()
        # 设置环境变量来禁用SSL验证（fidelity作为subprocess运行，需要环境变量）
        # 注意：权重文件应该已经在缓存中，但fidelity仍会尝试验证下载
        env['PYTHONHTTPSVERIFY'] = '0'
        env['CURL_CA_BUNDLE'] = ''
        env['REQUESTS_CA_BUNDLE'] = ''
        
        # 创建一个包装脚本来禁用SSL验证
        # 由于fidelity是独立的Python脚本，我们需要通过环境变量或修改fidelity脚本
        # 最简单的方法是在subprocess中设置环境变量，但这可能不够
        # 更好的方法是创建一个包装脚本或直接修改fidelity的调用方式
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env
        )
        
        # 解析输出
        is_mean, is_std = parse_fidelity_output(result.stdout)
        return is_mean, is_std
    
    except subprocess.CalledProcessError as e:
        print(f"错误: 计算 {sample_name}+{anomaly_name} 的IS值时出错")
        print(f"命令: {' '.join(cmd)}")
        print(f"错误输出: {e.stderr}")
        return None, None
    except Exception as e:
        print(f"错误: 处理 {sample_name}+{anomaly_name} 时出现异常: {str(e)}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='批量计算IS值')
    parser.add_argument(
        "--gen_path",
        type=str,
        default="./output",
        help="生成数据集的路径"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="使用的GPU编号"
    )
    parser.add_argument(
        "--name_file",
        type=str,
        default="name-anomaly.txt",
        help="包含物体+缺陷名称的文件"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_is.csv",
        help="输出CSV文件路径"
    )
    
    args = parser.parse_args()
    
    # 在开始时检查fidelity命令是否可用
    fidelity_cmd = find_fidelity_command()
    if not fidelity_cmd:
        print("错误: 找不到fidelity命令")
        print("请确保:")
        print("  1. 已激活正确的conda环境（包含fidelity工具）")
        print("  2. 或已安装fidelity工具: pip install pytorch-fid")
        print("  3. 或fidelity工具在PATH中")
        return 1
    
    print(f"使用fidelity命令: {fidelity_cmd}")
    print()
    
    # 将gen_path转换为绝对路径
    args.gen_path = os.path.abspath(args.gen_path)
    
    # 读取name-anomaly.txt文件
    name_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.name_file)
    if not os.path.exists(name_file_path):
        print(f"错误: 找不到文件 {name_file_path}")
        return
    
    # 解析所有物体+缺陷组合
    anomaly_list = []
    with open(name_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and '+' in line:
                parts = line.split('+')
                if len(parts) == 2:
                    sample_name = parts[0]
                    anomaly_name = parts[1]
                    anomaly_list.append((sample_name, anomaly_name))
    
    print(f"找到 {len(anomaly_list)} 个物体+缺陷组合")
    
    # 按物体类别分组
    results_by_sample = defaultdict(list)
    
    # 计算每个组合的IS值
    if HAS_TQDM:
        iterator = tqdm(anomaly_list, desc="计算IS值")
    else:
        iterator = anomaly_list
        print(f"开始计算 {len(anomaly_list)} 个组合的IS值...")
    
    for sample_name, anomaly_name in iterator:
        is_mean, is_std = calculate_is_for_anomaly(
            args.gen_path, 
            sample_name, 
            anomaly_name, 
            args.gpu
        )
        
        if is_mean is not None:
            results_by_sample[sample_name].append({
                'anomaly': anomaly_name,
                'is_mean': is_mean,
                'is_std': is_std
            })
            print(f"{sample_name}+{anomaly_name}: IS={is_mean:.6f} ± {is_std:.6f}")
        else:
            print(f"{sample_name}+{anomaly_name}: 计算失败")
    
    # 写入CSV文件
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['sample_name', 'anomaly_name', 'is_mean', 'is_std'])
        
        # 写入每个组合的详细结果
        for sample_name in sorted(results_by_sample.keys()):
            for result in results_by_sample[sample_name]:
                writer.writerow([
                    sample_name,
                    result['anomaly'],
                    f"{result['is_mean']:.6f}",
                    f"{result['is_std']:.6f}"
                ])
    
    # 计算每个物体类别的平均IS值
    print("\n" + "="*50)
    print("各物体类别的平均IS值:")
    print("="*50)
    
    summary_output = args.output.replace('.csv', '_summary.csv')
    with open(summary_output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sample_name', 'avg_is_mean', 'avg_is_std', 'count'])
        
        for sample_name in sorted(results_by_sample.keys()):
            results = results_by_sample[sample_name]
            if results:
                avg_is_mean = sum(r['is_mean'] for r in results) / len(results)
                avg_is_std = sum(r['is_std'] for r in results) / len(results)
                count = len(results)
                
                writer.writerow([
                    sample_name,
                    f"{avg_is_mean:.6f}",
                    f"{avg_is_std:.6f}",
                    count
                ])
                
                print(f"{sample_name}: 平均IS={avg_is_mean:.6f} ± {avg_is_std:.6f} (共{count}个缺陷)")
    
    print(f"\n详细结果已保存到: {args.output}")
    print(f"汇总结果已保存到: {summary_output}")

if __name__ == '__main__':
    main()

# Example: python cal_is_batch.py --gen_path /path/to/generated --gpu 0 --name_file name-anomaly.txt --output results.csv

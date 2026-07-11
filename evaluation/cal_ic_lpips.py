import argparse
import random
import torch
import torch.nn as nn
from torchvision import utils
from tqdm import tqdm
import sys
import ssl
import lpips
from torchvision import transforms, utils
from torch.utils import data
import os
from PIL import Image
import numpy as np
from datetime import datetime

# 临时禁用SSL验证，避免下载VGG16模型时的SSL证书问题
# 注意：模型文件应该已经在缓存中，但LPIPS仍会尝试验证下载
ssl._create_default_https_context = ssl._create_unverified_context

def set_seed(seed=42):
    """设置随机种子以确保结果可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保CUDA操作的确定性（可能会降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

lpips_fn = lpips.LPIPS(net='vgg').cuda()
preprocess = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
device='cuda'
def ic_lpips(mvtec_path,gen_path,sample_name,anomaly_name): # ../dataset/mvtec_data, ../generated_data, 15种物体，73种缺陷
    print(sample_name,anomaly_name) # capsule, crack
    tar_path = '%s/%s/%s/image' % (gen_path,sample_name, anomaly_name) # ../generated_data/capsule/crack
    ori_path='%s/%s/test/%s'%(mvtec_path,sample_name,anomaly_name) # ../dataset/mvtec_data/capsule/test/crack (测试集中有缺陷的图片)
    with torch.no_grad():
        l = len(os.listdir(ori_path)) // 3 # 为什么要除以3
        avg_dist = torch.zeros([l, ]) # 初始化一个长度为l的零张量
        files_list=os.listdir(tar_path) # ../generated_data/capsule/crack下的所有文件列表
        input_tensors1=[]
        clusters=[[] for i in range(l)] # 聚类的索引列表，每个索引对应一个聚类，每个聚类对应一个列表，列表中存储的是每个聚类中图片的索引
        for k in range(l):
            input1_path = os.path.join(ori_path, '%03d.png' % k) # ../dataset/mvtec_data/capsule/test/crack/000.png
            input_image1 = Image.open(input1_path).convert('RGB') # 打开图片并转换为RGB格式
            input_tensor1 = preprocess(input_image1)
            input_tensor1 = input_tensor1.to(device) # 将图片转换为张量并移动到GPU
            input_tensors1.append(input_tensor1) # 将图片张量添加到列表中
        for i in range(len(files_list)):
            try:
                min_dist = 999999999 # 初始化最小距离为999999999
                input2_path = os.path.join(tar_path, files_list[i]) # ../generated_data/capsule/crack/000.png
                input_image2 = Image.open(input2_path).convert('RGB') # 打开图片并转换为RGB格式
                input_tensor2 = preprocess(input_image2)
                input_tensor2 = input_tensor2.to(device) # 将图片转换为张量并移动到GPU
                for k in range(l): # 相当于归类，将图片归类到距离最近的聚类（Mvtec决定）中
                    dist = lpips_fn(input_tensors1[k], input_tensor2) # 计算两个图片的LPIPS距离
                    if dist <= min_dist:
                        max_ind = k 
                        min_dist = dist
                clusters[max_ind].append(input2_path) # 将图片添加到对应的聚类中
                
                # 每处理100个文件打印一次进度
                if (i + 1) % 100 == 0:
                    print(f"  已归类: {i+1}/{len(files_list)} ({100*(i+1)//len(files_list)}%)")
            except Exception as e:
                print(f"  错误: 处理文件 {files_list[i]} 时出错: {e}")
                continue
        # 此时503张图片已经归好类
        cluster_size=50
        for k in range(l):
            print(f"处理聚类 {k+1}/{l} (共有 {len(clusters[k])} 个文件)")
            files_list=clusters[k]
            random.shuffle(files_list)
            files_list = files_list[:cluster_size]
            
            if len(files_list) < 2:
                print(f"  警告: 聚类 {k} 只有 {len(files_list)} 个文件，跳过")
                avg_dist[k] = 0.0
                continue
            
            total_pairs = len(files_list) * (len(files_list) - 1) // 2
            print(f"  需要计算 {total_pairs} 对图片的LPIPS距离...")
            
            dists = []
            pair_count = 0
            for i in range(len(files_list)):
                for j in range(i + 1, len(files_list)):
                    try:
                        input1_path = files_list[i]
                        input2_path = files_list[j]

                        input_image1 = Image.open(input1_path).convert('RGB')
                        input_image2 = Image.open(input2_path).convert('RGB')

                        input_tensor1 = preprocess(input_image1)
                        input_tensor2 = preprocess(input_image2)

                        input_tensor1 = input_tensor1.to(device)
                        input_tensor2 = input_tensor2.to(device)

                        dist = lpips_fn(input_tensor1, input_tensor2)

                        dists.append(dist.item() if isinstance(dist, torch.Tensor) else dist)
                        
                        pair_count += 1
                        # 每计算100对打印一次进度，防止SSH超时
                        if pair_count % 100 == 0:
                            print(f"    进度: {pair_count}/{total_pairs} ({pair_count*100//total_pairs}%)")
                        
                        # 定期清理GPU缓存
                        if pair_count % 500 == 0:
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        print(f"    错误: 处理 {input1_path} 和 {input2_path} 时出错: {e}")
                        continue
            
            if len(dists) > 0:
                dists = torch.tensor(dists)
                avg_dist[k] = dists.mean()
                print(f"  聚类 {k+1} 完成: 平均距离 = {avg_dist[k].item():.6f}")
            else:
                print(f"  警告: 聚类 {k} 没有成功计算的距离值")
                avg_dist[k] = 0.0
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
        return avg_dist[~torch.isnan(avg_dist)].mean()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mvtec_path",
        required=True,
        help="path ot mvtec dataset",
    )
    parser.add_argument(
        "--gen_path",
        required=True,
        help="path to your generated dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="结果保存目录（默认: 当前工作目录）",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="结果文件名（不含扩展名），如果不指定，默认使用 'results_ic_lpips'。最终文件名格式: <output_name>_<timestamp>.csv",
    )
    parser.add_argument(
        "--sample_names",
        nargs='+',
        type=str,
        default=None,
        help="要处理的物体类别列表（可指定多个，例如: --sample_names toothbrush screw grid）。如果不指定，默认处理所有可用类别",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于确保结果可重复（默认: 42）",
    )
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    print(f"随机种子已设置为: {args.seed}")
    
    # 所有可用的物体类别
    all_sample_names = [
        'capsule',
        'bottle',
        'carpet',
        'leather',
        'pill',
        'transistor',
        'tile',
        'cable',
        'zipper',
        'toothbrush',
        'metal_nut',
        'hazelnut',
        'screw',
        'grid',
        'wood'
    ]
    
    # 确定要处理的样本名称
    if args.sample_names is not None:
        # 验证指定的样本名称是否在可用列表中
        invalid_names = [name for name in args.sample_names if name not in all_sample_names]
        if invalid_names:
            print(f"警告: 以下样本名称不在可用列表中，将被忽略: {invalid_names}")
        sample_names = [name for name in args.sample_names if name in all_sample_names]
        if not sample_names:
            print(f"错误: 没有有效的样本名称！可用列表: {all_sample_names}")
            exit(1)
    else:
        # 如果没有指定，使用所有可用类别
        sample_names = all_sample_names
    
    print(f"将处理 {len(sample_names)} 种物体: {sample_names}")
    # print(f"每种物体将处理前 {args.num_anomalies} 类异常")
    print("="*50)
    
    import csv
    from pathlib import Path
    
    # 确定输出目录
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path.cwd()
    
    # 确定输出文件名
    if args.output_name is not None:
        output_name = args.output_name
    else:
        output_name = "results_ic_lpips"
    
    # 生成带时间戳的文件名：<output_name>_<timestamp>.csv
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"{output_name}_{timestamp}.csv"
    for sample_name in tqdm(sample_names, desc="处理样本类别"):
        dis=0
        cnt=0
        # 获取该物体的所有异常类型
        anomaly_list = os.listdir('%s/%s'%(args.gen_path,sample_name))
        # 只取前num_anomalies个异常类型
        # anomaly_list = sorted(anomaly_list)[:args.num_anomalies]
        
        print(f"\n处理物体: {sample_name}, 异常类型: {anomaly_list}")
        
        for anomaly_name in anomaly_list:
            dis+=ic_lpips(args.mvtec_path,args.gen_path,sample_name,anomaly_name)
            cnt+=1
        
        if cnt > 0:
            with open(results_file, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                # 如果是第一次写入，写入表头
                if results_file.stat().st_size == 0:
                    writer.writerow(['sample_name', 'avg_ic_lpips'])
                writer.writerow([sample_name, str(float(dis/cnt))])
            print(f"{sample_name}: 平均IC-LPIPS = {dis/cnt:.6f}")
        else:
            print(f"警告: {sample_name} 没有找到异常类型")
    
    print("\n" + "="*50)
    print(f"\n结果已保存到: {results_file}")
    print(f"输出目录: {output_dir}")
# Example: python cal_ic_lpips.py --mvtec_path /path/to/mvtec --gen_path /path/to/generated

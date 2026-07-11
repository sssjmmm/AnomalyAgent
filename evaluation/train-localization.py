
import torch
from torch import optim
from unet_utils.tensorboard_visualizer import TensorboardVisualizer
from unet_utils.loss import FocalLoss, SSIM
import os
from unet_utils.data_loader import MVTec_Anomaly_Detection,MVTecDRAEMTestDataset_partial
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from unet_utils.model_unet import DiscriminativeSubNetwork
import os
from unet_utils.au_pro_util import calculate_au_pro
import multiprocessing
import subprocess
import time
from multiprocessing import Process, Queue, Lock
def get_available_gpu(min_free_memory_gb: float = 10.0):
    """
    自动选择空闲的 GPU（使用 nvidia-smi 获取真实的 GPU 使用情况）
    
    Args:
        min_free_memory_gb: 最小可用显存要求（GB），默认 10.0 GB
    
    Returns:
        GPU索引 (int)，如果没有可用GPU则返回None
    """
    if not torch.cuda.is_available():
        print("[GPU] CUDA 不可用")
        return None
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("[GPU] 未检测到 GPU")
        return None
    
    # 尝试使用 nvidia-smi 获取真实的 GPU 使用情况
    use_smi = False
    smi_data = {}
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            use_smi = True
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 3:
                        gpu_idx = int(parts[0].strip())
                        memory_used_mb = float(parts[1].strip())
                        memory_total_mb = float(parts[2].strip())
                        smi_data[gpu_idx] = {
                            'used': memory_used_mb / 1024,  # MB to GB
                            'total': memory_total_mb / 1024  # MB to GB
                        }
    except Exception as e:
        use_smi = False
    
    # 检查每个 GPU 的内存使用情况
    best_gpu = None
    max_free_memory = 0.0
    
    for i in range(num_gpus):
        if use_smi and i in smi_data:
            memory_total = smi_data[i]['total']
            memory_used = smi_data[i]['used']
            memory_free = memory_total - memory_used
            memory_used_ratio = memory_used / memory_total if memory_total > 0 else 0.0
        else:
            # 回退到 PyTorch 方法
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            memory_used = memory_allocated + memory_reserved
            memory_free = memory_total - memory_used
            memory_used_ratio = memory_used / memory_total
        
        # 选择可用显存最多且满足最小要求的 GPU
        if memory_free >= min_free_memory_gb and memory_free > max_free_memory:
            max_free_memory = memory_free
            best_gpu = i
    
    if best_gpu is None:
        # 如果没有满足最小要求的 GPU，选择可用显存最多的
        for i in range(num_gpus):
            if use_smi and i in smi_data:
                memory_total = smi_data[i]['total']
                memory_used = smi_data[i]['used']
                memory_free = memory_total - memory_used
            else:
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                memory_free = memory_total - (memory_allocated + memory_reserved)
            
            if memory_free > max_free_memory:
                max_free_memory = memory_free
                best_gpu = i
    
    return best_gpu

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
def test(args,obj_name, model_seg):
    mvtec_path = args.mvtec_path
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []
    img_dim = 256
    model_seg.eval()
    dataset = MVTecDRAEMTestDataset_partial(mvtec_path +'/'+ obj_name + "/test/", resize_shape=[img_dim, img_dim])
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    mask_cnt = 0

    anomaly_score_gt = []
    anomaly_score_prediction = []

    gt_masks=[]
    predicted_masks=[]

    for i_batch, sample_batched in enumerate(dataloader):

        gray_batch = sample_batched["image"].cuda()
        gray_batch=gray_batch[:,[2,1,0],:,:]

        is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
        anomaly_score_gt.append(is_normal)
        true_mask = sample_batched["mask"]
        true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))
        out_mask = model_seg(gray_batch)
        out_mask_sm = torch.softmax(out_mask, dim=1)

        out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()
        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                           padding=21 // 2).cpu().detach().numpy()
        image_score = np.max(out_mask_averaged)
        anomaly_score_prediction.append(image_score)

        flat_true_mask = true_mask_cv.flatten()
        flat_out_mask = out_mask_cv.flatten()
        gt_masks.append(true_mask_cv.squeeze())
        predicted_masks.append(out_mask_cv.squeeze())

        total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
        total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
        mask_cnt += 1

    anomaly_score_prediction = np.array(anomaly_score_prediction)
    anomaly_score_gt = np.array(anomaly_score_gt)
    auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
    ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

    total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
    total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
    total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
    auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
    pro_pixel, _ = calculate_au_pro(gt_masks, predicted_masks)
    obj_ap_pixel_list.append(ap_pixel)
    obj_auroc_pixel_list.append(auroc_pixel)
    obj_auroc_image_list.append(auroc)
    obj_ap_image_list.append(ap)
    print(obj_name)
    print("AUC Image:  " +str(auroc))
    print("AP Image:  " +str(ap))
    print("AUC Pixel:  " +str(auroc_pixel))
    #print("AUC Pixel:  " +str(auroc_pixel))
    print("AP Pixel:  " +str(ap_pixel))
    print('PRO Pixel:' +str(pro_pixel))
    print("==============================")
    return float(auroc),float(auroc_pixel),float(ap_pixel),float(pro_pixel)


def train_on_device(obj_names, args, gpu_id=None):
    """
    在指定设备上训练模型
    
    Args:
        obj_names: 对象名称列表
        args: 训练参数
        gpu_id: 指定的GPU ID（如果为None，会自动选择）
    """
    # 如果指定了GPU，使用指定的；否则自动选择
    if gpu_id is None:
        gpu_id = get_available_gpu(min_free_memory_gb=10.0)
        if gpu_id is None:
            print(f"[GPU] 警告: 无法找到可用GPU，使用默认GPU 0")
            gpu_id = 0
    
    print(f"[训练任务] 使用 GPU {gpu_id}")
    
    # 设置CUDA设备
    torch.cuda.set_device(gpu_id)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    for obj_name in obj_names:

        run_name = obj_name

        model_seg = DiscriminativeSubNetwork(in_channels=3, out_channels=2)
        model_seg.cuda(device=gpu_id)
        model_seg.apply(weights_init)

        optimizer = torch.optim.Adam([
                                      {"params": model_seg.parameters(), "lr": args.lr}])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)

        loss_focal = FocalLoss()

        dataset = MVTec_Anomaly_Detection(args,obj_name,length=500)
        dataloader = DataLoader(dataset, batch_size=args.bs,
                                shuffle=True, num_workers=16)

        n_iter = 0
        last_sum=0
        for epoch in range(args.epochs):
            model_seg.train()
            print(f"[GPU {gpu_id}] Epoch: "+str(epoch))
            for i_batch, sample_batched in enumerate(dataloader):
                aug_gray_batch = sample_batched["image"].cuda(device=gpu_id)
                anomaly_mask = sample_batched["mask"].cuda(device=gpu_id)
                out_mask = model_seg(aug_gray_batch)
                out_mask_sm = torch.softmax(out_mask, dim=1)
                segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                loss = segment_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_iter +=1
            scheduler.step()

            auroc,auroc_px,ap_px,pro_px=test(args,obj_name, model_seg)
            sum_metric=auroc+auroc_px+ap_px+pro_px
            if sum_metric>last_sum:
                torch.save(model_seg.state_dict(), os.path.join(args.save_path, run_name + ".pckl"))
                last_sum=sum_metric
    
    print(f"[训练任务完成] GPU {gpu_id} - {args.mask_dir}")


def train_single_threshold(threshold, base_args, obj_batch, gpu_lock):
    """
    在单独的进程中训练单个阈值
    
    Args:
        threshold: 阈值（0.1, 0.3, 0.5, 0.7, 0.9）
        base_args: 基础参数对象
        obj_batch: 对象名称列表
        gpu_lock: GPU锁（用于同步GPU选择）
    """
    import copy
    
    # 复制参数，避免多进程间的参数冲突
    args = copy.deepcopy(base_args)
    args.mask_dir = f"mask_{threshold}"
    
    # 根据save_path基础路径构建新的保存路径
    base_save_path = getattr(base_args, 'base_save_path', 'checkpoints/localization')
    args.save_path = os.path.join(base_save_path, f"mask_{threshold}")
    
    # 使用锁来确保GPU选择的原子性
    with gpu_lock:
        gpu_id = get_available_gpu(min_free_memory_gb=10.0)
        if gpu_id is None:
            print(f"[阈值 {threshold}] 警告: 无法找到可用GPU，使用默认GPU 0")
            gpu_id = 0
        time.sleep(0.5)  # 短暂延迟，避免同时选择同一个GPU
    
    print(f"[阈值 {threshold}] 开始训练，使用 GPU {gpu_id}, mask_dir={args.mask_dir}, save_path={args.save_path}")
    
    try:
        train_on_device(obj_batch, args, gpu_id=gpu_id)
        print(f"[阈值 {threshold}] 训练完成")
    except Exception as e:
        print(f"[阈值 {threshold}] 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_name', type=str, default='all')
    parser.add_argument('--generated_data_path', action='store', type=str, required=True)
    parser.add_argument('--save_path', default='checkpoints/localization', type=str)
    parser.add_argument('--mvtec_path', action='store', type=str, required=True)
    parser.add_argument('--bs', action='store', type=int,default=8, required=False)
    parser.add_argument('--lr', action='store', type=float,default=0.0001, required=False)
    parser.add_argument('--epochs', action='store', type=int,default=200, required=False)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--log_path', action='store', type=str,default='./logs/ ', required=False)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--test_separately', action='store_true',default=False)
    parser.add_argument('--reverse', action='store_true',default=False)
    parser.add_argument('--data_name',type=str, default='text_inversion')
    parser.add_argument('--mask_dir', type=str, default='mask', help='Mask directory name (e.g., mask_0.1, mask_0.3, mask_0.5, mask_0.7, mask_0.9), default: mask')
    parser.add_argument('--train_all_thresholds', action='store_true', help='是否并发训练所有阈值 (0.1, 0.3, 0.5, 0.7, 0.9)')
    parser.add_argument('--base_save_path', type=str, default='checkpoints/localization', help='并发训练时的基础保存路径')
    args = parser.parse_args()

    obj_batch =  [
                    'bottle',
                    'capsule',
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
    if args.reverse:
        obj_batch=reversed(obj_batch)
    if args.sample_name!='all':
        # 支持空格分隔的多个物体名称
        # 例如: "bottle hazelnut leather metal_nut" -> ["bottle", "hazelnut", "leather", "metal_nut"]
        obj_list = [name.strip() for name in args.sample_name.split() if name.strip()]
        picked_classes = obj_list
    else:
        picked_classes = obj_batch

    # 如果启用并发训练所有阈值
    if args.train_all_thresholds:
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        print("="*60)
        print("并发训练所有阈值模式")
        print(f"阈值列表: {thresholds}")
        print(f"基础保存路径: {args.base_save_path}")
        print("="*60 + "\n")
        
        # 保存基础save_path到args，供子进程使用
        args.base_save_path = args.base_save_path
        
        # 创建GPU锁（用于多进程间的GPU选择同步）
        gpu_lock = multiprocessing.Lock()
        
        # 创建进程列表
        processes = []
        
        # 为每个阈值创建训练进程
        for threshold in thresholds:
            p = Process(target=train_single_threshold, args=(threshold, args, picked_classes, gpu_lock))
            p.start()
            processes.append(p)
            # 短暂延迟，避免同时选择GPU时冲突
            time.sleep(1)
        
        # 等待所有进程完成
        print(f"\n等待 {len(processes)} 个训练进程完成...\n")
        for i, p in enumerate(processes):
            p.join()
            print(f"进程 {i+1}/{len(processes)} 完成")
        
        print("\n" + "="*60)
        print("所有阈值训练完成！")
        print("="*60)
    else:
        # 单阈值训练模式（原有逻辑）
        with torch.cuda.device(args.gpu_id):
            train_on_device(picked_classes, args, gpu_id=args.gpu_id)
#python train-unet.py --data_path $path_to_the_generated_data  --save_path ./ --mvtec_path=$path_to_mvtec --sample_name=capsule
# python train-localization.py \
#     --generated_data_path /path/to/generated/toothbrush/defective \
#     --mvtec_path /path/to/your/mvtec/dataset \
#     --train_all_thresholds \
#     --base_save_path checkpoints/localization_toothbrush


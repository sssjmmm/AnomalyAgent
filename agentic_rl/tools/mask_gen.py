"""
掩码生成工具类
使用MetaUAS模型生成异常图像和原图之间的差异mask
基于 workflow/tools/mask_generation_tool.py 的实现
"""
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Union, Any
import numpy as np
import torch
from PIL import Image

# 添加项目根目录到路径，以便导入workflow模块
current_dir = Path(__file__).resolve().parent
agent_dir = current_dir.parent
project_root = agent_dir.parent
third_party_dir = project_root / "third_party"

# 添加路径以便导入
for path in [str(project_root), str(third_party_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# 导入metauas模块
try:
    from third_party.metauas import (
        MetaUAS,
        set_random_seed,
        safely_load_state_dict,
        read_image_as_tensor,
        normalize,
        apply_ad_scoremap,
    )
except ImportError as e:
    print(f"警告: 无法导入 metauas 模块: {e}")
    print("请确保 third_party/metauas.py 存在且可访问")
    # 设置占位符以避免后续错误
    MetaUAS = None
    set_random_seed = None
    safely_load_state_dict = None
    read_image_as_tensor = None
    normalize = None
    apply_ad_scoremap = None

try:
    import kornia as K
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    print("警告: kornia未安装，将使用PIL进行图像resize")

config = None


def get_available_gpu(min_free_memory_gb: float = 10.0):
    """
    自动选择空闲的 GPU（使用 nvidia-smi 获取真实的 GPU 使用情况）
    
    Args:
        min_free_memory_gb: 最小可用显存要求（GB），默认 10.0 GB
    
    Returns:
        设备字符串，如 "cuda:1" 或 "cpu"
    """
    if not torch.cuda.is_available():
        print("[GPU] CUDA 不可用，使用 CPU")
        return "cpu"
    
    # 获取所有 GPU 数量
    num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        print("[GPU] 未检测到 GPU，使用 CPU")
        return "cpu"
    
    # 尝试使用 nvidia-smi 获取真实的 GPU 使用情况
    use_smi = False
    smi_data = {}
    
    try:
        import subprocess
        # 查询 GPU 索引、已用内存、总内存
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
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as e:
        print(f"[GPU] 无法使用 nvidia-smi，将使用 PyTorch 方法（可能不准确）: {e}")
        use_smi = False
    
    # 检查每个 GPU 的内存使用情况
    best_gpu = None
    max_free_memory = 0.0
    gpu_info = []
    
    print(f"[GPU] 检测到 {num_gpus} 个 GPU，正在检查可用显存...")
    
    for i in range(num_gpus):
        if use_smi and i in smi_data:
            # 使用 nvidia-smi 获取的真实数据
            memory_total = smi_data[i]['total']
            memory_used = smi_data[i]['used']
            memory_free = memory_total - memory_used
            memory_used_ratio = memory_used / memory_total if memory_total > 0 else 0.0
        else:
            # 回退到 PyTorch 方法（只能看到当前进程的内存）
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            memory_used = memory_allocated + memory_reserved
            memory_free = memory_total - memory_used
            memory_used_ratio = memory_used / memory_total
        
        gpu_info.append({
            'id': i,
            'total': memory_total,
            'used': memory_used,
            'free': memory_free,
            'used_ratio': memory_used_ratio
        })
        
        print(f"  GPU {i}: {memory_used_ratio*100:.1f}% 已使用 "
              f"({memory_used:.2f}GB / {memory_total:.2f}GB), "
              f"可用: {memory_free:.2f}GB")
        
        # 选择可用显存最多且满足最小要求的 GPU
        if memory_free >= min_free_memory_gb and memory_free > max_free_memory:
            max_free_memory = memory_free
            best_gpu = i
    
    if best_gpu is None:
        # 如果没有满足最小要求的 GPU，选择可用显存最多的
        print(f"[GPU] 警告: 没有 GPU 满足最小显存要求 ({min_free_memory_gb}GB)，选择可用显存最多的 GPU")
        for info in gpu_info:
            if info['free'] > max_free_memory:
                max_free_memory = info['free']
                best_gpu = info['id']
    
    if best_gpu is not None:
        selected_info = gpu_info[best_gpu]
        print(f"[GPU] 选择 GPU {best_gpu}: 可用显存 {selected_info['free']:.2f}GB "
              f"({(1-selected_info['used_ratio'])*100:.1f}% 空闲)")
        return f"cuda:{best_gpu}"
    else:
        print("[GPU] 所有 GPU 显存不足，使用 CPU")
        return "cpu"


class MaskGenTool:
    """
    MetaUAS-based anomaly mask generation tool
    基于MetaUAS模型生成异常mask
    
    Input: 正常图像 + 异常图像
    Output: 异常mask
    """
    
    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        img_size: int = 256,
        device: str = "cuda",
        config_dict: Optional[dict] = None,
        auto_select_gpu: bool = True,
        min_free_memory_gb: float = 10.0
    ):
        """
        初始化Mask生成工具
        
        Args:
            ckpt_path: 模型权重文件路径（如果为None，使用配置中的路径）
            img_size: 图像输入尺寸（256或512）
            device: 设备 ('cuda', 'cpu', 或 'cuda:0'等具体设备)
                    如果为 'cuda' 且 auto_select_gpu=True，将自动选择最佳GPU
            config_dict: 自定义配置字典
            auto_select_gpu: 是否自动选择最佳GPU（当device='cuda'时生效）
            min_free_memory_gb: 自动选择GPU时的最小可用显存要求（GB）
        """
        # 使用配置或默认值
        if config_dict is None and config is not None:
            self.config = config.MASK_GENERATION_CONFIG
        elif config_dict is not None:
            self.config = config_dict
        else:
            # 默认配置
            self.config = {
                "model_path": os.getenv("METAUAS_CKPT", "models/metauas-256.ckpt"),
                "img_size": 256,
                "default_threshold": 0.5,
                "output_dir": "./masks"
            }
        
        # 设置随机种子
        if set_random_seed is not None:
            set_random_seed(1)
        else:
            raise RuntimeError("metauas 模块未正确导入，无法初始化模型")
        
        # 设备配置
        if device == "cuda" and auto_select_gpu and torch.cuda.is_available():
            # 自动选择最佳GPU
            selected_device = get_available_gpu(min_free_memory_gb=min_free_memory_gb)
            self.device = torch.device(selected_device)
        else:
            # 使用指定的设备
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.img_size = img_size
        
        # 模型路径（优先使用传入参数，否则使用配置）
        if ckpt_path is None:
            ckpt_path = self.config.get("model_path", "models/metauas-256.ckpt")
        
        # 尝试解析相对路径
        if not os.path.isabs(ckpt_path):
            # 尝试多个可能的路径
            possible_paths = [
                ckpt_path,  # 相对当前工作目录
                project_root / ckpt_path,  # 相对项目根目录
                project_root / "models" / "metauas-256.ckpt",  # project_root/models/
            ]
            
            for path in possible_paths:
                path_str = str(path) if isinstance(path, Path) else path
                if os.path.exists(path_str):
                    ckpt_path = path_str
                    break
        
        self.ckpt_path = ckpt_path
        
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(
                f"模型文件不存在: {self.ckpt_path}\n"
                "请确保模型文件路径正确"
            )
        
        # 模型架构配置（必须与checkpoint匹配）
        encoder = 'efficientnet-b4'
        decoder = 'unet'
        encoder_depth = 5
        decoder_depth = 5
        num_crossfa_layers = 3
        alignment_type = 'sa'
        fusion_policy = 'cat'
        
        print(f"[Mask Gen Tool] 正在加载MetaUAS模型: {self.ckpt_path}")
        
        # 创建模型
        if MetaUAS is None:
            raise RuntimeError("MetaUAS 类未正确导入，请检查 metauas 模块")
        
        self.model = MetaUAS(
            encoder, 
            decoder, 
            encoder_depth, 
            decoder_depth, 
            num_crossfa_layers, 
            alignment_type, 
            fusion_policy
        )
        
        # 加载权重
        if safely_load_state_dict is None:
            raise RuntimeError("safely_load_state_dict 函数未正确导入")
        self.model = safely_load_state_dict(self.model, self.ckpt_path)
        
        # 移到设备并设置为评估模式
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[Mask Gen Tool] 模型加载完成，设备: {self.device}")
    
    def _load_image(self, image_input: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        加载图像并转换为tensor
        
        Args:
            image_input: 图像路径、PIL Image或tensor
            
        Returns:
            tensor: [1, 3, H, W]，值域[0, 1]
        """
        if isinstance(image_input, torch.Tensor):
            # 如果已经是tensor，确保格式正确
            if image_input.dim() == 3:
                image_input = image_input.unsqueeze(0)
            if image_input.shape[1] != 3:
                raise ValueError(f"Tensor通道数必须是3，当前为: {image_input.shape[1]}")
            return image_input.to(self.device)
        elif isinstance(image_input, Image.Image):
            # PIL Image -> tensor
            pil_image = image_input.convert('RGB')
            return self._read_image_as_tensor_from_pil(pil_image)
        elif isinstance(image_input, str):
            # 文件路径 -> tensor
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"图像文件不存在: {image_input}")
            
            # 添加重试机制，处理并发时文件可能正在写入的情况
            max_retries = 3
            retry_delay = 0.5  # 秒
            for attempt in range(max_retries):
                try:
                    # 先验证文件是否可读且完整
                    test_img = Image.open(image_input)
                    test_img.verify()  # 验证图像完整性
                    test_img.close()
                    
                    # 文件验证通过，读取tensor
                    if read_image_as_tensor is None:
                        raise RuntimeError("read_image_as_tensor 函数未正确导入")
                    tensor = read_image_as_tensor(image_input)
                    break
                except (OSError, IOError) as e:
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(retry_delay)
                        print(f"[Mask Gen Tool] 文件读取失败（尝试 {attempt + 1}/{max_retries}），等待后重试: {e}")
                    else:
                        # 最后一次尝试失败，抛出异常
                        raise IOError(f"无法读取图像文件（已重试{max_retries}次）: {image_input}. 错误: {e}")
            
            # read_image_as_tensor 返回 [C, H, W]，需要添加 batch 维度
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
            
            # Resize到指定尺寸（检查高度和宽度）
            if tensor.shape[2] != self.img_size or tensor.shape[3] != self.img_size:
                tensor = self._resize_tensor(tensor)
            
            return tensor.to(self.device)
        else:
            raise TypeError(f"不支持的图像输入类型: {type(image_input)}")
    
    def _read_image_as_tensor_from_pil(self, pil_image: Image.Image) -> torch.Tensor:
        """
        从PIL Image创建tensor
        
        Args:
            pil_image: PIL Image对象
            
        Returns:
            tensor: [1, 3, H, W]，值域[0, 1]
        """
        from torchvision.transforms.functional import pil_to_tensor
        
        # 转换为tensor并归一化到[0, 1]
        # pil_to_tensor 返回 [C, H, W]
        tensor = pil_to_tensor(pil_image).float() / 255.0
        
        # 确保是3D tensor [C, H, W]
        if tensor.dim() != 3:
            raise ValueError(f"pil_to_tensor应返回3D tensor [C, H, W]，实际得到 {tensor.dim()}D: {tensor.shape}")
        
        # Resize: 检查高度(H)和宽度(W)，即 shape[1] 和 shape[2]
        if tensor.shape[1] != self.img_size or tensor.shape[2] != self.img_size:
            if KORNIA_AVAILABLE:
                # kornia需要4D输入，先添加batch维度
                tensor = tensor.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
                resize_trans = K.augmentation.Resize([self.img_size, self.img_size])
                tensor = resize_trans(tensor)[0]  # 返回 [C, H, W]
            else:
                # 使用PIL resize
                pil_resized = pil_image.resize((self.img_size, self.img_size), Image.LANCZOS)
                tensor = pil_to_tensor(pil_resized).float() / 255.0  # [C, H, W]
        
        # 添加batch维度: [C, H, W] -> [1, C, H, W]
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _resize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Resize tensor到指定尺寸
        
        Args:
            tensor: [1, 3, H, W]
            
        Returns:
            resized tensor: [1, 3, img_size, img_size]
        """
        if tensor.shape[2] == self.img_size and tensor.shape[3] == self.img_size:
            return tensor
        
        if KORNIA_AVAILABLE:
            resize_trans = K.augmentation.Resize([self.img_size, self.img_size])
            return resize_trans(tensor).to(self.device)
        else:
            # 使用torch的interpolate
            return torch.nn.functional.interpolate(
                tensor, 
                size=(self.img_size, self.img_size), 
                mode='bilinear', 
                align_corners=False
            ).to(self.device)
    
    def _generate_mask_path(self, image_path: str, output_base_dir: Optional[str] = None) -> str:
        """
        根据输入图像路径生成mask保存路径
        
        Args:
            image_path: 输入图像路径，例如：
                /path/to/mvtec_eval/screw/manipulated_front/image/0_2.jpg
                或
                /path/to/mvtec_eval/screw/manipulated_front/ori/0.jpg
            output_base_dir: 输出基础目录（可选），例如：/path/to/outputs
                
        Returns:
            mask保存路径，例如：
            - 如果output_base_dir为None: /path/to/mvtec_eval/screw/manipulated_front/mask/0_2.jpg
            - 如果output_base_dir指定: /path/to/outputs/screw/manipulated_front/mask/0_2.jpg
        """
        # 分离目录和文件名
        image_dir = os.path.dirname(image_path)
        image_filename = os.path.basename(image_path)
        
        # 如果指定了output_base_dir，在新目录下创建相同的目录结构
        if output_base_dir:
            # 从原始路径中提取相对路径部分（item_name/anomaly_type/image或ori）
            path_parts = Path(image_path).parts
            # 查找 'ori' 或 'image' 目录的索引
            dir_idx = None
            for i, part in enumerate(path_parts):
                if part in ['ori', 'image']:
                    dir_idx = i
                    break
            
            if dir_idx and dir_idx >= 2:
                # 提取 item_name 和 anomaly_type
                item_name = path_parts[dir_idx - 2]
                anomaly_type = path_parts[dir_idx - 1]
                # 构建新路径：output_base_dir/item_name/anomaly_type/mask/
                mask_dir = os.path.join(output_base_dir, item_name, anomaly_type, 'mask')
            else:
                # 如果无法解析，使用默认方式
                if '/image/' in image_dir or '\\image\\' in image_dir:
                    mask_dir = image_dir.replace('/image/', '/mask/').replace('\\image\\', '\\mask\\')
                    # 替换基础目录部分
                    if 'mvtec_eval' in mask_dir:
                        mvtec_idx = mask_dir.find('mvtec_eval')
                        if mvtec_idx >= 0:
                            next_slash = mask_dir.find('/', mvtec_idx)
                            if next_slash >= 0:
                                mask_dir = output_base_dir + mask_dir[next_slash:]
                            else:
                                mask_dir = os.path.join(output_base_dir, 'mask')
                    else:
                        mask_dir = os.path.join(output_base_dir, 'mask')
                elif '/ori/' in image_dir or '\\ori\\' in image_dir:
                    mask_dir = image_dir.replace('/ori/', '/mask/').replace('\\ori\\', '\\mask\\')
                    if 'mvtec_eval' in mask_dir:
                        mvtec_idx = mask_dir.find('mvtec_eval')
                        if mvtec_idx >= 0:
                            next_slash = mask_dir.find('/', mvtec_idx)
                            if next_slash >= 0:
                                mask_dir = output_base_dir + mask_dir[next_slash:]
                            else:
                                mask_dir = os.path.join(output_base_dir, 'mask')
                    else:
                        mask_dir = os.path.join(output_base_dir, 'mask')
                else:
                    mask_dir = os.path.join(output_base_dir, 'mask')
        else:
            # 原有逻辑：将 image 或 ori 目录替换为 mask 目录
            if image_dir.endswith('/image') or image_dir.endswith('\\image'):
                mask_dir = image_dir[:-5] + 'mask'  # 移除 '/image'，添加 'mask'
            elif image_dir.endswith('/ori') or image_dir.endswith('\\ori'):
                mask_dir = image_dir[:-3] + 'mask'  # 移除 '/ori'，添加 'mask'
            elif '/image/' in image_dir or '\\image\\' in image_dir:
                mask_dir = image_dir.replace('/image/', '/mask/').replace('\\image\\', '\\mask\\')
            elif '/ori/' in image_dir or '\\ori\\' in image_dir:
                mask_dir = image_dir.replace('/ori/', '/mask/').replace('\\ori\\', '\\mask\\')
            else:
                # 如果路径中没有 image 或 ori，使用默认输出目录
                mask_dir = self.config.get("output_dir", "./masks")
        
        # 确保mask目录存在
        os.makedirs(mask_dir, exist_ok=True)
        
        # 生成mask文件路径（保持原文件名）
        mask_path = os.path.join(mask_dir, image_filename)
        
        return mask_path
    
    @torch.no_grad()
    def generate_mask(
        self,
        normal_image: Union[str, Image.Image, torch.Tensor],
        anomaly_image: Union[str, Image.Image, torch.Tensor],
        threshold: float = 0.5,
        mask_path: Optional[str] = None,
        output_base_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成异常mask
        
        Args:
            normal_image: 正常图像（路径、PIL Image或tensor）
            anomaly_image: 异常图像（路径、PIL Image或tensor）
            threshold: 二值化阈值（默认0.5）
            mask_path: mask保存路径（如果为None，则根据anomaly_image路径自动生成）
            
        Returns:
            dict包含:
                - mask_url: str，mask文件路径
                - quality_score: float，质量分数（1.0 - 异常区域比例）
                - anomaly_ratio: float，异常区域比例 [0, 1]
                - threshold: float，使用的阈值
        """
        # 加载图像
        reference_tensor = self._load_image(normal_image)  # 正常图像作为参考
        query_tensor = self._load_image(anomaly_image)     # 异常图像作为查询
        
        # 准备输入数据
        test_data = {
            "query_image": query_tensor,
            "prompt_image": reference_tensor,
        }
        
        # 模型推理
        predicted_masks = self.model(test_data)
        
        # 获取预测mask（值域[0,1]，值越大表示异常概率越高）
        pred_mask = predicted_masks.squeeze().detach().cpu().numpy()
        
        # 使用阈值转换为二值化mask
        binary_mask = (pred_mask > threshold).astype(np.uint8) * 255
        
        # 计算统计信息
        anomaly_ratio = (binary_mask > 0).sum() / binary_mask.size
        quality_score = 1.0 - anomaly_ratio
        
        # 转换为PIL Image
        mask_image = Image.fromarray(binary_mask, mode='L')
        
        # 生成mask保存路径
        if mask_path is None:
            # 如果anomaly_image是路径，根据它生成mask路径
            if isinstance(anomaly_image, str) and os.path.exists(anomaly_image):
                mask_path = self._generate_mask_path(anomaly_image, output_base_dir)
            else:
                # 如果无法从路径生成，使用默认输出目录
                output_dir = self.config.get("output_dir", "./masks")
                os.makedirs(output_dir, exist_ok=True)
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                mask_path = os.path.join(output_dir, f"mask_{timestamp}.png")
        
        # 保存mask
        try:
            mask_image.save(mask_path, format='PNG')
            print(f"[Mask Gen Tool] Mask已保存到: {mask_path}")
        except Exception as e:
            print(f"[Mask Gen Tool] 警告: 保存mask失败: {e}")
            # 即使保存失败，也返回路径
        
        # 构建返回结果
        result = {
            "mask_url": mask_path,
            "quality_score": float(quality_score),
            "anomaly_ratio": float(anomaly_ratio),
            "threshold": float(threshold),
        }
        
        return result


# 全局工具实例（单例模式）
_mask_gen_tool = None


def get_mask_gen_tool() -> MaskGenTool:
    """获取掩码生成工具实例（单例模式）"""
    global _mask_gen_tool
    if _mask_gen_tool is None:
        _mask_gen_tool = MaskGenTool()
    return _mask_gen_tool


def mask_gen(
    normal_image: Union[str, Image.Image, torch.Tensor, int],
    anomaly_image: Union[str, Image.Image, torch.Tensor, int],
    conversation_images: Optional[list] = None,
    threshold: float = 0.5,
    original_image_path: Optional[str] = None,
    output_base_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    掩码生成函数（工具接口）
    
    这是agent调用的主要接口函数
    
    Args:
        normal_image: 正常图像，可以是：
            - 整数索引（1-based，指向conversation_images中的图像）
            - 图像路径（字符串）
            - PIL Image 对象
            - torch.Tensor
        anomaly_image: 异常图像，可以是：
            - 整数索引（1-based，指向conversation_images中的图像）
            - 图像路径（字符串）
            - PIL Image 对象
            - torch.Tensor
        conversation_images: 对话历史中的图像列表（当使用索引时必需）
        threshold: 二值化阈值（默认0.5）
        original_image_path: 原始图像路径（可选），用于推断mask保存路径
        output_base_dir: 输出基础目录（可选），用于指定结果保存位置
        
    Returns:
        包含生成结果的字典：
        {
            "mask_url": str,        # mask文件路径
            "quality_score": float, # 质量分数
            "anomaly_ratio": float, # 异常区域比例
            "threshold": float,     # 使用的阈值
            "status": str           # "success" 或 "error"
        }
    """
    try:
        tool = get_mask_gen_tool()
        
        # 保存原始图像路径（用于生成mask路径）
        original_anomaly_image = anomaly_image
        
        # 解析正常图像
        if isinstance(normal_image, int):
            if conversation_images is None:
                raise ValueError("当normal_image为索引时，必须提供conversation_images")
            if normal_image < 1 or normal_image > len(conversation_images):
                raise ValueError(f"正常图像索引 {normal_image} 超出范围 (1-{len(conversation_images)})")
            normal_image = conversation_images[normal_image - 1]
        
        # 解析异常图像
        if isinstance(anomaly_image, int):
            if conversation_images is None:
                raise ValueError("当anomaly_image为索引时，必须提供conversation_images")
            if anomaly_image < 1 or anomaly_image > len(conversation_images):
                raise ValueError(f"异常图像索引 {anomaly_image} 超出范围 (1-{len(conversation_images)})")
            original_anomaly_image = conversation_images[anomaly_image - 1]
            anomaly_image = original_anomaly_image
        
        # 生成mask路径（如果anomaly_image是路径）
        mask_path = None
        if isinstance(original_anomaly_image, str):
            # 检查路径是否包含 /image/ 或 /ori/（绝对路径且符合预期格式）
            if os.path.isabs(original_anomaly_image) and ('/image/' in original_anomaly_image or '/ori/' in original_anomaly_image or original_anomaly_image.endswith('/image') or original_anomaly_image.endswith('/ori')):
                if os.path.exists(original_anomaly_image):
                    mask_path = tool._generate_mask_path(original_anomaly_image, output_base_dir)
                    print(f"[Mask Gen] 从异常图像路径生成mask路径: {mask_path}")
        
        # 如果无法从异常图像路径生成mask路径，尝试使用原始图像路径
        if mask_path is None and original_image_path is not None:
            # 从原始图像路径推断mask路径
            # 例如：/path/to/ori/0.jpg -> /path/to/mask/0_2_3.jpg
            if os.path.exists(original_image_path):
                # 获取异常图像的文件名（如果可用）
                anomaly_filename = None
                if isinstance(original_anomaly_image, str):
                    anomaly_filename = os.path.basename(original_anomaly_image)
                
                # 从原始图像路径生成mask路径
                mask_path = tool._generate_mask_path(original_image_path, output_base_dir)
                
                # 如果异常图像有文件名，使用它；否则使用原始文件名
                if anomaly_filename:
                    mask_dir = os.path.dirname(mask_path)
                    mask_path = os.path.join(mask_dir, anomaly_filename)
                print(f"[Mask Gen] 使用原始图像路径推断mask路径: {mask_path}")
        
        # 生成mask
        result = tool.generate_mask(normal_image, anomaly_image, threshold, mask_path, output_base_dir)
        result["status"] = "success"
        return result
        
    except Exception as e:
        error_msg = f"生成mask失败: {str(e)}"
        print(f"[Mask Gen] 错误: {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "mask_url": "",
            "quality_score": 0.0,
            "anomaly_ratio": 0.0,
            "threshold": threshold,
            "status": "error",
            "error": error_msg
        }

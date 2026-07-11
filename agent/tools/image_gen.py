"""
图像生成工具类
用于调用 nano-banana (Gemini API) 进行局部图像编辑，生成异常图像
基于 Google Gemini API 实现
"""
from typing import Optional, Dict, Any, Union
import os
import json
from datetime import datetime
from pathlib import Path
from PIL import Image
from io import BytesIO
import base64
from google import genai
from google.genai import types


def load_gemini_config() -> Dict[str, Any]:
    """
    加载 Gemini API 配置
    
    优先级：
    1. 环境变量 GEMINI_API_KEY（最高优先级）
    2. 配置文件 config/gemini_config.json
    3. 安全的内置默认值（最低优先级）
    
    Returns:
        配置字典
    """
    default_config = {
        "model": "gemini-3.1-flash-image-preview",
        "api_key": "",
        "base_url": "",
        "aspect_ratio": "1:1",
        "image_size": "1K",
        "output_dir": "./outputs",
    }
    
    # 尝试从配置文件加载
    config_file = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'config', 
        'gemini_config.json'
    )
    config_file = os.path.abspath(config_file)
    
    file_config = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            print(f"[Image Gen Tool] 从配置文件加载: {config_file}")
        except Exception as e:
            print(f"[Image Gen Tool] 警告: 无法加载配置文件 {config_file}: {e}")
    
    # 合并配置：默认配置 < 文件配置 < 环境变量
    merged_config = default_config.copy()
    merged_config.update(file_config)
    
    # 环境变量优先级最高
    env_api_key = os.getenv('GEMINI_API_KEY')
    if env_api_key:
        merged_config['api_key'] = env_api_key
        print(f"[Image Gen Tool] 从环境变量加载 API key")
    if os.getenv('GEMINI_BASE_URL'):
        merged_config['base_url'] = os.environ['GEMINI_BASE_URL']
    if os.getenv('GEMINI_IMAGE_MODEL'):
        merged_config['model'] = os.environ['GEMINI_IMAGE_MODEL']
    
    return merged_config


class ImageGenTool:
    """图像生成工具类"""
    
    def __init__(self, config_dict: dict = None):
        """
        初始化图像生成工具
        
        Args:
            config_dict: 配置字典，如果为None则从配置文件加载
        """
        if config_dict is None:
            # 从配置文件加载配置
            self.config = load_gemini_config()
        else:
            self.config = config_dict
        
        # 初始化 Gemini API 客户端
        base_url = self.config.get("base_url")
        self.client = genai.Client(
            api_key=self.config.get("api_key", ""),
            http_options={"base_url": base_url} if base_url else None,
        )
        self.model = self.config.get("model", "gemini-2.5-flash-image")
        self.aspect_ratio = self.config.get("aspect_ratio", "1:1")
        self.image_size = self.config.get("image_size", "1K")
        # self.response_modalities = self.config.get("response_modalities", ["Image"])
        self.output_dir = self.config.get("output_dir", "./output")
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _load_image(self, image_path: str) -> Image.Image:
        """
        加载图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            PIL Image 对象
        """
        if not os.path.exists(image_path):
            raise ValueError(f"图片文件不存在: {image_path}")
        
        return Image.open(image_path)
    
    def _load_image_from_base64(self, image_data: str) -> Image.Image:
        """
        从base64字符串加载图片
        
        Args:
            image_data: base64编码的图片数据
            
        Returns:
            PIL Image 对象
        """
        # 移除可能的数据URL前缀
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        return Image.open(BytesIO(image_bytes))
    
    def _generate_output_path(self, input_path: str, image_index: int, output_base_dir: Optional[str] = None) -> str:
        """
        根据输入图片路径生成输出图片路径
        
        Args:
            input_path: 输入图片路径，例如：/path/to/mvtec_eval/screw/manipulated_front/ori/0.jpg
            image_index: 生成图片的索引（2, 3, 4...）
            output_base_dir: 输出基础目录（可选），例如：/path/to/outputs
            
        Returns:
            输出图片路径，例如：
            - 如果output_base_dir为None: /path/to/mvtec_eval/screw/manipulated_front/image/0_2.jpg
            - 如果output_base_dir指定: /path/to/outputs/screw/manipulated_front/image/0_2.jpg
        """
        # 分离目录和文件名
        input_dir = os.path.dirname(input_path)
        input_filename = os.path.basename(input_path)
        
        # 分离文件名和扩展名
        name_without_ext, ext = os.path.splitext(input_filename)
        
        # 如果指定了output_base_dir，在新目录下创建相同的目录结构
        if output_base_dir:
            # 从原始路径中提取相对路径部分（item_name/anomaly_type/ori）
            # 例如：/path/to/mvtec_eval/screw/manipulated_front/ori/0.jpg
            # 需要提取：screw/manipulated_front
            path_parts = Path(input_path).parts
            # 查找 'ori' 或 'image' 目录的索引
            ori_idx = None
            for i, part in enumerate(path_parts):
                if part in ['ori', 'image']:
                    ori_idx = i
                    break
            
            if ori_idx and ori_idx >= 2:
                # 提取 item_name 和 anomaly_type
                item_name = path_parts[ori_idx - 2]
                anomaly_type = path_parts[ori_idx - 1]
                # 构建新路径：output_base_dir/item_name/anomaly_type/image/
                output_dir = os.path.join(output_base_dir, item_name, anomaly_type, 'image')
            else:
                # 如果无法解析，使用默认方式
                if '/ori/' in input_dir or '\\ori\\' in input_dir:
                    output_dir = input_dir.replace('/ori/', '/image/').replace('\\ori\\', '\\image\\')
                    # 替换基础目录部分
                    if 'mvtec_eval' in output_dir:
                        # 找到mvtec_eval的位置，替换为output_base_dir
                        mvtec_idx = output_dir.find('mvtec_eval')
                        if mvtec_idx >= 0:
                            # 找到下一个斜杠
                            next_slash = output_dir.find('/', mvtec_idx)
                            if next_slash >= 0:
                                output_dir = output_base_dir + output_dir[next_slash:]
                            else:
                                output_dir = os.path.join(output_base_dir, 'image')
                    else:
                        output_dir = os.path.join(output_base_dir, 'image')
                else:
                    output_dir = os.path.join(output_base_dir, 'image')
        else:
            # 原有逻辑：将 ori 目录替换为 image 目录
            if input_dir.endswith('/ori') or input_dir.endswith('\\ori'):
                output_dir = input_dir[:-3] + 'image'
            elif '/ori/' in input_dir or '\\ori\\' in input_dir:
                output_dir = input_dir.replace('/ori/', '/image/').replace('\\ori\\', '\\image\\')
            else:
                # 如果路径中没有 ori，使用默认输出目录
                output_dir = self.output_dir
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名：原文件名_索引.扩展名
        output_filename = f"{name_without_ext}_{image_index}{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        return output_path
    
    def generate(
        self,
        target_image: Any,
        prompt: str,
        conversation_images: Optional[list] = None,
        output_base_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行图像生成（局部编辑）
        
        注意：此工具只能对原图（索引1）进行操作，其他索引将被自动修正为1
        
        Args:
            target_image: 目标图像，可以是：
                - 整数索引（1-based，必须为1，指向conversation_images中的原图）
                - 图像路径（字符串）
                - PIL Image 对象
                - base64编码的图像数据
            prompt: 编辑提示词，应该遵循格式："Using the provided image, change only... Keep the rest unchanged."
            conversation_images: 对话历史中的图像列表（当target_image为索引时使用）
            output_base_dir: 输出基础目录（可选），用于指定结果保存位置
            
        Returns:
            包含生成结果的字典：
            {
                "image_index": int,  # 新生成图像的索引（在对话中的位置）
                "image_url": str,     # 图像路径或base64数据URL
                "status": str         # "success" 或 "error"
            }
        """
        print(f"[Image Gen Tool] 开始生成...")
        print(f"  - Prompt: {prompt[:100]}...")
        
        try:
            # 1. 解析目标图像
            if isinstance(target_image, int):
                # 索引模式：从对话历史中获取
                if conversation_images is None:
                    raise ValueError("当target_image为索引时，必须提供conversation_images")
                
                # 健壮性检查：强制只能使用索引1（原图）
                if target_image != 1:
                    print(f"[Image Gen Tool] 错误: image_gen 工具只能对原图（索引1）进行操作")
                    print(f"  - 传入的索引: {target_image}")
                    print(f"  - 已自动修正为: 1")
                    target_image = 1
                
                if target_image < 1 or target_image > len(conversation_images):
                    raise ValueError(f"图像索引 {target_image} 超出范围 (1-{len(conversation_images)})")
                
                image_source = conversation_images[target_image - 1]  # 转换为0-based索引
                print(f"  - 使用对话中的图像索引: {target_image} (原图)")
            elif isinstance(target_image, str):
                # 字符串：可能是路径或base64数据
                if os.path.exists(target_image):
                    image_source = target_image
                    print(f"  - 使用图像路径: {target_image}")
                elif target_image.startswith('data:image') or len(target_image) > 100:
                    # 可能是base64数据
                    image_source = target_image
                    print(f"  - 使用base64图像数据")
                else:
                    raise ValueError(f"无法识别图像源: {target_image}")
            elif isinstance(target_image, Image.Image):
                # PIL Image对象
                image_source = target_image
                print(f"  - 使用PIL Image对象")
            else:
                raise ValueError(f"不支持的图像类型: {type(target_image)}")
            
            # 2. 加载图像
            if isinstance(image_source, str) and os.path.exists(image_source):
                base_img = self._load_image(image_source)
            elif isinstance(image_source, str) and (image_source.startswith('data:image') or len(image_source) > 100):
                base_img = self._load_image_from_base64(image_source)
            elif isinstance(image_source, Image.Image):
                base_img = image_source
            else:
                raise ValueError(f"无法加载图像: {image_source}")
            
            print(f"  - 图像尺寸: {base_img.size}")
            
            # 3. 准备API调用内容
            contents = []
            contents.append(base_img)
            contents.append(prompt)
            
            print(f"[Image Gen Tool] 调用 Gemini API 进行生成...")
            print(f"  - 模型: {self.model}")
            
            # 4. 构建配置
            image_config = types.ImageConfig(
                aspect_ratio=self.aspect_ratio
            )
            
            # 设置图像尺寸（Gemini 3.1 Flash Image 支持 1K/2K/4K）
            if self.image_size in ["1K", "2K", "4K"]:
                image_config.image_size = self.image_size
                print(f"  - 图像尺寸: {self.image_size}")
            
            config_obj = types.GenerateContentConfig(
                # response_modalities=self.response_modalities,
                image_config=image_config
            )
            
            # print(f"  - 响应模态: {self.response_modalities}")
            print(f"  - 宽高比: {self.aspect_ratio}")
            
            # 5. 调用 API
            print(f"[Image Gen Tool] 正在发送API请求...")
            import time
            api_start_time = time.time()
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config_obj
            )
            
            api_duration = time.time() - api_start_time
            print(f"[Image Gen Tool] API 请求完成，耗时: {api_duration:.2f}秒")
            
            # 6. 计算新图像的索引（假设是对话中的下一张图像）
            # 如果提供了conversation_images，新图像索引为 len(conversation_images) + 1
            if conversation_images is not None:
                new_image_index = len(conversation_images) + 1
            else:
                new_image_index = 2  # 默认假设这是第二张图像（第一张是原图）
            
            # 7. 根据输入路径生成输出路径（如果输入是文件路径）
            input_image_path = None
            if isinstance(image_source, str) and os.path.exists(image_source):
                input_image_path = image_source
            elif isinstance(target_image, int) and conversation_images:
                # 如果target_image是索引，尝试从conversation_images获取路径
                img_ref = conversation_images[target_image - 1]
                if isinstance(img_ref, str) and os.path.exists(img_ref):
                    input_image_path = img_ref
            
            # 8. 处理响应并保存图像
            generated_image_path = None
            
            for part in response.parts:
                # 处理文本响应（如果有）
                if part.text is not None:
                    print(f"[Image Gen Tool] 收到文本响应: {part.text[:100]}...")
                
                # 处理图像响应
                elif part.inline_data is not None:
                    # 根据输入路径生成输出路径
                    if input_image_path:
                        generated_image_path = self._generate_output_path(input_image_path, new_image_index, output_base_dir)
                        print(f"[Image Gen Tool] 生成输出路径: {generated_image_path}")
                    else:
                        # 如果没有输入路径，使用默认路径（带时间戳）
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"image_gen_{timestamp}.jpg"
                        generated_image_path = os.path.join(self.output_dir, filename)
                    
                    # 从inline_data中提取图像数据
                    try:
                        inline_data = part.inline_data
                        
                        # 获取图像字节数据
                        if hasattr(inline_data, 'data'):
                            image_bytes = inline_data.data
                        elif isinstance(inline_data, dict):
                            image_bytes = inline_data.get('data')
                            # 如果是base64字符串，需要解码
                            if isinstance(image_bytes, str):
                                image_bytes = base64.b64decode(image_bytes)
                        else:
                            image_bytes = inline_data
                        
                        # 确保是bytes类型
                        if not isinstance(image_bytes, bytes):
                            image_bytes = bytes(image_bytes)
                        
                        # 从字节数据创建PIL Image
                        generated_image = Image.open(BytesIO(image_bytes))
                        
                        # Resize到256×256像素
                        # 注意: 1K (1024×1024) 需要 resize 到 256×256
                        # 这是为了与训练数据集保持一致
                        original_size = generated_image.size
                        target_size = (256, 256)
                        if generated_image.size != target_size:
                            print(f"[Image Gen Tool] 调整图像尺寸: {original_size} → {target_size}")
                            generated_image = generated_image.resize(target_size, Image.Resampling.LANCZOS)
                        else:
                            print(f"[Image Gen Tool] 图像尺寸已为 {target_size}，无需调整")
                        
                        # 保存图像（确保文件完整写入）
                        generated_image.save(generated_image_path, format='JPEG', quality=95)
                        
                        # 强制刷新文件系统缓存
                        if hasattr(os, 'sync'):
                            os.sync()
                        
                        # 验证文件是否可读
                        try:
                            test_img = Image.open(generated_image_path)
                            test_img.verify()
                            test_img.close()
                        except Exception as verify_error:
                            print(f"[Image Gen Tool] 警告: 文件验证失败: {verify_error}")
                            raise
                        
                        print(f"[Image Gen Tool] 生成的图像已保存到: {generated_image_path}")
                        print(f"  - 原始尺寸: {original_size}")
                        print(f"  - 保存尺寸: {generated_image.size}")
                        
                    except Exception as img_error:
                        print(f"[Image Gen Tool] 图像处理错误: {img_error}")
                        import traceback
                        traceback.print_exc()
                        
                        # 尝试备用方法：先resize再保存
                        try:
                            print(f"[Image Gen Tool] 尝试备用保存方法...")
                            # 从字节数据创建PIL Image并resize
                            backup_image = Image.open(BytesIO(image_bytes))
                            backup_original_size = backup_image.size
                            if backup_image.size != (256, 256):
                                backup_image = backup_image.resize((256, 256), Image.Resampling.LANCZOS)
                                print(f"[Image Gen Tool] 备用方法调整尺寸: {backup_original_size} → (256, 256)")
                            # 保存resize后的图片
                            backup_image.save(generated_image_path, format='JPEG', quality=95)
                            print(f"[Image Gen Tool] 使用备用方法保存成功: {generated_image_path}")
                            print(f"  - 保存尺寸: {backup_image.size}")
                        except Exception as save_error:
                            print(f"[Image Gen Tool] 备用保存方法也失败: {save_error}")
                            generated_image_path = None
                            continue
            
            if generated_image_path is None:
                error_msg = "API响应中未找到图像数据"
                print(f"[Image Gen Tool] 错误: {error_msg}")
                return {
                    "image_index": None,
                    "image_url": "",
                    "status": "error",
                    "error": error_msg
                }
            
            # 返回结果（使用实际图片路径）
            return {
                "image_index": new_image_index,
                "image_url": generated_image_path,  # 实际图片路径
                "status": "success"
            }
            
        except Exception as e:
            error_msg = f"生成失败: {str(e)}"
            print(f"[Image Gen Tool] 错误: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "image_index": None,
                "image_url": "",
                "status": "error",
                "error": error_msg
            }


# 全局工具实例（单例模式）
_image_gen_tool = None


def get_image_gen_tool() -> ImageGenTool:
    """获取图像生成工具实例（单例模式）"""
    global _image_gen_tool
    if _image_gen_tool is None:
        _image_gen_tool = ImageGenTool()
    return _image_gen_tool


def image_gen(target_image: Any, prompt: str, conversation_images: Optional[list] = None, output_base_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    图像生成函数（工具接口）
    
    这是agent调用的主要接口函数
    
    注意：image_gen 工具只能对原图（索引1）进行操作，如果传入其他索引，将自动强制为1
    
    Args:
        target_image: 目标图像索引（1-based，必须为1）或图像路径
        prompt: 编辑提示词
        conversation_images: 对话历史中的图像列表（可选）
        output_base_dir: 输出基础目录（可选），用于指定结果保存位置
        
    Returns:
        包含生成结果的字典
    """
    # 健壮性处理：强制 target_image 只能为 1（原图）
    if isinstance(target_image, int):
        if target_image != 1:
            print(f"[Image Gen Tool] 警告: image_gen 工具只能对原图（索引1）进行操作")
            print(f"  - 传入的索引: {target_image}")
            print(f"  - 已自动修正为: 1")
            target_image = 1
    
    tool = get_image_gen_tool()
    return tool.generate(target_image, prompt, conversation_images, output_base_dir)

"""
提示词生成工具类
使用Gemini API根据物体类型和异常类型生成初始提示词
基于 workflow/tools/mllm_critic.py 的实现
"""
from typing import Optional, Dict, Any, Union
import os
import sys
import json
import base64
from pathlib import Path
from PIL import Image

current_dir = Path(__file__).resolve().parent
agent_dir = current_dir.parent
project_root = agent_dir.parent

# 添加项目根目录到路径
for path in [str(project_root)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# 导入Gemini API
try:
    from google import genai
    from google.genai import types
except ImportError as e:
    print(f"警告: 无法导入 google.genai: {e}")
    genai = None
    types = None


def load_gemini_config() -> Dict[str, Any]:
    """
    加载 Gemini API 配置
    
    从配置文件 config/gemini_config.json 读取配置
    如果配置文件不存在，则使用默认配置
    
    Returns:
        配置字典
    """
    # 默认配置（仅在配置文件不存在时使用）
    default_config = {
        "model_prompt": "gemini-3.1-pro-preview",  # 提示词生成工具使用的模型
        "api_key": "",
        "base_url": "",
        "thinking_level": "low",
        "timeout": 120,  # HTTP 连接超时（秒）
        "api_timeout": 120,  # API 调用超时（秒）
        "max_retries": 3,  # 最大重试次数
        "retry_delay": 5,  # 重试延迟（秒）
    }
    
    # 从配置文件加载
    config_file = os.path.join(
        current_dir.parent, 
        'config', 
        'gemini_config.json'
    )
    config_file = os.path.abspath(config_file)
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"[Prompt Gen Tool] 从配置文件加载: {config_file}")
            default_config.update(config)
        except Exception as e:
            print(f"[Prompt Gen Tool] 警告: 无法加载配置文件 {config_file}: {e}")
            print(f"[Prompt Gen Tool] 使用默认配置")
    else:
        print(f"[Prompt Gen Tool] 配置文件不存在: {config_file}")
        print(f"[Prompt Gen Tool] 使用默认配置")
    if os.getenv("GEMINI_API_KEY"):
        default_config["api_key"] = os.environ["GEMINI_API_KEY"]
    if os.getenv("GEMINI_BASE_URL"):
        default_config["base_url"] = os.environ["GEMINI_BASE_URL"]
    if os.getenv("GEMINI_REASONING_MODEL"):
        default_config["model_prompt"] = os.environ["GEMINI_REASONING_MODEL"]
    return default_config


class PromptGenTool:
    """提示词生成工具类"""
    
    def __init__(self, config_dict: dict = None):
        """
        初始化提示词生成工具
        
        Args:
            config_dict: 配置字典，如果为None则从配置文件加载
        """
        if config_dict is None:
            self.config = load_gemini_config()
        else:
            self.config = config_dict
        
        if genai is None:
            raise RuntimeError("google.genai 模块未正确导入，请检查依赖")
        
        # 初始化 Gemini API 客户端
        # 添加超时设置，避免长时间阻塞
        # 创建自定义的 httpx.Client，设置详细的超时参数
        import httpx
        timeout_seconds = self.config.get("timeout", 120)
        
        # 创建自定义的 httpx.Client，设置详细的超时参数
        httpx_client = httpx.Client(
            timeout=httpx.Timeout(
                connect=30.0,  # 连接超时：30秒
                read=timeout_seconds,  # 读取超时：120秒
                write=30.0,  # 写入超时：30秒
                pool=5.0  # 连接池超时：5秒
            ),
            follow_redirects=True
        )
        
        # 同时设置 base_url 和自定义的 httpx.Client
        # base_url 必须在 http_options 的顶层设置，不能在 httpxClient 中设置
        http_options = {"httpxClient": httpx_client}
        if self.config.get("base_url"):
            http_options["base_url"] = self.config["base_url"]
        self.client = genai.Client(
            api_key=self.config.get("api_key", ""),
            http_options=http_options
        )
        # 优先使用 model_prompt，如果不存在则使用 model 或默认值
        self.model = self.config.get("model_prompt", self.config.get("model", "gemini-3.1-pro-preview"))
        self.thinking_level = self.config.get("thinking_level", "low")
    
    def _load_image(self, image_input: Union[str, Image.Image, bytes]) -> bytes:
        """
        加载图片，支持文件路径、PIL Image或字节数据
        
        Args:
            image_input: 图片路径、PIL Image或字节数据
            
        Returns:
            图片的字节数据
        """
        # 如果是PIL Image
        if isinstance(image_input, Image.Image):
            from io import BytesIO
            buffer = BytesIO()
            image_input.save(buffer, format='JPEG')
            return buffer.getvalue()
        
        # 如果是字节数据
        if isinstance(image_input, bytes):
            return image_input
        
        # 如果是字符串
        if isinstance(image_input, str):
            # 优先检查是否为文件路径（避免长路径被误判为base64）
            if os.path.exists(image_input):
                # 使用 PIL 重新编码图像，确保格式正确并移除可能的元数据问题
                try:
                    from io import BytesIO
                    img = Image.open(image_input)
                    # 转换为 RGB 模式（如果需要）
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    buffer = BytesIO()
                    img.save(buffer, format='JPEG', quality=95)
                    return buffer.getvalue()
                except Exception as e:
                    # 如果 PIL 处理失败，回退到直接读取
                    print(f"[Prompt Gen Tool] 警告: 使用 PIL 重新编码图像失败，使用原始文件: {e}")
                    import traceback
                    traceback.print_exc()
                    with open(image_input, "rb") as f:
                        return f.read()
            
            # 如果不是文件路径，检查是否为base64字符串
            if image_input.startswith("data:image"):
                # 尝试解码base64
                try:
                    if "," in image_input:
                        # data:image/png;base64,xxx 格式
                        base64_data = image_input.split(",")[1]
                    else:
                        base64_data = image_input
                    return base64.b64decode(base64_data)
                except Exception:
                    pass
        
        raise ValueError(f"无法加载图片: {image_input}")
    
    def _prepare_image_content(self, image_data: bytes, mime_type: str = "image/jpeg"):
        """
        准备图片内容用于API调用
        
        Args:
            image_data: 图片字节数据
            mime_type: 图片MIME类型
            
        Returns:
            图片内容对象
        """
        # 尝试使用 Part API（如果可用）
        try:
            # 检查是否有 from_bytes 方法
            if hasattr(types.Part, 'from_bytes'):
                return types.Part.from_bytes(data=image_data, mime_type=mime_type)
            # 或者使用 inline_data
            elif hasattr(types, 'Blob'):
                return types.Part(inline_data=types.Blob(data=image_data, mime_type=mime_type))
        except (AttributeError, TypeError):
            pass
        
        # 如果 Part API 不可用，返回字典格式（API应该能处理）
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        return {
            "inline_data": {
                "data": image_base64,
                "mime_type": mime_type
            }
        }
    
    def generate(
        self,
        image: Union[str, Image.Image, bytes],
        item_name: str,
        anomaly_type: str
    ) -> str:
        """
        生成用于nano-banana的图像编辑prompt
        直接返回文本prompt，不需要JSON解析
        
        Args:
            image: 正常图片（路径、PIL Image或字节数据）
            item_name: 物品名称
            anomaly_type: 异常类型
            
        Returns:
            用于nano-banana的图像编辑prompt字符串
        """
        print(f"[Prompt Gen Tool] 生成nano-banana prompt...")
        print(f"  - 物品名称: {item_name}")
        print(f"  - 异常类型: {anomaly_type}")
        
        prompt = f"""You are an expert prompt engineer for industrial image editing.  
Your task is to generate a **single, high-quality text prompt** for an image editing model (nano-banana) to synthesize **realistic industrial anomalies**.

You will be given the following inputs:
- `normal_image`: the reference image of a normal {item_name}
- `item_name`: {item_name}, the object category
- `anomaly_type`: {anomaly_type}, the defect type

Your goal is to produce a **local image editing prompt** that improves or refines the anomaly in `anomaly_image` while preserving the rest of the image.

---

### Internal reasoning steps (do NOT include these in the output):

1. Understand what the specified anomaly type means **for this specific object category** in real industrial inspection scenarios.
2. Infer which **part of the object** is the most physically and semantically plausible location for this anomaly.
3. Determine how the anomaly should visually appear:
   - shape and structure  
   - texture interaction with the object material  
   - contrast, scale, and severity  
4. Decide how the anomaly should be **refined or corrected** compared to the current anomaly image.

---

### Prompt construction rules (VERY IMPORTANT):

- The prompt MUST follow a **local image editing style**, such as:
  "Using the provided image, change only … Keep the rest of the image unchanged."
- Only describe **what should be edited**, never describe global or stylistic changes.
- Be **hyper-specific** about:
  - the exact object part
  - the anomaly appearance
  - how the anomaly integrates with surrounding material
  - the **limited spatial extent of the anomaly (small, localized, subtle)**
- Explicitly state what must remain unchanged (background, lighting, object geometry).
- Use **positive, semantic constraints** instead of negative commands.
- The intent is **industrial realism**, not artistic or aesthetic enhancement.

---

### Output format (STRICT):

- Output **only one paragraph**.
- Output **only the final nano-banana image editing prompt string**.
- Do NOT include explanations, bullet points, headings, or metadata.

---

### Example style (for reference only):

"Using the provided image of a metal screw, modify only the surface near the threaded region to introduce a subtle longitudinal scratch that follows the screw's axis. The scratch should appear shallow, metallic, and consistent with wear from mechanical contact. Keep the overall shape, background, lighting, and all other regions of the screw unchanged."

---

Now generate the nano-banana image editing prompt based on the given inputs.

"""
        
        contents = []
        try:
            image_data = self._load_image(image)
            contents.append(self._prepare_image_content(image_data))
            contents.append(f"This is the normal_image: a reference image of a normal {item_name}.")
        except Exception as e:
            print(f"警告: 无法加载图片: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认prompt
            return f"Using the provided image of a {item_name}, modify it to introduce a realistic {anomaly_type} anomaly while keeping the rest of the image unchanged."
        
        contents.append(prompt)
        
        try:
            print(f"[Prompt Gen Tool] 调用 Gemini API 生成prompt...")
            print(f"  - 模型: {self.model}")
            # 强制刷新输出，确保日志及时显示
            import sys
            sys.stdout.flush()
            
            import time
            api_start_time = time.time()
            
            # 添加超时和重试机制
            max_retries = self.config.get("max_retries", 3)
            retry_delay = self.config.get("retry_delay", 5)
            api_timeout = self.config.get("api_timeout", 120)  # API调用超时（秒）
            
            response = None
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        print(f"[Prompt Gen Tool] 重试第 {attempt} 次...")
                        time.sleep(retry_delay * attempt)  # 递增延迟
                        sys.stdout.flush()
                    
                    # 显示进度提示
                    print(f"[Prompt Gen Tool] 正在调用 API (尝试 {attempt + 1}/{max_retries})，请稍候...")
                    print(f"[Prompt Gen Tool] 超时设置: {api_timeout}秒")
                    sys.stdout.flush()
                    
                    # 检查是否已经超时
                    elapsed = time.time() - api_start_time
                    if elapsed > api_timeout:
                        print(f"[Prompt Gen Tool] 已达到总超时时间 ({api_timeout}秒)，停止重试")
                        sys.stdout.flush()
                        raise TimeoutError(f"API调用总超时 ({api_timeout}秒)")
                    
                    # 调用 API（依赖 http_options 中的 timeout 设置）
                    # 注意：如果 google.genai 库不支持超时，这里可能会阻塞
                    # 建议检查网络连接和 API 服务状态
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level)
                        ),
                    )
                    
                    # 如果成功，跳出重试循环
                    break
                    
                except Exception as e:
                    last_error = e
                    elapsed = time.time() - api_start_time
                    print(f"[Prompt Gen Tool] API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    print(f"[Prompt Gen Tool] 已耗时: {elapsed:.2f}秒")
                    sys.stdout.flush()
                    
                    # 如果已经超时，不再重试
                    if elapsed > api_timeout:
                        print(f"[Prompt Gen Tool] 已达到超时时间 ({api_timeout}秒)，停止重试")
                        sys.stdout.flush()
                        raise TimeoutError(f"API调用超时 ({api_timeout}秒)")
                    
                    # 最后一次尝试失败，抛出异常
                    if attempt == max_retries - 1:
                        raise
            
            if response is None:
                raise Exception(f"API调用失败，已重试 {max_retries} 次: {last_error}")
            
            api_duration = time.time() - api_start_time
            print(f"[Prompt Gen Tool] API响应接收成功，耗时: {api_duration:.2f}秒")
            sys.stdout.flush()
            
            response_text = response.text.strip()
            
            # 直接返回响应文本（已经是prompt字符串，不需要JSON解析）
            # 清理可能的引号和多余空白
            if response_text.startswith('"') and response_text.endswith('"'):
                response_text = response_text[1:-1]
            elif response_text.startswith("'") and response_text.endswith("'"):
                response_text = response_text[1:-1]
            
            response_text = response_text.strip()
            
            print(f"[Prompt Gen Tool] Prompt生成完成，长度: {len(response_text)} 字符")
            print(f"[Prompt Gen Tool] Prompt预览: {response_text[:200]}...")
            sys.stdout.flush()
            
            # 如果响应为空或太短，使用默认prompt
            if not response_text or len(response_text) < 20:
                print(f"[Prompt Gen Tool] 响应过短，使用默认prompt")
                sys.stdout.flush()
                return f"Using the provided image of a {item_name}, modify it to introduce a realistic {anomaly_type} anomaly while keeping the rest of the image unchanged."
            
            return response_text
            
        except Exception as e:
            print(f"[Prompt Gen Tool] API调用失败: {e}")
            import traceback
            traceback.print_exc()
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            # 返回默认prompt
            return f"Using the provided image of a {item_name}, modify it to introduce a realistic {anomaly_type} anomaly while keeping the rest of the image unchanged."


# 全局工具实例（单例模式）
_prompt_gen_tool = None


def get_prompt_gen_tool() -> PromptGenTool:
    """获取提示词生成工具实例（单例模式）"""
    global _prompt_gen_tool
    if _prompt_gen_tool is None:
        _prompt_gen_tool = PromptGenTool()
    return _prompt_gen_tool


def prompt_gen(
    image: Union[str, Image.Image, bytes, int],
    item_name: str,
    anomaly_type: str,
    conversation_images: Optional[list] = None
) -> str:
    """
    提示词生成函数（工具接口）
    
    这是agent调用的主要接口函数
    
    Args:
        image: 正常图像，可以是：
            - 整数索引（1-based，指向conversation_images中的图像）
            - 图像路径（字符串）
            - PIL Image 对象
            - 字节数据
        item_name: 物品名称
        anomaly_type: 异常类型
        conversation_images: 对话历史中的图像列表（当使用索引时必需）
        
    Returns:
        用于nano-banana的图像编辑prompt字符串
    """
    try:
        tool = get_prompt_gen_tool()
        
        # 解析图像
        if isinstance(image, int):
            if conversation_images is None:
                raise ValueError("当image为索引时，必须提供conversation_images")
            if image < 1 or image > len(conversation_images):
                raise ValueError(f"图像索引 {image} 超出范围 (1-{len(conversation_images)})")
            image = conversation_images[image - 1]
        
        # 执行生成
        result = tool.generate(image, item_name, anomaly_type)
        return result
        
    except Exception as e:
        error_msg = f"提示词生成失败: {str(e)}"
        print(f"[Prompt Gen] 错误: {error_msg}")
        import traceback
        traceback.print_exc()
        # 返回默认prompt
        return f"Using the provided image of a {item_name}, modify it to introduce a realistic {anomaly_type} anomaly while keeping the rest of the image unchanged."

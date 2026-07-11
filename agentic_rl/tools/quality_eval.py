"""
质量评估工具类
使用Gemini API评估异常图像的质量和位置合理性
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
        "model_eval": "gemini-3.1-pro-preview",  # 质量评估工具使用的模型
        "api_key": "",
        "base_url": "",
        "thinking_level": "low",
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
            print(f"[Quality Eval Tool] 从配置文件加载: {config_file}")
            default_config.update(config)
        except Exception as e:
            print(f"[Quality Eval Tool] 警告: 无法加载配置文件 {config_file}: {e}")
            print(f"[Quality Eval Tool] 使用默认配置")
    else:
        print(f"[Quality Eval Tool] 配置文件不存在: {config_file}")
        print(f"[Quality Eval Tool] 使用默认配置")
    if os.getenv("GEMINI_API_KEY"):
        default_config["api_key"] = os.environ["GEMINI_API_KEY"]
    if os.getenv("GEMINI_BASE_URL"):
        default_config["base_url"] = os.environ["GEMINI_BASE_URL"]
    if os.getenv("GEMINI_REASONING_MODEL"):
        default_config["model_eval"] = os.environ["GEMINI_REASONING_MODEL"]
    return default_config


class QualityEvalTool:
    """质量评估工具类"""
    
    def __init__(self, config_dict: dict = None):
        """
        初始化质量评估工具
        
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
        # 优先使用 model_eval，如果不存在则使用默认值
        self.model = self.config.get("model_eval",  "gemini-3.1-pro-preview")
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
                    print(f"[Quality Eval Tool] 警告: 使用 PIL 重新编码图像失败，使用原始文件: {e}")
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
    
    def evaluate(
        self,
        normal_image: Union[str, Image.Image, bytes],
        anomaly_image: Union[str, Image.Image, bytes],
        item_name: str,
        anomaly_type: str
    ) -> Dict[str, Any]:
        """
        评估异常图像的质量和位置合理性
        
        Args:
            normal_image: 正常图片（路径、PIL Image或字节数据）
            anomaly_image: 异常图像（路径、PIL Image或字节数据）
            item_name: 物品名称
            anomaly_type: 异常类型
            
        Returns:
            评估结果字典：
            {
                "pass": bool,      # 是否通过评估
                "review": str,     # 评估评论文本
                "quality_acceptable": bool,  # 质量是否可接受
                "location_reasonable": bool  # 位置是否合理
            }
        """
        print(f"[Quality Eval Tool] 评估质量和位置...")
        print(f"  - 物品名称: {item_name}")
        print(f"  - 异常类型: {anomaly_type}")
        
        prompt = f"""
        ### Role
        You are an expert in Industrial Quality Inspection and Computer Vision. Your task is to analyze a synthetic anomaly image from the MVTec AD dataset context.

        ### Inputs
        - **Normal Image:** a normal image of the object.
        - **Anomaly Image:** an image containing a manufactured object with the specified anomaly type generated from the normal image.
        - **Object Name:** {item_name}
        - **Anomaly Type:** {anomaly_type}

        ### Analysis Criteria
        Your task is to evaluate the generated anomaly strictly from two perspectives:
        **(1) anomaly location** and **(2) anomaly visual quality**.

        First, clearly understand what the specified anomaly type means for this object category, including its typical visual appearance, material interaction, and physical cause in real manufacturing scenarios.

        Then evaluate:

        **Location Reasonableness**: whether the anomaly is placed on a physically valid and semantically correct part of the object, aligned with object geometry, and not floating in the background or crossing irrelevant regions.

        **Quality Acceptability**: whether the anomaly appears realistic in texture, scale, contrast, and integration with surrounding material, without obvious artifacts or signs of artificial overlay.

        Based on your analysis, determine whether the anomaly image passes the anomaly generation criteria used in the MVTec AD benchmark.
        
        ### Output Format
        You MUST return the analysis strictly in the following JSON format. 
        Do not include any conversational text before or after the JSON.
        Use lowercase boolean values (true / false).

        {{
            "quality_acceptable": true/false,
            "location_reasonable": true/false,
            "evaluation_passed": true/false,
            "review": "A comprehensive review text summarizing the evaluation, including strengths and weaknesses of the generated anomaly."
        }}
        Set "evaluation_passed" to true only if both "quality_acceptable" and "location_reasonable" are true.

        If the evaluation is passed, indicate the positive aspects of the anomaly in the "review".

        If the evaluation is not passed, list specific and concrete problems (e.g., incorrect object part, unrealistic texture, inconsistent scale).

        The "review" field should provide a detailed, professional assessment of the anomaly quality, location, and overall realism.

        Be objective, precise, and consistent with real industrial defects and the MVTec AD dataset standards.
        """
        
        contents = []
        try:
            normal_img_data = self._load_image(normal_image)
            contents.append(self._prepare_image_content(normal_img_data))
            contents.append("The first image is a normal image of the object.")
        except Exception as e:
            print(f"警告: 无法加载正常图片: {e}")
            import traceback
            traceback.print_exc()
            return {
                "pass": False,
                "review": f"Failed to load normal image: {e}",
                "quality_acceptable": False,
                "location_reasonable": False
            }
        
        try:
            anomaly_img_data = self._load_image(anomaly_image)
            contents.append(self._prepare_image_content(anomaly_img_data))
            contents.append("The second image is an anomaly image containing a manufactured object with the specified anomaly type generated from the normal image (the first image).")
        except Exception as e:
            print(f"错误: 无法加载异常图像: {e}")
            import traceback
            traceback.print_exc()
            return {
                "pass": False,
                "review": f"Failed to load anomaly image: {e}",
                "quality_acceptable": False,
                "location_reasonable": False
            }
        
        contents.append(prompt)
        
        try:
            print(f"[Quality Eval Tool] 调用 Gemini API 进行评估...")
            print(f"  - 模型: {self.model}")
            # 强制刷新输出，确保日志及时显示
            import sys
            sys.stdout.flush()
            
            import time
            api_start_time = time.time()
            
            # 显示进度提示
            print(f"[Quality Eval Tool] 正在调用 API，请稍候...")
            timeout_seconds = self.config.get("timeout", 120)
            print(f"[Quality Eval Tool] 超时设置: {timeout_seconds}秒")
            sys.stdout.flush()
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level)
                ),
            )
            
            api_duration = time.time() - api_start_time
            print(f"[Quality Eval Tool] API响应接收成功，耗时: {api_duration:.2f}秒")
            sys.stdout.flush()
            
            response_text = response.text
            print(f"[Quality Eval Tool] 响应文本长度: {len(response_text)} 字符")
            print(f"[Quality Eval Tool] 响应文本前200字符: {response_text[:200]}")
            sys.stdout.flush()
            
            # 解析JSON响应
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                print(f"[Quality Eval Tool] JSON解析成功")
            else:
                # 简单解析
                print(f"[Quality Eval Tool] 未找到JSON格式，使用文本解析")
                result = {
                    "quality_acceptable": "true" in response_text.lower() or "pass" in response_text.lower(),
                    "location_reasonable": "true" in response_text.lower() or "pass" in response_text.lower(),
                    "evaluation_passed": False,
                    "review": response_text[:500] if len(response_text) > 500 else response_text
                }
                result["evaluation_passed"] = result["quality_acceptable"] and result["location_reasonable"]
            
            # 构建返回结果
            evaluation_passed = result.get("evaluation_passed", False)
            review = result.get("review", "")
            
            # 如果没有review，生成默认review
            if not review:
                review = "Evaluation completed. Quality acceptable: {}, Location reasonable: {}.".format(
                    result.get("quality_acceptable", False),
                    result.get("location_reasonable", False)
                )
            
            print(f"[Quality Eval Tool] 评估结果: pass={evaluation_passed}")
            sys.stdout.flush()
            
            return {
                "pass": evaluation_passed,
                "review": review,
                "quality_acceptable": result.get("quality_acceptable", False),
                "location_reasonable": result.get("location_reasonable", False),
                "status": "success"
            }
            
        except Exception as e:
            error_msg = f"API调用失败: {str(e)}"
            print(f"[Quality Eval Tool] 错误: {error_msg}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            return {
                "pass": False,
                "review": error_msg,
                "quality_acceptable": False,
                "location_reasonable": False,
                "status": "error"
            }


# 全局工具实例（单例模式）
_quality_eval_tool = None


def get_quality_eval_tool() -> QualityEvalTool:
    """获取质量评估工具实例（单例模式）"""
    global _quality_eval_tool
    if _quality_eval_tool is None:
        _quality_eval_tool = QualityEvalTool()
    return _quality_eval_tool


def quality_eval(
    normal_image: Union[str, Image.Image, bytes, int],
    anomaly_image: Union[str, Image.Image, bytes, int],
    item_name: str,
    anomaly_type: str,
    conversation_images: Optional[list] = None
) -> Dict[str, Any]:
    """
    质量评估函数（工具接口）
    
    这是agent调用的主要接口函数
    
    Args:
        normal_image: 正常图像，可以是：
            - 整数索引（1-based，指向conversation_images中的图像）
            - 图像路径（字符串）
            - PIL Image 对象
            - 字节数据
        anomaly_image: 异常图像，可以是：
            - 整数索引（1-based，指向conversation_images中的图像）
            - 图像路径（字符串）
            - PIL Image 对象
            - 字节数据
        item_name: 物品名称
        anomaly_type: 异常类型
        conversation_images: 对话历史中的图像列表（当使用索引时必需）
        
        Returns:
        包含评估结果的字典：
        {
            "pass": bool,      # 是否通过评估
            "review": str,     # 评估评论文本
            "quality_acceptable": bool,  # 质量是否可接受
            "location_reasonable": bool  # 位置是否合理
        }
    """
    try:
        tool = get_quality_eval_tool()
        
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
            anomaly_image = conversation_images[anomaly_image - 1]
        
        # 执行评估
        result = tool.evaluate(normal_image, anomaly_image, item_name, anomaly_type)
        return result
        
    except Exception as e:
        error_msg = f"质量评估失败: {str(e)}"
        print(f"[Quality Eval] 错误: {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "pass": False,
            "review": error_msg,
            "quality_acceptable": False,
            "location_reasonable": False
        }

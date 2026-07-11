"""
工业异常合成 Agent 推理脚本
基于 vLLM 进行多轮对话推理，调用工具完成异常图像合成任务

使用方法:
    python anomaly_synthesis_inference.py \
        --model_path /path/to/model \
        --test_data_path /path/to/mvtec_eval \
        --save_dir /path/to/output \
        --save_name results \
        --max_iterations 10 \
        --max_new_tokens 2048

工具说明:
    - image_gen: 根据原图和提示词生成异常图像
    - quality_eval: 评估生成的异常图像质量
    - knowledge_retrieval: 检索异常类型的专业知识
    - mask_gen: 生成异常区域的掩码（仅在quality_eval通过后调用）

数据格式:
    目录结构: {test_data_path}/{item_name}/{anomaly_type}/ori/{image_file}
    例如: /path/to/mvtec_eval/screw/manipulated_front/ori/0.jpg
    代码会自动从路径中提取item_name和anomaly_type

输出格式:
    结果保存为JSONL格式，每行包含:
    {
        "images": [...],
        "conversation_images": [...],  # 实际生成的图像路径列表
        "item_name": "...",
        "anomaly_type": "...",
        "answer": {...},  # 最终答案
        "messages": [...],  # 完整对话历史
        "status": "success" | "incomplete" | "error:...",
        "iterations": N
    }
"""
import argparse
import base64
import copy
import json
import os
import re
from dataclasses import asdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

from PIL import Image
from tqdm import tqdm
from vllm import LLM, EngineArgs, SamplingParams

from qwen_vl_utils import smart_resize

current_dir = Path(__file__).resolve().parent
agent_dir = current_dir.parent

# 导入工具
from agentic_rl.tools.image_gen import image_gen
from agentic_rl.tools.quality_eval import quality_eval
from agentic_rl.tools.knowledge_retrieval import knowledge_retrieval
from agentic_rl.tools.mask_gen import mask_gen


def load_system_prompt(prompt_path: str) -> str:
    """加载系统提示词"""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_user_prompt_template(prompt_path: str) -> str:
    """加载用户提示词模板"""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def format_user_prompt(template: str, item_name: str, anomaly_type: str) -> str:
    """格式化用户提示词"""
    return template.format(item_name=item_name, anomaly_type=anomaly_type)


def resize_image(original_image, factor=28, min_pixels=4*28*28, max_pixels=3840*3840):
    """
    Resize an image or image size tuple to meet pixel count constraints while preserving aspect ratio.
    """
    if isinstance(original_image, Image.Image):
        original_width, original_height = original_image.size
        new_height, new_width = smart_resize(
            height=original_height,
            width=original_width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        resized_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image
    elif isinstance(original_image, tuple):
        original_width, original_height = original_image[0], original_image[1]
        new_height, new_width = smart_resize(
            height=original_height,
            width=original_width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        return (new_width, new_height)


def encode_pil_image_to_base64(pil_image):
    """将PIL图像编码为base64字符串"""
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def load_model(model_path, max_model_len=16384, gpu_memory_utilization=0.8, tensor_parallel_size=1):
    """
    加载vLLM模型
    
    Args:
        model_path: 模型路径
        max_model_len: 最大序列长度（降低以节省KV cache内存）
        gpu_memory_utilization: GPU内存利用率（降低以在多进程模式下避免冲突）
        tensor_parallel_size: 张量并行大小（1表示不使用张量并行）
    """
    engine_args = EngineArgs(
        model=model_path,
        max_model_len=max_model_len,
        limit_mm_per_prompt={"image": 6},
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        # trust_remote_code=True,  # Qwen3-VL需要信任远程代码
        # disable_mm_preprocessor_cache=True,  # 某些vLLM版本不支持此参数，已注释
    )
    engine_args = asdict(engine_args)
    llm = LLM(**engine_args)
    return llm


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    解析文本中的工具调用
    
    Returns:
        工具调用列表，每个元素包含 name 和 arguments
    """
    tool_calls = []
    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    
    matches = re.finditer(tool_call_pattern, text, re.DOTALL)
    for match in matches:
        try:
            tool_call_content = match.group(1).strip()
            tool_call_data = json.loads(tool_call_content)
            tool_calls.append(tool_call_data)
        except json.JSONDecodeError as e:
            print(f"[WARNING] 无法解析工具调用 JSON: {e}")
            print(f"  内容: {tool_call_content[:200]}")
    
    return tool_calls


def parse_answer(text: str) -> Optional[Dict[str, Any]]:
    """
    解析最终答案
    
    Returns:
        答案字典，如果未找到则返回None
    """
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.search(answer_pattern, text, re.DOTALL)
    
    if match:
        try:
            answer_content = match.group(1).strip()
            answer_data = json.loads(answer_content)
            return answer_data
        except json.JSONDecodeError as e:
            print(f"[WARNING] 无法解析答案 JSON: {e}")
            print(f"  内容: {answer_content[:200]}")
    
    return None


def execute_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    conversation_images: List[str],
    output_base_dir: str
) -> Dict[str, Any]:
    """
    执行工具调用
    
    Args:
        tool_name: 工具名称
        arguments: 工具参数
        conversation_images: 对话历史中的图像路径列表
        
    Returns:
        工具执行结果
    """
    print(f"[Tool Execution] 调用工具: {tool_name}")
    print(f"  参数: {arguments}")
    
    try:
        if tool_name == "image_gen":
            target_image = arguments.get("target_image")
            prompt = arguments.get("prompt")
            result = image_gen(target_image, prompt, conversation_images, output_base_dir=output_base_dir)
            return result
        
        elif tool_name == "quality_eval":
            anomaly_image = arguments.get("anomaly_image")
            # 兼容item_name和object_name两种参数名
            item_name = arguments.get("item_name") or arguments.get("object_name")
            anomaly_type = arguments.get("anomaly_type")
            
            # 验证异常类型是否匹配（从数据中获取正确的异常类型）
            # 注意：这里需要从外部传入正确的anomaly_type，但由于execute_tool_call是独立函数，
            # 我们无法直接访问，所以这个验证会在调用时通过警告来提醒
            # 实际验证应该在模型调用前进行，这里只做日志记录
            
            # quality_eval需要normal_image，默认使用第一张图像（索引1）
            normal_image = 1
            result = quality_eval(normal_image, anomaly_image, item_name, anomaly_type, conversation_images)
            
            # 在结果中添加明确的pass状态标记，确保模型能清楚看到
            if "pass" in result:
                pass_status = result["pass"]
                print(f"[Tool Result] quality_eval 返回 pass: {pass_status}")
                if not pass_status:
                    print(f"[Tool Result] ⚠️  质量评估未通过，模型必须根据review进行改进")
            
            return result
        
        elif tool_name == "knowledge_retrieval":
            # 兼容item_name和object_name两种参数名
            item_name = arguments.get("item_name") or arguments.get("object_name")
            anomaly_type = arguments.get("anomaly_type")
            result = knowledge_retrieval(item_name, anomaly_type)
            return result
        
        elif tool_name == "mask_gen":
            anomaly_image = arguments.get("anomaly_image")
            # mask_gen需要normal_image，默认使用第一张图像（索引1）
            normal_image = 1
            original_image_path = conversation_images[0] if conversation_images else None
            result = mask_gen(normal_image, anomaly_image, conversation_images, original_image_path=original_image_path, output_base_dir=output_base_dir)
            return result
        
        else:
            return {
                "status": "error",
                "error": f"未知的工具: {tool_name}"
            }
    
    except Exception as e:
        error_msg = f"工具执行失败: {str(e)}"
        print(f"[Tool Execution] 错误: {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": error_msg
        }


def format_tool_response(result: Dict[str, Any]) -> str:
    """
    格式化工具响应为JSON字符串
    
    Args:
        result: 工具执行结果
        
    Returns:
        JSON字符串
    """
    # 移除PIL Image对象（如果有），因为无法序列化
    formatted_result = {}
    for key, value in result.items():
        if isinstance(value, Image.Image):
            # 如果是PIL Image，跳过或转换为路径
            continue
        formatted_result[key] = value
    
    return json.dumps(formatted_result, ensure_ascii=False)


def load_image_and_encode(image_path: str, min_pixels: int = 4*28*28) -> tuple:
    """
    加载图像并编码为base64
    
    Returns:
        (base64_string, PIL.Image) 元组
    """
    image = Image.open(image_path).convert('RGB')
    resized_image = resize_image(image, min_pixels=min_pixels)
    base64_str = encode_pil_image_to_base64(resized_image)
    return base64_str, resized_image


def scan_image_directory(dataset_dir: str) -> List[Dict[str, Any]]:
    """
    扫描数据集目录，从路径中提取item_name和anomaly_type
    
    目录结构: dataset_dir/{item_name}/{anomaly_type}/ori/{image_file}
    例如: /path/to/mvtec_eval/screw/manipulated_front/ori/0.jpg
    
    Args:
        dataset_dir: 数据集根目录路径
        
    Returns:
        测试数据列表，每个元素包含:
        {
            "image_path": str,
            "item_name": str,
            "anomaly_type": str
        }
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_dir}")
    
    test_data = []
    supported_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    # 遍历所有item_name目录
    for item_dir in dataset_path.iterdir():
        if not item_dir.is_dir():
            continue
        
        item_name = item_dir.name
        
        # 遍历所有anomaly_type目录
        for anomaly_dir in item_dir.iterdir():
            if not anomaly_dir.is_dir():
                continue
            
            anomaly_type = anomaly_dir.name
            
            # 查找ori目录
            ori_dir = anomaly_dir / "ori"
            if not ori_dir.exists() or not ori_dir.is_dir():
                print(f"[WARNING] 跳过 {item_name}/{anomaly_type}，未找到ori目录")
                continue
            
            # 扫描ori目录下的所有图片文件
            for image_file in sorted(ori_dir.iterdir()):
                if image_file.is_file() and image_file.suffix in supported_extensions:
                    test_data.append({
                        "image_path": str(image_file),
                        "item_name": item_name,
                        "anomaly_type": anomaly_type
                    })
    
    print(f"[INFO] 从目录 {dataset_dir} 扫描到 {len(test_data)} 个图片文件")
    return test_data


def process_single_sample(
    data_with_idx: tuple,
    model_path: str,
    args_dict: dict,
    system_prompt: str,
    user_prompt_template: str,
    min_pixels: int,
    output_base_dir: str,
    llm_instance: Optional[LLM] = None,
    gpu_id: Optional[int] = None,
    max_model_len: int = 16384,
    gpu_memory_utilization: float = 0.8
) -> Dict[str, Any]:
    """
    处理单个样本（用于并行处理）
    
    Args:
        data_with_idx: (data_idx, data) 元组
        model_path: 模型路径（仅在llm_instance为None时使用）
        args_dict: 参数字典（因为multiprocessing不能传递argparse.Namespace）
        system_prompt: 系统提示词
        user_prompt_template: 用户提示词模板
        min_pixels: 最小像素数
        output_base_dir: 输出基础目录
        llm_instance: LLM实例（多线程模式下可以共享，多进程模式下必须为None）
        
    Returns:
        结果字典
    """
    data_idx, data = data_with_idx
    
    # 如果指定了GPU ID，设置CUDA_VISIBLE_DEVICES（多进程模式下为每个worker分配独立GPU）
    original_cuda_visible = None
    if gpu_id is not None:
        import os
        original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"[Sample {data_idx}] 使用 GPU {gpu_id}")
    
    # 如果提供了LLM实例，使用它（多线程模式）
    # 否则加载新模型（多进程模式）
    if llm_instance is not None:
        llm = llm_instance
    else:
        # 在每个worker中加载模型（多进程模式）
        # 注意：在多进程模式下，降低max_model_len和gpu_memory_utilization以避免内存冲突
        try:
            llm = load_model(
                model_path, 
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=1  # 每个worker使用单个GPU
            )
        except Exception as e:
            return {
                "images": [data.get("image_path", "")],
                "item_name": data.get("item_name"),
                "anomaly_type": data.get("anomaly_type"),
                "messages": [],
                "status": f"error: 模型加载失败: {e}",
                "iterations": 0
            }
    
    sampling_params = SamplingParams(
        max_tokens=args_dict.get("max_new_tokens", 4096),
        temperature=0
    )
    max_iterations = args_dict.get("max_iterations", 10)
    
    try:
        # 从目录扫描的数据格式中获取信息
        normal_image_path = data["image_path"]
        item_name = data.get("item_name")
        anomaly_type = data.get("anomaly_type")
        images = [normal_image_path]
        
        if not item_name or not anomaly_type:
            print(f"[WARNING] 样本 {data_idx} 缺少item_name或anomaly_type，跳过")
            return {
                "images": images,
                "item_name": item_name,
                "anomaly_type": anomaly_type,
                "messages": [],
                "status": "error: 缺少item_name或anomaly_type",
                "iterations": 0
            }
        
        if not normal_image_path or not os.path.exists(normal_image_path):
            print(f"[ERROR] 样本 {data_idx} 图像文件不存在: {normal_image_path}")
            return {
                "images": images,
                "item_name": item_name,
                "anomaly_type": anomaly_type,
                "messages": [],
                "status": f"error: 图像文件不存在: {normal_image_path}",
                "iterations": 0
            }
        
        print(f"\n[Sample {data_idx}] item_name: {item_name}, anomaly_type: {anomaly_type}, image: {normal_image_path}")
        base64_image, resized_image = load_image_and_encode(normal_image_path, min_pixels)
        
        # 构建用户提示词
        user_prompt = format_user_prompt(user_prompt_template, item_name, anomaly_type)
        
        # 初始化对话消息
        chat_message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        
        # 管理对话历史中的图像路径
        conversation_images = [normal_image_path]  # 第一张是正常图像
        
        iteration = 0
        answer = None
        
        while iteration < max_iterations:
            print(f"\n[Sample {data_idx}][Iteration {iteration}]")
            
            # 调用模型
            output = llm.chat(chat_message, sampling_params)
            generated_text = output[0].outputs[0].text
            
            print(f"[Sample {data_idx}][Model Output] {generated_text[:200]}...")
            
            # 检查是否有最终答案
            answer = parse_answer(generated_text)
            if answer:
                print(f"[Sample {data_idx}][Answer Found] {answer}")
                chat_message.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": generated_text}]
                })
                break
            
            # 解析工具调用
            tool_calls = parse_tool_calls(generated_text)
            
            if not tool_calls:
                print(f"[Sample {data_idx}][WARNING] 未找到工具调用，结束对话")
                chat_message.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": generated_text}]
                })
                break
            
            # 添加assistant消息
            chat_message.append({
                "role": "assistant",
                "content": [{"type": "text", "text": generated_text}]
            })
            
            # 执行每个工具调用
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("arguments", {})
                
                # 验证工具调用中的anomaly_type是否与任务指定的匹配
                if tool_name in ["quality_eval", "knowledge_retrieval"]:
                    called_anomaly_type = tool_args.get("anomaly_type")
                    if called_anomaly_type and called_anomaly_type != anomaly_type:
                        print(f"[Sample {data_idx}][WARNING] 工具 {tool_name} 调用时使用了错误的异常类型: '{called_anomaly_type}'")
                        print(f"  正确的异常类型应该是: '{anomaly_type}'")
                        print(f"  这可能导致工具调用失败或返回错误结果")
                
                # 执行工具
                tool_result = execute_tool_call(tool_name, tool_args, conversation_images, output_base_dir)
                
                # 处理工具结果
                if tool_name == "image_gen" and tool_result.get("status") == "success":
                    # 图像生成成功，添加新图像到对话历史
                    new_image_path = tool_result.get("image_url")
                    if new_image_path and os.path.exists(new_image_path):
                        conversation_images.append(new_image_path)
                        print(f"[Sample {data_idx}][Image Added] 新图像已添加到对话历史 (索引 {len(conversation_images)}): {new_image_path}")
                        # 将image_url替换为占位符（符合trajectory格式）
                        tool_result["image_url"] = "<image>"
                
                # 格式化工具响应（移除image_base64等临时字段）
                tool_response_text = format_tool_response(tool_result)
                
                # 添加工具响应到对话
                chat_message.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"[Tool Response from {tool_name}]\n{tool_response_text}"
                        }
                    ]
                })
            
            iteration += 1
        
        # 保存结果
        save_info = {
            "images": images,
            "item_name": item_name,
            "anomaly_type": anomaly_type,
            "conversation_images": conversation_images,
            "answer": answer,
            "messages": chat_message,
            "status": "success" if answer else "incomplete",
            "iterations": iteration
        }
        
        return save_info
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Sample {data_idx}][ERROR] 处理样本时出错: {e}")
        
        # 获取错误时的基本信息
        error_images = [data.get("image_path", "")]
        error_item_name = data.get("item_name")
        error_anomaly_type = data.get("anomaly_type")
        
        return {
            "images": error_images,
            "item_name": error_item_name,
            "anomaly_type": error_anomaly_type,
            "messages": [],
            "status": f"error: {e}",
            "iterations": 0
        }
    finally:
        # 清理模型资源（仅在多进程模式下，且不是共享实例时）
        if llm_instance is None:
            try:
                del llm
            except:
                pass
        
        # 恢复CUDA_VISIBLE_DEVICES（如果修改过）
        if gpu_id is not None:
            import os
            if original_cuda_visible is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
            elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                del os.environ['CUDA_VISIBLE_DEVICES']


def run_inference_loop(
    llm: LLM,
    test_data: List[Dict[str, Any]],
    args,
    system_prompt: str,
    user_prompt_template: str,
    min_pixels: int,
    output_base_dir: str,
    num_workers: int = 1,
    use_multiprocessing: bool = True,
    max_model_len: int = 16384,
    gpu_memory_utilization: float = 0.8
) -> List[Dict[str, Any]]:
    """
    运行推理循环（支持并行处理）
    
    Parameters:
        llm: vLLM模型实例（仅在单线程模式下使用）
        test_data: 测试数据列表
        args: 命令行参数
        system_prompt: 系统提示词
        user_prompt_template: 用户提示词模板
        min_pixels: 最小像素数
        output_base_dir: 输出基础目录
        num_workers: 并行worker数量
        use_multiprocessing: 是否使用多进程（True）或多线程（False）
        
    Returns:
        结果列表
    """
    # 将args转换为字典（用于multiprocessing）
    args_dict = {
        "max_new_tokens": args.max_new_tokens,
        "max_iterations": args.max_iterations,
    }
    
    # 准备数据（添加索引）
    data_with_indices = [(idx, data) for idx, data in enumerate(test_data)]
    
    # 如果只有一个worker，使用串行处理（避免模型加载开销）
    if num_workers == 1:
        print("[INFO] 使用串行处理模式")
        results = []
        for data_with_idx in tqdm(data_with_indices, desc="Processing samples"):
            result = process_single_sample(
                data_with_idx,
                args.model_path,
                args_dict,
                system_prompt,
                user_prompt_template,
                min_pixels,
                output_base_dir
            )
            results.append(result)
        return results
    
    # 并行处理
    print(f"[INFO] 使用并行处理模式: {num_workers} workers, {'多进程' if use_multiprocessing else '多线程'}")
    
    # 检测可用GPU数量
    try:
        import torch
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"[INFO] 检测到 {num_gpus} 个GPU")
    except:
        num_gpus = 0
        print(f"[WARNING] 无法检测GPU数量，假设为 {num_gpus}")
    
    results = []
    executor_class = ProcessPoolExecutor if use_multiprocessing else ThreadPoolExecutor
    
    # 多线程模式下可以共享LLM实例
    llm_for_threads = llm if not use_multiprocessing else None
    
    # 多进程模式下，为每个worker分配GPU（循环分配）
    if use_multiprocessing and num_gpus > 0:
        if num_workers > num_gpus:
            print(f"[WARNING] worker数量({num_workers})大于GPU数量({num_gpus})")
            print(f"[WARNING] 建议设置 --num_workers {num_gpus} 或更少")
    
    with executor_class(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_idx = {}
        for i, data_with_idx in enumerate(data_with_indices):
            # 为每个worker分配GPU（循环分配）
            gpu_id = i % num_gpus if (use_multiprocessing and num_gpus > 0) else None
            
            future = executor.submit(
                process_single_sample,
                data_with_idx,
                args.model_path,
                args_dict,
                system_prompt,
                user_prompt_template,
                min_pixels,
                output_base_dir,
                llm_instance=llm_for_threads,
                gpu_id=gpu_id,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization
            )
            future_to_idx[future] = data_with_idx[0]
        
        # 使用tqdm显示进度
        with tqdm(total=len(test_data), desc="Processing samples") as pbar:
            for future in as_completed(future_to_idx):
                data_idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"[ERROR] 样本 {data_idx} 处理失败: {e}")
                    results.append({
                        "images": [test_data[data_idx].get("image_path", "")],
                        "item_name": test_data[data_idx].get("item_name"),
                        "anomaly_type": test_data[data_idx].get("anomaly_type"),
                        "messages": [],
                        "status": f"error: {e}",
                        "iterations": 0
                    })
                finally:
                    pbar.update(1)
    
    # 按原始顺序排序结果（通过索引）
    # 创建一个索引映射以便排序
    index_map = {idx: i for i, (idx, _) in enumerate(data_with_indices)}
    results_with_indices = [(index_map.get(i, -1), r) for i, r in enumerate(results)]
    results_with_indices.sort(key=lambda x: x[0])
    results = [r for _, r in results_with_indices]
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="工业异常合成 Agent 推理脚本")
    
    parser.add_argument("--model_path", type=str, required=True, help="Qwen3-VL base or trained checkpoint path")
    parser.add_argument("--test_data_path", type=str, required=True,
                       help="Evaluation directory containing category/anomaly/ori images")
    parser.add_argument("--system_prompt_path", type=str, 
                       default=str(agent_dir / "prompts" / "system.md"), help="系统提示词文件路径")
    parser.add_argument("--user_prompt_path", type=str,
                       default=str(agent_dir / "prompts" / "user.md"), help="用户提示词模板文件路径")
    parser.add_argument("--save_dir", type=str, default="outputs/eval", help="结果保存目录")
    parser.add_argument("--save_name", type=str, default="eval_results", help="结果文件名")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="最大生成token数")
    parser.add_argument("--max_iterations", type=int, default=10, help="最大迭代次数")
    parser.add_argument("--min_resolution", type=int, default=112, help="最小分辨率")
    parser.add_argument("--num_workers", type=int, default=4, help="并行worker数量（1表示串行处理）")
    parser.add_argument("--max_model_len", type=int, default=16384, 
                       help="最大序列长度（降低以节省KV cache内存，默认16384，原为32768）")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                       help="GPU内存利用率（0-1，默认0.8，多进程模式下建议降低）")
    parser.add_argument("--use_multiprocessing", action="store_true", default=None, 
                       help="使用多进程（默认None自动检测：多GPU环境自动使用多进程，单GPU使用多线程）。注意：多进程模式下每个worker都会加载模型，占用更多GPU内存")
    
    args = parser.parse_args()
    
    # 自动检测GPU数量并决定是否使用多进程
    if args.use_multiprocessing is None:
        try:
            import torch
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if num_gpus > 1 and args.num_workers > 1:
                args.use_multiprocessing = True
                print(f"[INFO] 检测到 {num_gpus} 个GPU，自动启用多进程模式")
            else:
                args.use_multiprocessing = False
                if num_gpus > 1:
                    print(f"[INFO] 检测到 {num_gpus} 个GPU，但worker数量为1，使用多线程模式")
                else:
                    print(f"[INFO] 检测到 {num_gpus} 个GPU，使用多线程模式")
        except:
            args.use_multiprocessing = False
            print(f"[WARNING] 无法检测GPU数量，默认使用多线程模式")
    
    # 解析路径
    agent_dir = Path(__file__).resolve().parent.parent
    system_prompt_path = args.system_prompt_path
    if not os.path.isabs(system_prompt_path):
        system_prompt_path = agent_dir / system_prompt_path
    
    user_prompt_path = args.user_prompt_path
    if not os.path.isabs(user_prompt_path):
        user_prompt_path = agent_dir / user_prompt_path
    
    # 加载提示词
    print(f"[INFO] 加载系统提示词: {system_prompt_path}")
    system_prompt = load_system_prompt(str(system_prompt_path))
    
    print(f"[INFO] 加载用户提示词模板: {user_prompt_path}")
    user_prompt_template = load_user_prompt_template(str(user_prompt_path))
    
    # 加载测试数据（扫描目录）
    print(f"[INFO] 扫描测试数据目录: {args.test_data_path}")
    test_data_path = Path(args.test_data_path)
    
    if not test_data_path.exists():
        raise FileNotFoundError(f"测试数据目录不存在: {args.test_data_path}")
    
    if not test_data_path.is_dir():
        raise ValueError(f"测试数据路径必须是目录: {args.test_data_path}")
    
    test_data = scan_image_directory(str(test_data_path))
    
    if not test_data:
        raise ValueError(f"未找到任何测试数据，请检查目录结构: {args.test_data_path}")
    
    print(f"[INFO] 总共扫描到 {len(test_data)} 个测试样本")
    
    # 计算最小像素数
    min_pixels = args.min_resolution**2 if args.min_resolution != 0 else 4*28*28
    print(f"[INFO] min_pixels: {min_pixels}")
    
    # 创建输出目录：mvtec_eval_{模型名}_{时间戳}
    from datetime import datetime
    model_path = Path(args.model_path)
    if model_path.name:
        model_name = model_path.name
    elif model_path.parent.name:
        model_name = model_path.parent.name
    else:
        model_name = os.path.basename(args.model_path.rstrip('/'))
    model_name = re.sub(r'[^\w\-_]', '_', model_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 输出目录基于test_data_path的父目录
    test_data_parent = Path(args.test_data_path).parent
    output_base_dir = test_data_parent / f"mvtec_eval_{model_name}_{timestamp}"
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"[INFO] 输出目录: {output_base_dir}")
    
    # 加载模型（仅在串行模式下需要，并行模式下每个worker会自己加载）
    llm = None
    if args.num_workers == 1:
        print(f"[INFO] 加载模型: {args.model_path}")
        llm = load_model(
            args.model_path,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    else:
        print(f"[INFO] 并行模式：每个worker将独立加载模型")
        print(f"[INFO] 配置: max_model_len={args.max_model_len}, gpu_memory_utilization={args.gpu_memory_utilization}")
        if args.use_multiprocessing:
            print(f"[WARNING] 多进程模式下，每个worker都会加载模型，请确保GPU内存充足")
            print(f"[WARNING] 建议：如果GPU内存不足，使用 --num_workers 1 或减少worker数量")
    
    # 运行推理
    print(f"[INFO] 开始推理...")
    results = run_inference_loop(
        llm, test_data, args, system_prompt, user_prompt_template, min_pixels, 
        str(output_base_dir), num_workers=args.num_workers, 
        use_multiprocessing=args.use_multiprocessing,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # 保存结果
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 从模型路径中提取模型名称
    model_path = Path(args.model_path)
    # 优先使用路径的最后一部分作为模型名称，如果为空则使用倒数第二部分
    if model_path.name:
        model_name = model_path.name
    elif model_path.parent.name:
        model_name = model_path.parent.name
    else:
        # 如果都为空，使用路径的basename
        model_name = os.path.basename(args.model_path.rstrip('/'))
    
    # 清理模型名称中的特殊字符，替换为下划线
    model_name = re.sub(r'[^\w\-_]', '_', model_name)
    
    # 生成时间戳（格式：YYYYMMDD_HHMMSS）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 生成文件名：{save_name}_{model_name}_{timestamp}.json
    save_file_path = os.path.join(args.save_dir, f"{args.save_name}_{model_name}_{timestamp}.json")
    
    print(f"[INFO] 生成结果文件名: {os.path.basename(save_file_path)}")
    print(f"[INFO]   - 模型名称: {model_name}")
    print(f"[INFO]   - 时间戳: {timestamp}")
    
    # 清理所有结果，准备保存为JSON数组格式
    clean_results = []
    for result in results:
        # 清理无法序列化的对象
        clean_result = {}
        for key, value in result.items():
            if key == "messages":
                # 清理messages中的图像数据
                clean_messages = []
                for msg in value:
                    clean_msg = copy.deepcopy(msg)
                    if "content" in clean_msg:
                        if isinstance(clean_msg["content"], list):
                            clean_content = []
                            for item in clean_msg["content"]:
                                if isinstance(item, dict):
                                    if item.get("type") == "image_url":
                                        # 保留占位符
                                        clean_item = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<image>"}}
                                    else:
                                        clean_item = item
                                    clean_content.append(clean_item)
                                else:
                                    clean_content.append(item)
                            clean_msg["content"] = clean_content
                    clean_messages.append(clean_msg)
                clean_result[key] = clean_messages
            else:
                clean_result[key] = value
        
        clean_results.append(clean_result)
    
    # 保存为JSON格式（数组）
    with open(save_file_path, 'w', encoding='utf-8') as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] 结果已保存到: {save_file_path}")
    print(f"[INFO] 成功处理: {sum(1 for r in results if r.get('status') == 'success')} 个样本")
    print(f"[INFO] 失败/未完成: {sum(1 for r in results if r.get('status') != 'success')} 个样本")

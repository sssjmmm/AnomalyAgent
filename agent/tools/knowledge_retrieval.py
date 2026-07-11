"""
知识检索工具类
用于从知识库中检索物体和异常类型的详细描述
"""
from typing import Optional, Dict, Any
import os
import json


class KnowledgeRetrievalTool:
    """知识检索工具类"""
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        """
        初始化知识检索工具
        
        Args:
            knowledge_base_path: 知识库JSON文件路径，如果为None则使用默认路径
        """
        if knowledge_base_path is None:
            # 默认路径：相对于当前文件的位置
            current_dir = os.path.dirname(os.path.abspath(__file__))
            knowledge_base_path = os.path.join(
                current_dir, 
                '..', 
                'knowledge_base.json'
            )
            knowledge_base_path = os.path.abspath(knowledge_base_path)
        
        self.knowledge_base_path = knowledge_base_path
        self._knowledge_base = None
        self._load_knowledge_base()
    
    def _load_knowledge_base(self) -> None:
        """加载知识库JSON文件"""
        if not os.path.exists(self.knowledge_base_path):
            raise FileNotFoundError(
                f"知识库文件不存在: {self.knowledge_base_path}"
            )
        
        try:
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                self._knowledge_base = json.load(f)
            print(f"[Knowledge Retrieval] 知识库已加载: {self.knowledge_base_path}")
            print(f"  - 包含 {len(self._knowledge_base)} 个物体类别")
        except json.JSONDecodeError as e:
            raise ValueError(f"知识库JSON格式错误: {e}")
        except Exception as e:
            raise RuntimeError(f"加载知识库失败: {e}")
    
    def _normalize_key(self, key: str) -> str:
        """
        规范化键名（转换为小写，去除空格）
        
        Args:
            key: 原始键名
            
        Returns:
            规范化后的键名
        """
        return key.lower().strip()
    
    def retrieve(
        self,
        item_name: str,
        anomaly_type: str
    ) -> Dict[str, Any]:
        """
        检索知识库中的描述
        
        Args:
            item_name: 物体名称（如 "bottle", "cable"）
            anomaly_type: 异常类型（如 "broken_large", "contamination"）
            
        Returns:
            包含检索结果的字典：
            {
                "context": str,      # 检索到的描述文本
                "status": str,       # "success" 或 "error"
                "error": str         # 错误信息（如果失败）
            }
        """
        print(f"[Knowledge Retrieval] 检索知识库...")
        print(f"  - 物体名称: {item_name}")
        print(f"  - 异常类型: {anomaly_type}")
        
        try:
            # 规范化键名
            normalized_item = self._normalize_key(item_name)
            normalized_anomaly = self._normalize_key(anomaly_type)
            
            # 尝试精确匹配
            if normalized_item in self._knowledge_base:
                item_data = self._knowledge_base[normalized_item]
                
                if normalized_anomaly in item_data:
                    context = item_data[normalized_anomaly]
                    print(f"[Knowledge Retrieval] 检索成功")
                    print(f"  - 描述长度: {len(context)} 字符")
                    return {
                        "context": context,
                        "status": "success"
                    }
                else:
                    # 异常类型不存在，列出可用的异常类型
                    available_anomalies = list(item_data.keys())
                    error_msg = (
                        f"异常类型 '{anomaly_type}' 不存在于物体 '{item_name}' 中。"
                        f"可用的异常类型: {', '.join(available_anomalies)}"
                    )
                    print(f"[Knowledge Retrieval] 错误: {error_msg}")
                    return {
                        "context": "",
                        "status": "error",
                        "error": error_msg
                    }
            else:
                # 物体不存在，列出可用的物体
                available_items = list(self._knowledge_base.keys())
                error_msg = (
                    f"物体 '{item_name}' 不存在于知识库中。"
                    f"可用的物体: {', '.join(available_items)}"
                )
                print(f"[Knowledge Retrieval] 错误: {error_msg}")
                return {
                    "context": "",
                    "status": "error",
                    "error": error_msg
                }
                
        except Exception as e:
            error_msg = f"检索失败: {str(e)}"
            print(f"[Knowledge Retrieval] 错误: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "context": "",
                "status": "error",
                "error": error_msg
            }
    
    def list_items(self) -> list:
        """
        列出知识库中所有可用的物体类别
        
        Returns:
            物体类别列表
        """
        return list(self._knowledge_base.keys())
    
    def list_anomalies(self, item_name: str) -> list:
        """
        列出指定物体的所有可用异常类型
        
        Args:
            item_name: 物体名称
            
        Returns:
            异常类型列表，如果物体不存在则返回空列表
        """
        normalized_item = self._normalize_key(item_name)
        if normalized_item in self._knowledge_base:
            return list(self._knowledge_base[normalized_item].keys())
        return []


# 全局工具实例（单例模式）
_knowledge_retrieval_tool = None


def get_knowledge_retrieval_tool() -> KnowledgeRetrievalTool:
    """获取知识检索工具实例（单例模式）"""
    global _knowledge_retrieval_tool
    if _knowledge_retrieval_tool is None:
        _knowledge_retrieval_tool = KnowledgeRetrievalTool()
    return _knowledge_retrieval_tool


def knowledge_retrieval(item_name: str, anomaly_type: str) -> Dict[str, Any]:
    """
    知识检索函数（工具接口）
    
    这是agent调用的主要接口函数
    
    Args:
        item_name: 物体名称（如 "bottle", "cable"）
        anomaly_type: 异常类型（如 "broken_large", "contamination"）
        
    Returns:
        包含检索结果的字典：
        {
            "context": str,      # 检索到的描述文本
            "status": str,       # "success" 或 "error"
            "error": str         # 错误信息（如果失败）
        }
    """
    tool = get_knowledge_retrieval_tool()
    return tool.retrieve(item_name, anomaly_type)

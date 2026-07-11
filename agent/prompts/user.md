Task: Evaluate and edit the provided **original image** <image> (Class: **{item_name}**) to synthesize a high-quality and physically realistic **{anomaly_type}** anomaly.

**CRITICAL REQUIREMENTS**:
- You MUST use the EXACT anomaly type **"{anomaly_type}"** specified above in ALL tool calls (knowledge_retrieval, quality_eval, etc.). Do NOT substitute it with other anomaly types like "scratch", "crack", etc., even if you think they are similar.
- **IMPORTANT**: After each tool call, you will receive a message formatted as `[Tool Response from <tool_name>]` followed by a JSON object. You MUST carefully read and parse this JSON response. The values in this JSON (especially the `score` field from `quality_eval`) are the SOURCE OF TRUTH. You MUST use the exact values from the JSON response, not your own interpretation or memory.

Reason with the information step by step, and output the final answer in the required XML format.
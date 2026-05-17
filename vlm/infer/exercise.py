import json
import re
import typing

#          提取json
def extract_json_from_text(text: str) -> dict:[str, Any] | None:
    """提取文本中的JSON对象
    输入模型返回的混乱文本
    输出：解析后的dict，或nono（如果找不到有效的json）
    """

    if not text:
        return None
        

    #尝试从花括号中提json
    #找到第一个和最后一个

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    #尝试直接解析整个文本
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    return None


    #规范化输出
    def normalize_engine_output(obj: dict[str, Any] | None, raw: str, *, error: bool) -> dict[str, Any]:
        """把模型输出规范化成统一格式
        功能：
        1.如果json——obj 是nono也就是返回失败，返回默认值
        2.如果成功了，添加额外字段（isanomaly，annomaly——type）
        3.统一返回格式"""
        #如果解析失败，返回默认值
        if not json_obj:
            json_obj = {
                "classfication" = "正常",
                "reason" = "模型未按格式输出，请检查模型状态或重试。",
            }

        #获得分类信息
        classfication = str(json_obj.get("classification") or "normal").strip()

        #判断是否异常
        #如果 classfication 不是 normal 或 non-anomaly 或 空字符串，则认为是异常
        is_abnormal = classfication.lower() not in ("normal", "non-anomaly", "", "正常")

        #构造派生字典（为了兼容旧代码）
        output["is_anomaly"] = is_abnormal
        output["anomaly_type"] = "normal" if not is_abnormal else classfication
        output["confidence"] = 1.0 if is_abnormal else 0.0
        output["reasoning"] = str(json_obj.get("reason") or "")
        output["error"] = error
        return output
    
    #第三步 清理模型输出

    def clean_model_output(text: str) -> str:
        """清理模型输出
        功能：
        1.去掉markdown、代码块、解释说明、分析过程、换行符以外的任何文字
        2.去掉<think>...</think>标签
        3.去掉```json```标签
        4.去掉``````标签
        5.去掉``````标签
        6.去掉``````标签
        7.去掉``````标签
        """
        cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()

        #第二部，去掉markdown 代码标记
        #去掉开头和结尾的反引号
        cleaned = cleaned.strip().lstrip("`").rstrip("`")

        #第三步，如果以json开头，去掉json前缀
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

        return cleaned
    
    #完整的分析流程
    def analyze_and_output_json(model_response: str)-> dict[str, Any]:
        """分析模型输出，返回结构化JSON
        功能：
        1.清理模型输出
        2.提取json
        3.规范化输出
        4.返回结构化JSON
        """
        # 第1步：清理
        cleaned = clean_model_response(model_response)
    
    # 第2步：提取JSON
        json_obj = extract_json_from_text(cleaned)
    
    # 第3步：规范化
        result = normalize_output(json_obj, cleaned, error=(json_obj is None))
    
        return result
        # ============ 用法示例 ============
    if __name__ == "__main__":
        # 情况1：正常输出
        print("="*50)
        print("情况1：正常JSON输出")
        print("="*50)
        response1 = '{"classification":"正常","reason":"画面清晰，无异常"}'
        result1 = analyze_and_output_json(response1)
        print("输入:", response1)
        print("输出:", json.dumps(result1, ensure_ascii=False, indent=2))
        
        # 情况2：有思考标签和 markdown 格式
        print("\n" + "="*50)
        print("情况2：带有思考标签和markdown格式")
        print("="*50)
        response2 = '<think>分析视频中...</think>\n```json\n{"classification":"打架","reason":"两个人在打架"}\n```'
        result2 = analyze_and_output_json(response2)
        print("输入:", response2)
        print("输出:", json.dumps(result2, ensure_ascii=False, indent=2))
        
        # 情况3：解析失败
        print("\n" + "="*50)
        print("情况3：模型输出乱码，无法解析")
        print("="*50)
        response3 = '这个画面看起来很复杂，我觉得可能有问题...'
        result3 = analyze_and_output_json(response3)
        print("输入:", response3)
        print("输出:", json.dumps(result3, ensure_ascii=False, indent=2))
        
        # 情况4：火灾检测
        print("\n" + "="*50)
        print("情况4：异常检测 - 火灾")
        print("="*50)
        response4 = '```\n{"classification":"火灾","reason":"画面中有明显火焰"}\n```'
        result4 = analyze_and_output_json(response4)
        print("输入:", response4)
        print("输出:", json.dumps(result4, ensure_ascii=False, indent=2))
        
        # 情况5：盗窃检测
        print("\n" + "="*50)
        print("情况5：异常检测 - 盗窃")
        print("="*50)
        response5 = '<think>检查中...</think>{"classification":"盗窃","reason":"检测到可疑物品转移"}'
        result5 = analyze_and_output_json(response5)
        print("输入:", response5)
        print("输出:", json.dumps(result5, ensure_ascii=False, indent=2))
        

        
import openai
from openai import OpenAI
from pydantic import BaseModel
import json
from tqdm import tqdm
from enum import Enum
import os
import math
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, f1_score, classification_report
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import bert_score
import numpy as np
import jieba
import re


def data_loading(input_file):
    with open(input_file,'r',encoding='utf-8') as file:
        datalist = []
        for line in file:
            data = json.loads(line)
            highlaw = ''.join(str(item) for item in data['high_level_laws'])
            test_data = {
                "id" : data['id'],
                "input1" : highlaw ,
                "input2" : data['content'],
                "conflict_revise" : data['golden_content'],
                "conflict_description" : data["conflict_description"]
            }
            if data['conflict_type'] == '无冲突':
                test_data['conflict_type'] = 0
            elif data['conflict_type'] == '职权或责任划分不符':
                test_data['conflict_type'] = 1
            elif data['conflict_type'] == '概念或定义范围不符':
                test_data['conflict_type'] = 2
            elif data['conflict_type'] == '处罚幅度或范围不符':
                test_data['conflict_type'] = 3
            elif data['conflict_type'] == '增设或变更适用条件':
                test_data['conflict_type'] = 4
            datalist.append(test_data)
    return datalist

def extract_conflict_json(text: str) -> dict:
    pattern = re.compile(r'(\{\s*"conflict_type".*?\})', flags=re.DOTALL)
    match = pattern.search(text)
    if not match:
        raise ValueError("未能在文本中找到以 'conflict_type' 开头的 JSON 对象")
    
    json_str = match.group(1)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"解析 JSON 失败：{e}")

def get_answer(datalist,output_file):
    
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="your_url",
    )
    class ConflictType(str, Enum):
        quanze = "职权或责任划分不符"
        gainian = "概念或定义范围不符"
        chufa = "处罚幅度或范围不符"
        tiaojian = "增设或变更适用条件"
        no = "无冲突"
        
    class ConflictDescription(BaseModel):
        id: str
        conflict_type: ConflictType
        conflict_description: str
        conflict_revise: str
        
    results_list = []
    
    with open(output_file,'a',encoding='utf-8') as file:
        for data in tqdm(datalist):
            input1 = data['input1']
            input2 = data['input2']
            try:
                completion = client.chat.completions.create(
                    model="model_name",
                    messages=[
                        {
                            "role": "user",
                            "content": "请你作为一个中文语义推理专家和法律审查助手，判断句子1{input1}和句子2{input2}冲突类型是什么？ 1. 冲突类型请从职权或责任划分不符、概念或定义范围不符、处罚幅度或范围不符、增设或变更适用条件和无冲突五种类型中选择一种输出,放在conflict_type字段。2. 只输出标准、合法、单条JSON对象，不要输出多余文本，不要有注释或换行，不要输出两个或多个JSON。注意，如果输出不是标准JSON，将被判为错误。不要输出任何非JSON内容".format(input1=input1,input2=input2), 
                        },
                    ],
                    response_format={"type": "json_object"},
                )
                event = completion.choices[0].message.content
                json_object = extract_conflict_json(event)
                json_object['id'] = data['id']
                type_map = {
                    "无冲突": 0,
                    "职权或责任划分不符": 1,
                    "概念或定义范围不符": 2,
                    "处罚幅度或范围不符": 3,
                    "增设或变更适用条件": 4
                }
                new_item = {
                    "id" : json_object['id'],
                    "conflict_type": type_map.get(json_object["conflict_type"], -1)
                }
                results_list.append(new_item)
                file.write(json.dumps(new_item, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"❌ 错误处理样本 id: {data.get('id', 'unknown')}，错误原因：{str(e)}")
                continue
    return results_list

def neweval_metrics(datalist, new_results):
    datalist_dict = {item['id']: item for item in datalist}
    new_results_dict = {item['id']: item for item in new_results}
    common_ids = list(set(datalist_dict.keys()) & set(new_results_dict.keys()))
    if not common_ids:
        print("没有匹配的样本id，无法评测！")
        return {}
    true_types = [datalist_dict[i]["conflict_type"] for i in common_ids]
    pred_types = [new_results_dict[i]["conflict_type"] for i in common_ids]

    type_acc = accuracy_score(true_types, pred_types)
    type_f1 = f1_score(true_types, pred_types, average='macro')
    print(f"conflict_type准确率: {type_acc:.4f}, F1: {type_f1:.4f}")


    return {
        "conflict_type_acc": type_acc,
        "conflict_type_f1": type_f1,
        
    }

if __name__ == "__main__":
    input_file = 'dataset/test.jsonl'
    output_file = '/output.jsonl'
    datalist = data_loading(input_file)
    results = get_answer(datalist,output_file)
    metrics = neweval_metrics(datalist, results)
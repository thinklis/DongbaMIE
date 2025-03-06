import os
import time
import json
import re
from tqdm import tqdm
from openai import OpenAI

import base64

# openai
os.environ["OPENAI_API_KEY"] = "xxxxxxxxxxxxxxxxxx"
os.environ["OPENAI_BASE_URL"] = "xxxxxxxxxxxxxxxxxx"

client = OpenAI()

# Function to encode the image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_gpt4o(image_path):
    """
    Call the GPT-4o API to process a single Dongba corpus.
    """
    dongba_IE_zero_shot_prompt = """
    你是一位东巴象形文字专家，需要根据提供的东巴文字图像抽取以下四个方面的结构化信息：

    1.对象(Object)：图像中涉及的主要实体。
    2.动作(Action)：图像中涉及的主要行为动作。
    3.关系(Relation)：图像中对象之间的语义关联。
    4.属性(Attribute)：图像中描述对象的特征或状态。

    注意事项：
    1.东巴文字是高度抽象的象形文字，需结合图像符号的形态特征推理语义。
    2.仅返回JSON，不添加额外解释或自然语言描述。
    3.键名必须严格符合规范（"Object", "Action", "Relation", "Attribute"）。
    4.所有值必须是数组（即使只有一个元素）。
    5.如果某个维度没有内容，返回空数组 `[]`
    6.必须使用中文进行回答。
    7.所有抽取的值不允许重复。
    8.返回的JSON数据内部必须使用双引号。

    输出的JSON格式为：
    {
        "Action": [
            {"action": "..."}
        ],
        "Object": [
            {"object": "..."}
        ],
        "Relation": [
            {"entity1": "...", "relation_type": "...", "entity2": "..."}
        ],
        "Attribute": [
            {"object": "...", "attribute": "...", "value": "..."}
        ]
    }

    输入的东巴文字图像如下: 
    """

    dongba_IE_zero_shot_prompt_Object = """
    你是一位东巴象形文字专家，需要根据提供的东巴文字图像抽取以下方面的结构化信息：

    1.对象(Object)：图像中涉及的主要实体。

    注意事项：
    1.东巴文字是高度抽象的象形文字，需结合图像符号的形态特征推理语义。
    2.仅返回JSON，不添加额外解释或自然语言描述。
    3.键名必须严格符合规范（"Object"）。
    4.所有值必须是数组（即使只有一个元素）。
    5.如果某个维度没有内容，返回空数组 `[]`
    6.必须使用中文进行回答。
    7.所有抽取的值不允许重复。
    8.返回的JSON数据内部必须使用双引号。

    输出的JSON格式为：
    {
        "Object": [
            {"object": "..."}
        ]
    }

    输入的东巴文字图像如下: 
    """
    
    dongba_IE_zero_shot_prompt_Action = """
    你是一位东巴象形文字专家，需要根据提供的东巴文字图像抽取以下方面的结构化信息：

    1.动作(Action)：图像中涉及的主要行为动作。

    注意事项：
    1.东巴文字是高度抽象的象形文字，需结合图像符号的形态特征推理语义。
    2.仅返回JSON，不添加额外解释或自然语言描述。
    3.键名必须严格符合规范（"Action"）。
    4.所有值必须是数组（即使只有一个元素）。
    5.如果某个维度没有内容，返回空数组 `[]`
    6.必须使用中文进行回答。
    7.所有抽取的值不允许重复。
    8.返回的JSON数据内部必须使用双引号。

    输出的JSON格式为：
    {
        "Action": [
            {"action": "..."}
        ]
    }

    输入的东巴文字图像如下: 
    """

    dongba_IE_zero_shot_prompt_Relation = """
    你是一位东巴象形文字专家，需要根据提供的东巴文字图像抽取以下方面的结构化信息：

    1.关系(Relation)：图像中对象之间的语义关联。

    注意事项：
    1.东巴文字是高度抽象的象形文字，需结合图像符号的形态特征推理语义。
    2.仅返回JSON，不添加额外解释或自然语言描述。
    3.键名必须严格符合规范（"Relation"）。
    4.所有值必须是数组（即使只有一个元素）。
    5.如果某个维度没有内容，返回空数组 `[]`
    6.必须使用中文进行回答。
    7.所有抽取的值不允许重复。
    8.返回的JSON数据内部必须使用双引号。

    输出的JSON格式为：
    {
        "Relation": [
            {"entity1": "...", "relation_type": "...", "entity2": "..."}
        ],
    }

    输入的东巴文字图像如下: 
    """

    dongba_IE_zero_shot_prompt_Attribute = """
    你是一位东巴象形文字专家，需要根据提供的东巴文字图像抽取以下方面的结构化信息：

    1.属性(Attribute)：图像中描述对象的特征或状态。

    注意事项：
    1.东巴文字是高度抽象的象形文字，需结合图像符号的形态特征推理语义。
    2.仅返回JSON，不添加额外解释或自然语言描述。
    3.键名必须严格符合规范（"Attribute"）。
    4.所有值必须是数组（即使只有一个元素）。
    5.如果某个维度没有内容，返回空数组 `[]`
    6.必须使用中文进行回答。
    7.所有抽取的值不允许重复。
    8.返回的JSON数据内部必须使用双引号。

    输出的JSON格式为：
    {
        "Attribute": [
            {"object": "...", "attribute": "...", "value": "..."}
        ]
    }

    输入的东巴文字图像如下: 
    """
    
    prompt = dongba_IE_zero_shot_prompt_Action
    
    # Getting the Base64 string
    base64_image = encode_image(image_path)

    while True:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
            )
            # Determine whether there is complete content
            if getattr(completion.choices[0].message, 'content', None):
                content = completion.choices[0].message.content
                return content
            else:
                print('error_wait_2s')
        except Exception as e:
            print(f"Error: {e}")
            pass
        time.sleep(2)

def dongba_IE_VQA(input_folder, output_file):
    """
    Iterate over the .jpg files in the input folder, call the GPT-4 API to process the samples one by one, and append them directly to the JSON file.
    """

    if not os.path.exists(input_folder):
        print(f"Input folder {input_folder} does not exist!")
        return
    
    jpg_files = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]
    
    if not jpg_files:
        print(f"No .jpg files were found in the input folder {input_folder}!")
        return
    
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("[\n")  # Write JSON array start symbol

    for idx, jpg_file in enumerate(tqdm(jpg_files, desc="Processing jpg Files")):
        file_path = os.path.join(input_folder, jpg_file)

        if not file_path:
            print(f"File {file_path} is empty, skipping.")
            continue

        gpt4_result = call_gpt4o(file_path)

        if gpt4_result is not None:
            # Clean up the code block markers
            match = re.search(r'\{[\s\S]*\}', gpt4_result)
            if match:
                cleaned_result = match.group(0)
            else:
                cleaned_result = None
            
            # Parsing JSON safely
            try:
                analysis_data = json.loads(cleaned_result) if cleaned_result else None
            except json.JSONDecodeError as e:
                print(f"JSON 解析失败: {e}，文件 {jpg_file} 的结果被设为 None")
                analysis_data = None

            # Process into JSON object
            json_obj = {
                "file": jpg_file,
                "analysis": analysis_data
            }

            # Append to JSON file
            with open(output_file, 'a', encoding='utf-8') as file:
                json.dump(json_obj, file, ensure_ascii=False, indent=4)
                if idx < len(jpg_files) - 1:  # If it is not the last element, add a comma to wrap the line
                    file.write(",\n")

    # Finally, append the end symbol of the JSON array
    with open(output_file, 'a', encoding='utf-8') as file:
        file.write("\n]")

    print(f"Processing completed, results saved to {output_file}")


if __name__ == "__main__":

    # sentence_image
    input_folder = f'./DongbaMIE/paragraph/paragraph_img_test/'
    output_file = f'./results/gpt4o/zero_shot/paragraph_img_test_gpt4o_IE_Action.json'

    dongba_IE_VQA(input_folder, output_file)

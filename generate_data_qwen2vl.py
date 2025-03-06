import json
import copy
from tqdm import tqdm


def save_data(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main(input_file, save_path, lmm_type, train_type, sent_or_para, pre_image_path):
    
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    objects = []
    actions = []
    relations = []
    attributes = []

    # prompt template
    object_prompt = """
你是一位东巴象形文字专家，需要根据提供的东巴文字图像抽取以下方面的结构化信息：

1.对象(Object)：图像中涉及的主要实体。

注意事项：
1.东巴文字是高度抽象的象形文字，需结合图像符号的形态特征推理语义。
2.必须使用中文进行回答。
3.所有抽取的值不允许重复。

输入的东巴文字图像如下: 
"""

    action_prompt = """
你是一位东巴象形文字专家，需要根据提供的东巴文字图像抽取以下方面的结构化信息：

1.动作(Action)：图像中涉及的主要行为动作。

注意事项：
1.东巴文字是高度抽象的象形文字，需结合图像符号的形态特征推理语义。
2.必须使用中文进行回答。
3.所有抽取的值不允许重复。

输入的东巴文字图像如下: 
"""

    relation_prompt = """
你是一位东巴象形文字专家，需要根据提供的东巴文字图像抽取以下方面的结构化信息：

1.关系(Relation)：图像中对象之间的语义关联。

注意事项：
1.东巴文字是高度抽象的象形文字，需结合图像符号的形态特征推理语义。
2.必须使用中文进行回答。
3.所有抽取的值不允许重复。

输入的东巴文字图像如下: 
"""

    attribute_prompt = """
你是一位东巴象形文字专家，需要根据提供的东巴文字图像抽取以下方面的结构化信息：

1.属性(Attribute)：图像中描述对象的特征或状态。

注意事项：
1.东巴文字是高度抽象的象形文字，需结合图像符号的形态特征推理语义。
2.必须使用中文进行回答。
3.所有抽取的值不允许重复。

输入的东巴文字图像如下: 
"""

    all_prompt = """
你是一位东巴象形文字专家，需要根据提供的东巴文字图像抽取以下方面的结构化信息：

1.对象(Object)：图像中涉及的主要实体。
2.动作(Action)：图像中涉及的主要行为动作。
3.关系(Relation)：图像中对象之间的语义关联。
4.属性(Attribute)：图像中描述对象的特征或状态。


注意事项：
1.东巴文字是高度抽象的象形文字，需结合图像符号的形态特征推理语义。
2.必须使用中文进行回答。
3.所有抽取的值不允许重复。

输入的东巴文字图像如下: 
"""

    for item in tqdm(original_data):
        image_file = item["file"].replace(".txt", ".jpg")
        base_structure = {
            "conversations": [
                {
                    "from": "human",
                    "value": ""
                },
                {
                    "from": "gpt",
                    "value": ""
                }
            ],
            "images": [pre_image_path + image_file]
        }

        object_str = "、".join([obj["object"] for obj in item["analysis"]["Object"]])
        obj = copy.deepcopy(base_structure)
        obj["conversations"][0]["value"] = "<image>" + object_prompt
        obj["conversations"][1]["value"] = "Object: " + object_str
        objects.append(obj)
        
        # Action 
        action_str = "、".join([action["action"] for action in item["analysis"]["Action"]])
        act = copy.deepcopy(base_structure)
        act["conversations"][0]["value"] = "<image>" + action_prompt
        act["conversations"][1]["value"] = "Action: " + action_str
        actions.append(act)
        
        #Relation
        relation_str = "、".join([f"{relation['entity1']} {relation['relation_type']} {relation['entity2']}" for relation in item["analysis"]["Relation"]])
        rel = copy.deepcopy(base_structure)
        rel["conversations"][0]["value"] = "<image>" + relation_prompt
        rel["conversations"][1]["value"] = "Relation: " + relation_str
        relations.append(rel)
        
        # Attribute
        attribute_str = "、".join([f"{attribute['object']} {attribute['attribute']} {attribute['value']}" for attribute in item["analysis"]["Attribute"]])
        attr = copy.deepcopy(base_structure)
        attr["conversations"][0]["value"] = "<image>" + attribute_prompt
        attr["conversations"][1]["value"] = "Attribute: " + attribute_str
        attributes.append(attr)

    
    save_data(objects, save_path + f'dongba_{sent_or_para}_{lmm_type}_object_{train_type}.json')
    save_data(actions, save_path + f'dongba_{sent_or_para}_{lmm_type}_action_{train_type}.json')
    save_data(relations, save_path + f'dongba_{sent_or_para}_{lmm_type}_relation_{train_type}.json')
    save_data(attributes, save_path + f'dongba_{sent_or_para}_{lmm_type}_attribute_{train_type}.json')
    



if __name__ == "__main__":
    
    train_type = "train"   # train, test, dev
    sent_or_para = "paragraph"  # sentece or paragraph
    
    # DongbaMIE json files path
    input_file = f"./DongbaMIE/{sent_or_para}/dongbaMIE_{sent_or_para}_{train_type}.json"
    # save files path
    save_path = f"./dongba_dataset/{sent_or_para}/qwen_vl_json/{train_type}/"
    lmm_type = "qwen_vl"  # model type
    
    # DongbaMIE image files path
    pre_image_path = f"DongbaMIE/{sent_or_para}/{sent_or_para}_img_{train_type}/"
    
    main(input_file, save_path, lmm_type, train_type, sent_or_para, pre_image_path)
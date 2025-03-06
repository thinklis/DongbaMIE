import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_filename(filename):
    return filename.rsplit('.', 1)[0]

def extract_field(data, field):
    """ Extract the data of the specified field and return a collection """
    if field == "Action":
        return set(item["action"] for item in data.get("analysis", {}).get("Action", []) if "action" in item)
    elif field == "Object":
        return set(item["object"] for item in data.get("analysis", {}).get("Object", []) if "object" in item)
    elif field == "Relation":
        return set((item["entity1"], item["relation_type"], item["entity2"]) 
                   for item in data.get("analysis", {}).get("Relation", []) 
                   if "entity1" in item and "relation_type" in item and "entity2" in item)
    elif field == "Attribute":
        return set((item["object"], item["attribute"], item["value"]) 
                   for item in data.get("analysis", {}).get("Attribute", []) 
                   if "object" in item and "attribute" in item and "value" in item)
    return set()

def calculate_metrics(true_set, pred_set):
    TP = len(true_set & pred_set)
    predicted_total = len(pred_set)
    gold_total = len(true_set)
    
    precision = TP / predicted_total if predicted_total > 0 else 0
    recall = TP / gold_total if gold_total > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def evaluate(label_path, generate_path, fields=["Action", "Object", "Relation", "Attribute"], missing_file_path="miss.txt"):
    label_data = load_json(label_path)
    generate_data = load_json(generate_path)
    
    generate_dict = {normalize_filename(item["file"]): item for item in generate_data}
    
    missing_files = []
    results = {}
    for field in fields:
        total_precision, total_recall, total_f1 = 0, 0, 0
        count = 0
        
        for label_item in label_data:
            file_key = normalize_filename(label_item["file"])
            if file_key in generate_dict:
                true_set = extract_field(label_item, field)
                pred_set = extract_field(generate_dict[file_key], field)

                # If label is empty and the corresponding field of generate is empty, skip the sample
                if not true_set and not pred_set:
                    continue  # Skip this sample and do not calculate the index of this field.

                precision, recall, f1 = calculate_metrics(true_set, pred_set)
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                count += 1
            else:
                missing_files.append(file_key)
        
        if count > 0:
            results[field] = {
                "Precision": total_precision / count,
                "Recall": total_recall / count,
                "F1": total_f1 / count
            }
        else:
            results[field] = {"Precision": 0, "Recall": 0, "F1": 0}
    
    # recording error logs
    if missing_files:
        with open(missing_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(missing_files))
    
    return results


if __name__ == "__main__":

    sent_or_para = "sentence"  # sentence or paragraph
    # fields includes the Action, Object, Relation, and Attribute fields to be calculated. It include one or more fields and is a list.
    fields=["Action", "Object", "Relation", "Attribute"]  # 
    # fields=["Action"]
    mode = "all"  # all, Action, Object, Relation, Attribute
    model = "gemini"   # qwen2_vl, gpt4o, gemini
    zero_or_few = "zero_shot"  # zero_shot or few_shot

    label_file = f"./DongbaMIE/dongbaMIE_{sent_or_para}_test.json"
    
    # model generation file path
    generate_file = f"./generate_results/{model}/{zero_or_few}/{sent_or_para}_img_test_{model}_IE_{mode}.json"
    
    # path to save metric files
    metric_file = f"./results_metrics/{model}/{sent_or_para}/{sent_or_para}_img_test_{model}_IE_{mode}_{zero_or_few}.txt"
    
    # recording error logs
    missing_file_path=f"./{sent_or_para}_img_test_{model}_IE_{mode}_{zero_or_few}_miss.txt"  # 用以记录 gpt4o或其他模型 可能出现的数据丢失 情况

    metrics = evaluate(label_file, generate_file, fields, missing_file_path)
    
    with open(metric_file, 'w', encoding='utf-8') as f:
        for field, scores in metrics.items():
            f.write(f"{field}: Precision={scores['Precision']*100:.2f}, Recall={scores['Recall']*100:.2f}, F1={scores['F1']*100:.2f}\n")

    for field, scores in metrics.items():
        print(f"{field}: Precision={scores['Precision']*100:.2f}, Recall={scores['Recall']*100:.2f}, F1={scores['F1']*100:.2f}")

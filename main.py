import json
import tqdm
from preprocess import write_file
import collections
import numpy as np 
from utils import get_labels

def trigger_classify_labels_merge(input_file_list, output_file):
    num_fold = len(input_file_list)
    labels_file_list= []
    for input_file in input_file_list:
        rows = open(input_file, encoding='utf-8').read().splitlines()
        labels_file = [json.loads(row)["labels"] for row in rows]
        labels_file_list.append(labels_file)
    
    num_samples = len(labels_file_list[0])
    labels = []
    for i in range(num_samples):
        cur_all_labels = []
        for labels_file in labels_file_list:
            cur_all_labels.extend(labels_file[i])

        cur_labels =[]
        obj = collections.Counter(cur_all_labels)
        for k,v in obj.items():
            if v>= (num_fold+1)//2:
                cur_labels.append(k)
        if cur_labels ==[]:
            # print(obj.most_common(1)[0][0])
            cur_labels.append(obj.most_common(1)[0][0])
        labels.append({"labels": cur_labels})
    
    write_file(labels, output_file)

def trigger_classify_logits_merge_and_eval(input_file_list, output_file, label_file):
    labels = get_labels(task="trigger", mode="classify")
    label_map = {i: label for i, label in enumerate(labels)}

    num_fold = len(input_file_list)
    logits_file_list= []
    for input_file in input_file_list:
        rows = open(input_file, encoding='utf-8').read().splitlines()
        logits_file = [json.loads(row)["logits"] for row in rows]
        logits_file_list.append(logits_file)
    
    threshold = 0.5
    logits = np.array(logits_file_list).mean(axis=0)
    preds = logits > threshold # 1498*65

    # 若所有类别对应的 logit 都没有超过阈值，则将 logit 最大的类别作为 label
    for i in range(preds.shape[0]):
        if sum(preds[i])==0:
            preds[i][np.argmax(logits[i])]=True

    preds_list = []
    batch_preds_list = [[] for _ in range(preds.shape[0])]

    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            if preds[i, j]:
                preds_list.append([i, label_map[j]])
                batch_preds_list[i].append(label_map[j])

    ############ labels
    label_rows = open(label_file, encoding='utf-8').read().splitlines()
    out_labels = [json.loads(row)["labels"] for row in label_rows]

    out_label_list = []

    for i, row in enumerate(label_rows):
        json_line = json.loads(row)
        for label in json_line["labels"]:
            out_label_list.append([i, label])

    print(compute_f1(preds_list, out_label_list))
    
    write_file(labels, output_file)

def eval(pred_file, label_file):
    preds_list= []
    labels_list = []
    pred_rows = open(pred_file, encoding='utf-8').read().splitlines()
    label_rows = open(label_file, encoding='utf-8').read().splitlines()
    
    for i, row in enumerate(pred_rows):
        json_line = json.loads(row)
        for label in json_line["labels"]:
            preds_list.append([i, label])
    
    for i, row in enumerate(label_rows):
        json_line = json.loads(row)
        for label in json_line["labels"]:
            labels_list.append([i, label])
    
    print(compute_f1(preds_list, labels_list ))

def compute_f1(preds_list, labels_list):
    nb_correct = 0
    for out_label in labels_list:
        # for pred in preds_list:
        #     if out_label==pred:
        #         nb_correct+=1
        if out_label in preds_list:
            nb_correct += 1
            continue
    nb_pred = len(preds_list)
    nb_true = len(labels_list)
    # print(nb_correct, nb_pred, nb_true)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0

    results = {
        "precision": p,
        "recall": r,
        "f1": f1,
    }
    return results

if __name__ == "__main__":
    # trigger_classify_labels_merge(input_file_list=[
    #     "./output/trigger_classify/0/checkpoint-best/eval_predictions.json",
    #     "./output/trigger_classify/1/checkpoint-best/eval_predictions.json",
    #     "./output/trigger_classify/2/checkpoint-best/eval_predictions.json"],
    #     output_file="./output/trigger_classify/merge/eval_predictions.json"
    # )
    # eval(pred_file="./output/trigger_classify/merge/eval_predictions_logits.json",\
    #     label_file="./data/trigger_classify/dev.json",
    #     )

    trigger_classify_logits_merge_and_eval(input_file_list=[
        "./output/trigger_classify/0/checkpoint-best/eval_logits.json",
        "./output/trigger_classify/1/checkpoint-best/eval_logits.json",
        "./output/trigger_classify/2/checkpoint-best/eval_logits.json"],
        output_file="./output/trigger_classify/merge/eval_predictions_logits.json",
        label_file="./data/trigger_classify/dev.json"
    )



        


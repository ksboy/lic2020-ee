import json
import os
from utils import get_labels, write_file
from postprocess import extract_result


def trigger_classify_file_remove_id(input_file, output_file):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        row.pop("id")
        row.pop("text")
        results.append(row)
    write_file(results,output_file)

def trigger_classify_process(input_file, output_file, is_predict=False):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        labels = []
        if is_predict: 
            results.append({"id":row["id"], "text":row["text"], "labels":labels})
            continue
        for event in row["event_list"]:
            event_type = event["event_type"]
            labels.append(event_type)
        labels = list(set(labels))
        results.append({"id":row["id"], "text":row["text"], "labels":labels})
    write_file(results,output_file)

def trigger_process(input_file, output_file, is_predict=False):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        labels = ['O']*len(row["text"])
        if is_predict: 
            results.append({"id":row["id"], "tokens":list(row["text"]), "labels":labels})
            continue
        for event in row["event_list"]:
            trigger = event["trigger"]
            event_type = event["event_type"]
            trigger_start_index = event["trigger_start_index"]
            labels[trigger_start_index]= "B-{}".format(event_type)
            for i in range(1, len(trigger)):
                labels[trigger_start_index+i]= "I-{}".format(event_type)
        results.append({"id":row["id"], "tokens":list(row["text"]), "labels":labels})
    write_file(results,output_file)

def trigger_process_binary(input_file, output_file, is_predict=False):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        start_labels = ['O']*len(row["text"])
        end_labels = ['O']*len(row["text"])
        if is_predict: 
            results.append({"id":row["id"], "tokens":list(row["text"]), "start_labels":start_labels, "end_labels":end_labels})
            continue
        for event in row["event_list"]:
            trigger = event["trigger"]
            event_type = event["event_type"]
            trigger_start_index = event["trigger_start_index"]
            trigger_end_index = trigger_start_index + len(trigger) - 1
            start_labels[trigger_start_index]= event_type
            end_labels[trigger_end_index]= event_type
        results.append({"id":row["id"], "tokens":list(row["text"]),  "start_labels":start_labels, "end_labels":end_labels})
    write_file(results,output_file)

def role_process(input_file, output_file, is_predict=False):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        labels = ['O']*len(row["text"])
        if is_predict: 
            results.append({"id":row["id"], "tokens":list(row["text"]), "labels":labels})
            continue
        for event in row["event_list"]:
            event_type = event["event_type"]
            for arg in event["arguments"]:
                role = arg['role']
                argument = arg['argument']
                argument_start_index = arg["argument_start_index"]
                labels[argument_start_index]= "B-{}".format(role)
                for i in range(1, len(argument)):
                    labels[argument_start_index+i]= "I-{}".format(role)
                if arg['alias']!=[]: print(arg['alias'])
        results.append({"id":row["id"], "tokens":list(row["text"]), "labels":labels})
    write_file(results,output_file)

def role_segment_process(input_file, output_file, is_predict=False):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    len_text = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        len_text.append(len(row["text"]))
        if len(list(row["text"]))!= len(row["text"]):
            print("list and text mismatched")
        labels = ['O']*len(row["text"])
        if is_predict: 
            results.append({"id":row["id"], "tokens":list(row["text"]), "labels":labels})
            continue
        for event in row["event_list"]:
            event_type = event["event_type"]
            trigger = event["trigger"]
            trigger_start_index = event["trigger_start_index"]
            segment_ids= [0] * len(row["text"])
            for i in range(trigger_start_index, trigger_start_index+ len(trigger) ):
                segment_ids[i] = 1

            for arg in event["arguments"]:
                role = arg['role']
                argument = arg['argument']
                argument_start_index = arg["argument_start_index"]
                labels[argument_start_index]= "B-{}".format(role)
                for i in range(1, len(argument)):
                    labels[argument_start_index+i]= "I-{}".format(role)
                if arg['alias']!=[]: print(arg['alias'])
            
            results.append({"id":row["id"],  "event_type":event_type, "segment_ids":segment_ids,\
                 "tokens":list(row["text"]), "labels":labels})
    write_file(results,output_file)
    print(min(len_text), max(len_text), mean(len_text))

def role_process_binary(input_file, output_file, is_predict=False):
    label_list = get_labels(task= "role", mode="classification")
    label_map = {label: i for i, label in enumerate(label_list)}
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        start_labels = ['O']*len(row["text"]) 
        end_labels = ['O']*len(row["text"]) 
        arguments = []
        if is_predict: 
            results.append({"id":row["id"], "tokens":list(row["text"]), "start_labels":start_labels, "end_labels":end_labels, "arguments":arguments})
            continue
        for event in row["event_list"]:
            event_type = event["event_type"]
            for arg in event["arguments"]:
                role = arg['role']
                role_id = label_map[role]
                argument = arg['argument']
                argument_start_index = arg["argument_start_index"]
                argument_end_index = argument_start_index + len(argument) -1

                if start_labels[argument_start_index]=="O":
                    start_labels[argument_start_index] = role
                else: 
                    start_labels[argument_start_index] += (" "+ role)
                if end_labels[argument_end_index]=="O":
                    end_labels[argument_end_index] = role
                else: 
                    end_labels[argument_end_index] += (" "+ role)

                if arg['alias']!=[]: print(arg['alias'])

                arg.pop('alias')
                arguments.append(arg)

        results.append({"id":row["id"], "tokens":list(row["text"]), "start_labels":start_labels, "end_labels":end_labels, "arguments":arguments})
    write_file(results,output_file)

def role_segment_process_binary(input_file, output_file, is_predict=False):
    label_list = get_labels(task= "role", mode="classification")
    label_map = {label: i for i, label in enumerate(label_list)}
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        if is_predict: 
            results.append({"id":row["id"], "tokens":list(row["text"]), \
                "start_labels":['O']*len(row["text"]), "end_labels":['O']*len(row["text"])})
            continue
        for event in row["event_list"]:
            event_type = event["event_type"]
            trigger = event["trigger"]
            trigger_start_index = event["trigger_start_index"]
            segment_ids= [0] * len(row["text"])
            for i in range(trigger_start_index, trigger_start_index+ len(trigger) ):
                segment_ids[i] = 1
            start_labels = ['O']*len(row["text"]) 
            end_labels = ['O']*len(row["text"]) 

            for arg in event["arguments"]:
                role = arg['role']
                role_id = label_map[role]
                argument = arg['argument']
                argument_start_index = arg["argument_start_index"]
                argument_end_index = argument_start_index + len(argument) -1

                if start_labels[argument_start_index]=="O":
                    start_labels[argument_start_index] = role
                else: 
                    start_labels[argument_start_index] += (" "+ role)
                if end_labels[argument_end_index]=="O":
                    end_labels[argument_end_index] = role
                else: 
                    end_labels[argument_end_index] += (" "+ role)

                if arg['alias']!=[]: print(arg['alias'])
            results.append({"id":row["id"], "tokens":list(row["text"]), "event_type":event_type, \
                "segment_ids":segment_ids,"start_labels":start_labels, "end_labels":end_labels})
    write_file(results,output_file)



def joint_process_binary(input_file, output_file, is_predict=False):
    label_list = get_labels(task= "role", mode="classification")
    label_map = {label: i for i, label in enumerate(label_list)}
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        
        if is_predict:
            results.append({"id":row["id"], "tokens":list(row["text"]), \
              "trigger_start_labels":['O']*len(row["text"]), "role_end_labels":['O']*len(row["text"]), \
              "role_start_labels":['O']*len(row["text"]), "role_end_labels":['O']*len(row["text"])})
            continue

        trigger_start_labels = ['O']*len(row["text"]) 
        trigger_end_labels = ['O']*len(row["text"]) 

        # trigger
        for event in row["event_list"]:
            event_type = event["event_type"]
            trigger = event["trigger"]
            trigger_start_index = event['trigger_start_index']
            trigger_end_index = trigger_start_index + len(trigger) -1
            if trigger_start_labels[trigger_start_index]=="O":
                trigger_start_labels[trigger_start_index] = event_type
            else: 
                trigger_start_labels[trigger_start_index] += (" "+ event_type)
            if trigger_end_labels[trigger_end_index]=="O":
                trigger_end_labels[trigger_end_index] = event_type
            else: 
                trigger_end_labels[trigger_end_index] += (" "+ event_type)
        
        # role
        for event in row["event_list"]:
            event_type = event["event_type"]
            trigger = event["trigger"]
            trigger_start_index = event['trigger_start_index']
            segment_ids= [0] * len(row["text"])
            for i in range(trigger_start_index, trigger_start_index+ len(trigger) ):
                segment_ids[i] = 1

            role_start_labels = ['O']*len(row["text"]) 
            role_end_labels = ['O']*len(row["text"]) 

            for arg in event["arguments"]:
                role = arg['role']
                role_id = label_map[role]
                argument = arg['argument']
                argument_start_index = arg["argument_start_index"]
                argument_end_index = argument_start_index + len(argument) -1
                
                if role_start_labels[argument_start_index]=="O":
                    role_start_labels[argument_start_index] = role
                else: 
                    role_start_labels[argument_start_index] += (" "+ role)
                    
                if role_end_labels[argument_end_index]=="O":
                    role_end_labels[argument_end_index] = role
                else: 
                    role_end_labels[argument_end_index] += (" "+ role)

                if arg['alias']!=[]: print(arg['alias'])
            results.append({"id":row["id"], "tokens":list(row["text"]), "segment_ids":segment_ids, \
                "trigger_start_labels":trigger_start_labels, "trigger_end_labels":trigger_end_labels, \
                    "role_start_labels":role_start_labels, "role_end_labels":role_end_labels})

    write_file(results,output_file)



def role_process_filter(event_class, input_file, output_file, is_predict=False):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        labels = ['O']*len(row["text"])
        if is_predict: continue
        flag = False
        for event in row["event_list"]:
            event_type = event["event_type"]
            if event_class != event["class"]:
                continue
            flag = True
            for arg in event["arguments"]:
                role = arg['role']
                argument = arg['argument']
                argument_start_index = arg["argument_start_index"]
                labels[argument_start_index]= "B-{}".format(role)
                for i in range(1, len(argument)):
                    labels[argument_start_index+i]= "I-{}".format(role)
        if not flag: continue
        results.append({"id":row["id"], "tokens":list(row["text"]), "labels":labels})
    write_file(results,output_file)

def get_event_class(schema_file):
    rows = open(schema_file, encoding='utf-8').read().splitlines()
    labels=[]
    for row in rows:
        row = json.loads(row)
        event_class = row["class"]
        if event_class in labels:
            continue
        labels.append(event_class)
    return labels

def index_output_bio_trigger(test_file, prediction_file, output_file):
    tests = open(test_file, encoding='utf-8').read().splitlines()
    predictions = open(prediction_file, encoding='utf-8').read().splitlines()
    results = []
    index = 0
    max_length = 256-2
    for test, prediction in zip(tests, predictions):
        index += 1
        test = json.loads(test)
        tokens = test.pop('tokens')
        test['text'] = ''.join(tokens)

        prediction = json.loads(prediction)
        labels = prediction["labels"]
        if len(labels)!=len(tokens) and len(labels) != max_length:
            print(labels, tokens)
            print(len(labels), len(tokens), index)
            break
        test["labels"] = labels

        results.append(test)
    write_file(results, output_file)

def index_output_bin_trigger(test_file, prediction_file, output_file):
    tests = open(test_file, encoding='utf-8').read().splitlines()
    predictions = open(prediction_file, encoding='utf-8').read().splitlines()
    results = []
    index = 0
    for test, prediction in zip(tests, predictions):
        index += 1
        test = json.loads(test)

        prediction = json.loads(prediction)
        labels = prediction["labels"]
        test["labels"] = labels

        results.append(test)
    write_file(results, output_file)

def index_output_bio_arg(test_file, prediction_file, output_file):
    tests = open(test_file, encoding='utf-8').read().splitlines()
    predictions = open(prediction_file, encoding='utf-8').read().splitlines()
    results = []
    index = 0
    max_length = 256-2
    for test, prediction in zip(tests, predictions):
        index += 1
        test = json.loads(test)
        tokens = test.pop('tokens')
        test['text'] = ''.join(tokens)

        prediction = json.loads(prediction)
        labels = prediction["labels"]
        if len(labels)!=len(tokens) and len(labels) != max_length:
            print(labels, tokens)
            print(len(labels), len(tokens), index)
            break

        args = extract_result(test["text"], labels)
        arguments = []
        for arg in args:
            argument = {}
            argument["role"] = arg["type"]
            argument["argument_start_index"] = arg['start']
            argument["argument"] =''.join(arg['text'])
            arguments.append(argument)
        
        test.pop("labels")
        test["arguments"] = arguments
        results.append(test)
    write_file(results, output_file)


def index_output_segment_bin(test_file, prediction_file, output_file):
    label_list = get_labels(task='role', mode="classification")
    label_map =  {i: label for i, label in enumerate(label_list)}

    tests = open(test_file, encoding='utf-8').read().splitlines()
    predictions = open(prediction_file, encoding='utf-8').read().splitlines()
    results = []
    index = 0
    max_length = 256-2
    for test, prediction in zip(tests, predictions):
        index += 1
        test = json.loads(test)
        start_labels = test.pop('start_labels')
        end_labels = test.pop('end_labels')

        tokens = test.pop('tokens')
        text = ''.join(tokens)
        test['text'] = text

        segment_ids =  test.pop('segment_ids')
        trigger = ''.join([tokens[i] for i in range(len(tokens)) if segment_ids[i]])
        for i in range(len(tokens)):
            if segment_ids[i]:
                trigger_start_index = i
                break
        
        event = {}
        # event['trigger'] = trigger
        # event['trigger_start_index']= trigger_start_index
        event_type = test.pop("event_type")
        event["event_type"]=event_type

        prediction = json.loads(prediction)
        arg_list = prediction["labels"]
        arguments =[]
        for arg in arg_list:
            sub_dict = {}
            argument_start_index = arg[1] -1 
            argument_end_index = arg[2] -1 
            argument = text[argument_start_index:argument_end_index+1]
            role = label_map[arg[3]]
            sub_dict["role"]=role
            sub_dict["argument"]=argument
            # sub_dict["argument_start_index"] = argument_start_index
            arguments.append(sub_dict)
        
        event["arguments"]= arguments

        test['event_list']= [event]
        results.append(test)
    write_file(results, output_file)

def index_output_bin_arg(test_file, prediction_file, output_file):
    label_list = get_labels(task='role', mode="classification")
    label_map =  {i: label for i, label in enumerate(label_list)}

    tests = open(test_file, encoding='utf-8').read().splitlines()
    predictions = open(prediction_file, encoding='utf-8').read().splitlines()
    results = []
    index = 0
    max_length = 256-2
    for test, prediction in zip(tests, predictions):
        index += 1
        test = json.loads(test)
        start_labels = test.pop('start_labels')
        end_labels = test.pop('end_labels')

        tokens = test.pop('tokens')
        text = ''.join(tokens)
        test['text'] = text

        prediction = json.loads(prediction)
        arg_list = prediction["labels"]
        arguments =[]
        for arg in arg_list:
            sub_dict = {}
            argument_start_index = arg[1] -1 
            argument_end_index = arg[2] -1 
            argument = text[argument_start_index:argument_end_index+1]
            role = label_map[arg[3]]
            sub_dict["role"]=role
            sub_dict["argument"]=argument
            sub_dict["argument_start_index"] = argument_start_index
            arguments.append(sub_dict)
        
        test["arguments"]= arguments
        results.append(test)
    write_file(results, output_file)


# un-finished
# def binary_to_bio(test_file, prediction_file, output_file):
#     tests = open(test_file, encoding='utf-8').read().splitlines()
#     predictions = open(prediction_file, encoding='utf-8').read().splitlines()
#     results = []
#     for test,prediction in zip(tests, predictions):
#         test = json.loads(test)
#         tokens = test.pop('tokens')
#         test['text'] = ''.join(tokens)

#         row_preds_list = json.loads(prediction)
        
#         labels= ['O']*len(tokens)
#         # for pred in row_preds_list:
        
#         test.update(prediction) 

#         results.append(test)
#     write_file(results, output_file)

# ner_segment_bi 输入的预处理函数

def convert_bio_to_segment(input_file, output_file):
    lines = open(input_file, encoding='utf-8').read().splitlines()
    res = []
    for line in lines:
        line = json.loads(line)
        text = line["text"]
        labels = line["labels"]
        tokens = list(text)
        if len(labels)!=len(tokens):
            print(len(labels), len(tokens))

        triggers = extract_result(text, labels)
        if len(triggers)==0:
            print("detect no trigger")
        for trigger in triggers:
            event_type= trigger["type"]
            segment_ids = [0]*(len(tokens))
            trigger_start_index = trigger['start']
            trigger_end_index = trigger['start'] + len(trigger['text'])
            for i in range(trigger_start_index, trigger_end_index):
                segment_ids[i] = 1
            start_labels = ['O']*(len(tokens))
            end_labels =  ['O']*(len(tokens))

            cur_line = {}
            cur_line["id"] = line["id"]
            cur_line["tokens"] = tokens
            cur_line["event_type"] = event_type
            cur_line["segment_ids"] = segment_ids
            cur_line["start_labels"] = start_labels
            cur_line["end_labels"] = end_labels
            res.append(cur_line)
    write_file(res, output_file)


def convert_bio_to_label(input_file, output_file):
    lines = open(input_file, encoding='utf-8').read().splitlines()
    res = []
    for line in lines:
        line_json = json.loads(line)
        labels = []
        for label in line_json["labels"]:
            if label.startswith("B-") and label[2:] not in labels:
                labels.append(label[2:])
        res.append({"labels":labels})
    write_file(res, output_file)

def compute_matric(label_file, pred_file):
    label_lines = open(label_file, encoding='utf-8').read().splitlines()
    pred_lines = open(pred_file, encoding='utf-8').read().splitlines()

    labels = []
    for i, line in enumerate(label_lines):
        json_line = json.loads(line)
        for label in json_line['labels']:
            labels.append([i, label])
    
    preds = []
    for i, line in enumerate(pred_lines):
        json_line = json.loads(line)
        for label in json_line['labels']:
            preds.append([i, label])

    nb_correct  = 0
    for out_label in labels:
        if out_label in preds:
            nb_correct += 1
            continue
    nb_pred = len(preds)
    nb_true = len(labels)
    # print(nb_correct, nb_pred, nb_true)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    
    print(p, r, f1)



def split_data(input_file, output_dir, num_split=5):
    datas = open(input_file, encoding='utf-8').read().splitlines()
    for i in range(num_split):
        globals()["train_data"+str(i+1)] = []
        globals()["dev_data"+str(i+1)] = []
    for i, data in enumerate(datas):
        cur = i % num_split + 1
        for j in range(num_split):
            if cur == j+1:
                globals()["dev_data" + str(j + 1)].append(json.loads(data))
            else:
                globals()["train_data"+str(j + 1)].append(json.loads(data))
    for i in range(num_split):
        cur_dir = os.path.join(output_dir, str(i))
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        write_file(globals()["train_data"+str(i + 1)], os.path.join(cur_dir, "train.json"))
        write_file(globals()["dev_data"+str(i + 1)], os.path.join(cur_dir, "dev.json"))


if __name__ == '__main__':

    # trigger_classify_file_remove_id("./data/trigger_classify/dev.json", "./data/trigger_classify/dev_without_id.json")

    # split_data("./data/trigger_classify/train.json",  "./data/trigger_classify",  num_split=5)
    # split_data("./data/role_bin/train.json",  "./data/role_bin",  num_split=5)

    # trigger_classify_process("./data/train_data/train.json", "./data/trigger_classify/train.json")
    # trigger_classify_process("./data/dev_data/dev.json", "./data/trigger_classify/dev.json")
    # trigger_classify_process("./data/test1_data/test1.json", "./data/trigger_classify/test.json",is_predict=True)

    # trigger_process_binary("./data/train_data/train.json", "./data/trigger_bin/train.json")
    # trigger_process_binary("./data/dev_data/dev.json","./data/trigger_bin/dev.json")
    # trigger_process_binary("./data/test1_data/test1.json", "./data/trigger_bin/test.json",is_predict=True)

    # role_process_binary("./data/train_data/train.json", "./data/role_bin/train.json")
    # role_process_binary("./data/dev_data/dev.json","./data/role_bin/dev.json")
    # role_process_binary("./data/test1_data/test1.json", "./data/role_bin/test.json",is_predict=True)

    # role_segment_process("./data/train_data/train.json", "./data/role_segment/train.json")
    # role_segment_process("./data/dev_data/dev.json","./data/role_segment/dev.json")
    # role_segment_process("./data/test1_data/test1.json", "./data/role_segment/test.json",is_predict=True)

    # joint_process_binary("./data/train_data/train.json", "./data/joint_bin/train.json")
    # joint_process_binary("./data/dev_data/dev.json","./data/joint_bin/dev.json")
    # joint_process_binary("./data/test1_data/test1.json", "./data/joint_bin/test.json",is_predict=True)

    # role_segment_process_binary("./data/train_data/train.json", "./data/role_segment_bin/train.json")
    # role_segment_process_binary("./data/dev_data/dev.json","./data/role_segment_bin/dev.json")


    # event_class_list = get_event_class("./data/event_schema/event_schema.json")
    # for event_class in event_class_list:
    #     if not os.path.exists("./data/role/{}".format(event_class)):
    #         os.makedirs("./data/role/{}".format(event_class))
    #     role_process_filter(event_class, "./data/train_data/train.json", "./data/role/{}/train.json".format(event_class))
    #     role_process_filter(event_class, "./data/dev_data/dev.json","./data/role/{}/dev.json".format(event_class))

    # index_output_bio_trigger("./data/trigger/dev.json" , "./output/trigger/checkpoint-best/eval_predictions.json","./output/trigger/checkpoint-best/eval_predictions_indexed.json" )
    # index_output_bio_trigger("./data/trigger/test.json" , "./output/trigger/checkpoint-best/test_predictions.json","./output/trigger/checkpoint-best/test_predictions_indexed.json" )
    
    # index_output_bin_trigger("./data/trigger_classify/dev.json" , "./output/trigger_classify/merge/eval_predictions_labels.json","./output/trigger_classify/merge/eval_predictions_indexed_labels.json" )
    # index_output_bin_trigger("./data/trigger_classify/test.json" , "./output/trigger_classify/merge/test_predictions_labels.json","./output/trigger_classify/merge/test_predictions_indexed_labels.json" )

    # index_output_bio_arg("./data/role/dev.json" , "./output/role/checkpoint-best/eval_predictions.json","./output/role/checkpoint-best/eval_predictions_labels.json" )
    # index_output_bio_arg("./data/role/test.json" , "./output/role/checkpoint-best/test_predictions.json","./output/role/checkpoint-best/test_predictions_indexed.json" )

    # index_output_segment_bin("./data/role_segment_bin/dev.json" , "./output/role_segment_bin/checkpoint-best/eval_predictions.json","./output/role_segment_bin/checkpoint-best/eval_predictions_indexed.json" )
    # index_output_segment_bin("./data/role_segment_bin/test.json" , "./output/role_segment_bin/checkpoint-best/test_predictions.json","./output/role_segment_bin/checkpoint-best/test_predictions_indexed.json" )

    # index_output_bin_arg("./data/role_bin/dev.json" , "./output/role_bin/merge/eval_predictions_labels.json","./output/role_bin/merge/eval_predictions_indexed_labels.json" )
    # index_output_bin_arg("./data/role_bin/test.json" , "./output/role_bin/merge/test_predictions_labels.json","./output/role_bin/merge/test_predictions_indexed_labels.json" )

    # convert_bio_to_segment("./output/trigger/checkpoint-best/test_predictions_indexed.json",\
    #     "./output/trigger/checkpoint-best/test_predictions_indexed_semgent_id.json")

    # convert_bio_to_label("./output/trigger/checkpoint-best/eval_predictions.json",\
    #      "./output/trigger/checkpoint-best/eval_predictions_labels.json")
    # compute_matric("./data/trigger_classify/dev.json", "./output/trigger/checkpoint-best/eval_predictions_labels.json")



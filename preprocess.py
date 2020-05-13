import json
import os
from utils import get_labels

def write_file(datas, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in datas:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")

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
        if is_predict: 
            results.append({"id":row["id"], "tokens":list(row["text"]), "start_labels":start_labels, "end_labels":end_labels})
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
        results.append({"id":row["id"], "tokens":list(row["text"]), "start_labels":start_labels, "end_labels":end_labels})
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



def data_val(input_file):
    rows = open(input_file, encoding='utf-8').read().splitlines()

    event_class_count = 0
    role_count = 0
    arg_count = 0
    arg_role_count = 0
    arg_role_one_event_count = 0
    trigger_count = 0
    argument_len_list =[]

    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)

        arg_start_index_list=[]
        arg_start_index_map={}
        event_class_list = []
        trigger_start_index_list = []

        event_class_flag = False
        arg_start_index_flag= False
        role_flag = False
        arg_role_flag= False
        arg_role_one_event_flag= False
        trigger_flag = False

        for event in row["event_list"]:
            event_class = event["class"]
            if event_class_list==[]: 
                event_class_list.append(event_class)
            elif event_class not in event_class_list:
                # event_class_count += 1
                event_class_flag = True
                # print(row)
            
            trigger_start_index= event["trigger_start_index"]
            if trigger_start_index not in trigger_start_index_list:
                trigger_start_index_list.append(trigger_start_index)
            else:
                trigger_flag = True
                print(row)

            role_list = []
            arg_start_index_map_in_one_event = {}
            for arg in  event["arguments"]:
                role = arg['role']
                argument = arg['argument']
                argument_start_index = arg["argument_start_index"]
                argument_len_list.append([len(argument),argument])
                if role not in role_list:
                    role_list.append(role)
                else: 
                    # role_count += 1
                    arg_start_index_flag = True
                    # print(row)
                
                if argument_start_index not in arg_start_index_map_in_one_event:
                    arg_start_index_map_in_one_event[argument_start_index]= role
                else:
                    if role!= arg_start_index_map_in_one_event[argument_start_index]:
                        arg_role_one_event_flag = True
                        # print(row)


                if argument_start_index not in arg_start_index_list:
                    arg_start_index_list.append(argument_start_index)
                    arg_start_index_map[argument_start_index]= role
                else: 
                    # arg_count+= 1
                    role_flag = True
                    if role!= arg_start_index_map[argument_start_index]:
                        arg_role_flag = True
                        # print(row)
    
        if role_flag:
            role_count += 1
            # print(row)
        if event_class_flag:
            event_class_count += 1
            # print(row)
        if arg_start_index_flag:
            arg_count += 1
            # print(row)
        if arg_role_flag:
            arg_role_count += 1
        if arg_role_one_event_flag:
            arg_role_one_event_count += 1
        if trigger_flag:
            trigger_count += 1
    
    print(event_class_count, role_count, arg_count, arg_role_count, arg_role_one_event_count, trigger_count)
    argument_len_list.sort(key=lambda x:x[0], reverse= True)
    print(argument_len_list[:10])

def position_val(input_file):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    trigger_count = 0
    arg_count = 0

    for row in rows:
        # position_flag = False

        if len(row)==1: print(row)
        row = json.loads(row)
        text = row['text']
        for event in row["event_list"]:
            event_class = event["class"]
            trigger = event["trigger"]
            event_type = event["event_type"]
            trigger_start_index = event["trigger_start_index"]

            if text[trigger_start_index: trigger_start_index+len(trigger)]!= trigger:
                print("trigger position mismatch")
                trigger_count += 1

            for arg in  event["arguments"]:
                role = arg['role']
                argument = arg['argument']
                argument_start_index = arg["argument_start_index"]
                
                if text[argument_start_index: argument_start_index+len(argument)]!= argument:
                    print("argument position mismatch")
                    arg_count+=1
    
    print(trigger_count, arg_count)
            


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

def index_output(test_file, prediction_file, output_file):
    tests = open(test_file, encoding='utf-8').read().splitlines()
    predictions = open(prediction_file, encoding='utf-8').read().splitlines()
    results = []
    index = 0
    max_length = 256-2
    for test,prediction in zip(tests, predictions):
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
        test.update(prediction) 

        results.append(test)
    write_file(results, output_file)

def index_output_segment_bin(test_file, prediction_file, output_file):
    from utils_bi_ner_segment import get_labels
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

def index_output_bin(test_file, prediction_file, output_file):
    from utils_bi_ner_segment import get_labels
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
            # sub_dict["argument_start_index"] = argument_start_index
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
    from postprocess import extract_result
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
        

def read_write(input_file, output_file):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    results = []
    for row in rows:
        row = json.loads(row)
        id = row.pop('id')
        text = row.pop('text')
        # labels = row.pop('labels')
        event_list = row.pop('event_list')
        row['text'] = text
        row['id'] = id
        # row['labels'] = labels
        row['event_list'] = event_list
        results.append(row)
    write_file(results, output_file)


if __name__ == '__main__':
    trigger_classify_process("./data/train_data/train.json", "./data/trigger_classify/train.json")
    trigger_classify_process("./data/dev_data/dev.json", "./data/trigger_classify/dev.json")
    trigger_classify_process("./data/test1_data/test1.json", "./data/trigger_classify/test.json",is_predict=True)

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

    # data_val("./data/train_data/train.json")
    # data_val("./data/dev_data/dev.json")

    # 无异常
    # position_val("./data/train_data/train.json")
    # position_val("./data/dev_data/dev.json")


    # event_class_list = get_event_class("./data/event_schema/event_schema.json")
    # for event_class in event_class_list:
    #     if not os.path.exists("./data/role/{}".format(event_class)):
    #         os.makedirs("./data/role/{}".format(event_class))
    #     role_process_filter(event_class, "./data/train_data/train.json", "./data/role/{}/train.json".format(event_class))
    #     role_process_filter(event_class, "./data/dev_data/dev.json","./data/role/{}/dev.json".format(event_class))

    # index_output("./data/trigger/dev.json" , "./output/trigger/checkpoint-best/eval_predictions.json","./output/trigger/checkpoint-best/eval_predictions_indexed.json" )
    # index_output("./data/trigger/test.json" , "./output/trigger/checkpoint-best/test_predictions.json","./output/trigger/checkpoint-best/test_predictions_indexed.json" )
    
    # index_output("./data/role/dev.json" , "./output/role/checkpoint-best/eval_predictions.json","./output/role/checkpoint-best/eval_predictions_indexed.json" )
    # index_output("./data/role/test.json" , "./output/role2/checkpoint-best/test_predictions.json","./output/role2/checkpoint-best/test_predictions_indexed.json" )

    # index_output_segment_bin("./data/role_segment_bin/dev.json" , "./output/role_segment_bin/checkpoint-best/eval_predictions.json","./output/role_segment_bin/checkpoint-best/eval_predictions_indexed.json" )
    # index_output_segment_bin("./data/role_segment_bin/test.json" , "./output/role_segment_bin/checkpoint-best/test_predictions.json","./output/role_segment_bin/checkpoint-best/test_predictions_indexed.json" )

    # index_output_bin("./data/role_bin/dev.json" , "./output/role_bin/checkpoint-best/eval_predictions.json","./output/role_bin/checkpoint-best/eval_predictions_indexed.json" )
    # index_output_bin("./data/role_bin/test.json" , "./output/role_bin2/checkpoint-best/test_predictions.json","./output/role_bin2/checkpoint-best/test_predictions_indexed.json" )

    # convert_bio_to_segment("./output/trigger/checkpoint-best/test_predictions_indexed.json",\
    #     "./output/trigger/checkpoint-best/test_predictions_indexed_semgent_id.json")

    # read_write("./output/eval_pred.json", "./results/eval_pred.json")
    # read_write("./results/test1.trigger.pred.json", "./results/paddle.trigger.json")


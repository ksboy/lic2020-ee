import json

def write_file(datas, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in datas:
            json.dump(obj, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")

def remove_duplication(alist):
    res = []
    for item in alist:
        if item not in res:
            res.append(item)
    return res


def get_labels(path="./data/event_schema/event_schema.json", task='trigger', mode="ner"):
    if not path:
        if mode=='ner':
            return ["O", "B-ENTITY", "I-ENTITY"]
        else:
            return ["O"]

    elif task=='trigger':
        labels = []
        rows = open(path, encoding='utf-8').read().splitlines()
        if mode == "ner": labels.append('O')
        for row in rows:
            row = json.loads(row)
            event_type = row["event_type"]
            if mode == "ner":
                labels.append("B-{}".format(event_type))
                labels.append("I-{}".format(event_type))
            else:
                labels.append(event_type)
        return remove_duplication(labels)

    elif task=='role':
        labels = []
        rows = open(path, encoding='utf-8').read().splitlines()
        if mode == "ner": labels.append('O')
        for row in rows:
            row = json.loads(row)
            for role in row["role_list"]:
                role_type = role['role']
                if mode == "ner":
                    labels.append("B-{}".format(role_type))
                    labels.append("I-{}".format(role_type))
                else:
                    labels.append(role_type)
        return remove_duplication(labels)
        
    else:
        labels = []
        rows = open(path, encoding='utf-8').read().splitlines()
        if mode == "ner": labels.append('O')
        for row in rows:
            row = json.loads(row)
            if row['class']!=task:
                continue
            for role in row["role_list"]:
                role_type = role['role']
                if mode == "ner":
                    labels.append("B-{}".format(role_type))
                    labels.append("I-{}".format(role_type))
                else:
                    labels.append(role_type)
        return remove_duplication(labels)

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
            

# 统计 event_type 分布
def data_analysis(input_file):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    label_list= get_labels(task='trigger', mode="classification")
    label_map = {label: i for i, label in enumerate(label_list)}
    label_count = [0 for i in range(len(label_list))]
    for row in rows:
        row = json.loads(row)
        for event in row["event_list"]:
            event_type = event["event_type"]
            label_count[label_map[event_type]] += 1
    print(label_count)

def get_num_of_arguments(input_file):
    lines = open(input_file, encoding='utf-8').read().splitlines()
    arg_count = 0
    for line in lines:
        line = json.loads(line)
        for event in line["event_list"]:
            arg_count += len(event["arguments"])
    print(arg_count)

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

def schema_analysis(path="./data/event_schema/event_schema.json"):
    rows = open(path, encoding='utf-8').read().splitlines()
    argument_map = {}
    for row in rows:
        d_json = json.loads(row)
        event_type = d_json["event_type"]
        for r in d_json["role_list"]:
            role = r["role"]
            if role in argument_map:
                argument_map[role].append(event_type)
            else: 
                argument_map[role]= [event_type]
    argument_unique = []
    argument_duplicate = []
    for argument, event_type_list in argument_map.items():
        if len(event_type_list)==1:
            argument_unique.append(argument)
        else:
            argument_duplicate.append(argument)

    print(argument_unique, argument_duplicate)
    for argument in argument_duplicate:
        print(argument_map[argument])

    return argument_map


if __name__ == '__main__':
    # labels = get_labels(path="./data/event_schema/event_schema.json", task='trigger', mode="classification")
    # print(len(labels), labels[50:60])
    
    # data_val("./data/train_data/train.json")
    # data_val("./data/dev_data/dev.json")

    # data_analysis("./data/train_data/train.json")

    # 无异常
    # position_val("./data/train_data/train.json")
    # position_val("./data/dev_data/dev.json")

    # get_num_of_arguments("./results/test_pred_bin_segment.json")

    # read_write("./output/eval_pred.json", "./results/eval_pred.json")
    # read_write("./results/test1.trigger.pred.json", "./results/paddle.trigger.json")

    schema_analysis()



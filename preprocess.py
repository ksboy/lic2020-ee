import json
import os

def write_file(datas, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in datas:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")

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

def role_process_binary(input_file, output_file, is_predict=False):
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
                argument = arg['argument']
                argument_start_index = arg["argument_start_index"]
                argument_end_index = argument_start_index + len(argument) -1
                start_labels[argument_start_index]= role
                end_labels[argument_end_index]= role
                if arg['alias']!=[]: print(arg['alias'])
        results.append({"id":row["id"], "tokens":list(row["text"]), "start_labels":start_labels, "end_labels":end_labels})
    write_file(results,output_file)

def trigger_role_process(input_file, output_file, is_predict=False):
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


def data_val(input_file):
    rows = open(input_file, encoding='utf-8').read().splitlines()

    event_class_count = 0
    role_count = 0
    arg_count = 0

    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)

        arg_start_index_list=[]
        event_class_list = []

        event_class_flag = False
        arg_start_index_flag= False
        role_flag = False

        for event in row["event_list"]:
            event_class = event["class"]
            if event_class_list==[]: 
                event_class_list.append(event_class)
            elif event_class not in event_class_list:
                # event_class_count += 1
                event_class_flag = True
                # print(row)

            role_list = []
            for arg in  event["arguments"]:
                role = arg['role']
                argument = arg['argument']
                argument_start_index = arg["argument_start_index"]
                if role not in role_list:
                    role_list.append(role)
                else: 
                    # role_count += 1
                    arg_start_index_flag = True
                    # print(row)

                if argument_start_index not in arg_start_index_list:
                    arg_start_index_list.append(argument_start_index)
                else: 
                    # arg_count+= 1
                    role_flag = True
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

    print(event_class_count, role_count, arg_count)

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
    for test,prediction in zip(tests, predictions):
        test = json.loads(test)
        tokens = test.pop('tokens')
        test['text'] = ''.join(tokens)

        prediction = json.loads(prediction)
        test.update(prediction) 

        results.append(test)
    write_file(results, output_file)
        
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
    # trigger_process_binary("./data/train_data/train.json", "./data/trigger_bin/train.json")
    # trigger_process_binary("./data/dev_data/dev.json","./data/trigger_bin/dev.json")
    # trigger_process_binary("./data/test1_data/test1.json", "./data/trigger_bin/test.json",is_predict=True)

    # role_process_binary("./data/train_data/train.json", "./data/role_bin/train.json")
    # role_process_binary("./data/dev_data/dev.json","./data/role_bin/dev.json")
    # role_process_binary("./data/test1_data/test1.json", "./data/role_bin/test.json",is_predict=True)

    # trigger_role_process("./data/train_data/train.json", "./data/trigger_role/train.json")
    # trigger_role_process("./data/dev_data/dev.json","./data/trigger_role/dev.json")
    # trigger_role_process("./data/test1_data/test1.json", "./data/trigger_role/test.json",is_predict=True)

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

    # index_output("./data/role/dev.json" , "./output/role/eval_predictions.json","./output/role/eval_predictions_indexed.json" )
    # index_output("./data/trigger/dev.json" , "./output/trigger/eval_predictions.json","./output/trigger/eval_predictions_indexed.json" )

    read_write("./output/eval_pred.json", "./results/eval_pred.json")
    # read_write("./results/test1.trigger.pred.json", "./results/paddle.trigger.json")


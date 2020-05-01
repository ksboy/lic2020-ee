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

def role_val(input_file):
    rows = open(input_file, encoding='utf-8').read().splitlines()
    count = 0
    for row in rows:
        if len(row)==1: print(row)
        row = json.loads(row)
        event_class = ""
        for event in row["event_list"]:
            cur_class = event["class"]
            if not event_class: event_class=cur_class
            elif event_class!=cur_class:
                print(row)
                count += 1
    print(count)

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
        labels = row.pop('labels')
        row['id'] = id
        row['labels'] = labels
        row['text'] = text
        results.append(row)
    write_file(results, output_file)


if __name__ == '__main__':
    # trigger_process("./data/train_data/train.json", "./data/trigger/train.json")
    # trigger_process("./data/dev_data/dev.json","./data/trigger/dev.json")
    # trigger_process("./data/test1_data/test1.json", "./data/trigger/test.json",is_predict=True)

    # role_process("./data/train_data/train.json", "./data/role/train.json")
    # role_process("./data/dev_data/dev.json","./data/role/dev.json")
    # role_process("./data/test1_data/test1.json", "./data/role/test.json",is_predict=True)

    role_val("./data/train_data/train.json")
    # role_val("./data/dev_data/dev.json")

    # event_class_list = get_event_class("./data/event_schema/event_schema.json")
    # for event_class in event_class_list:
    #     if not os.path.exists("./data/role/{}".format(event_class)):
    #         os.makedirs("./data/role/{}".format(event_class))
    #     role_process_filter(event_class, "./data/train_data/train.json", "./data/role/{}/train.json".format(event_class))
    #     role_process_filter(event_class, "./data/dev_data/dev.json","./data/role/{}/dev.json".format(event_class))

    # index_output("./data/role/test.json" , "./output/role/test_predictions.json","./output/role/test_predictions_indexed.json" )
    # index_output("./data/trigger/test.json" , "./output/trigger/test_predictions.json","./output/trigger/test_predictions_indexed.json" )

    # read_write("./results/test1.role.pred.json", "./results/paddle.role.json")
    # read_write("./results/test1.trigger.pred.json", "./results/paddle.trigger.json")


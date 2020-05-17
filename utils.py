import json

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


if __name__ == '__main__':
    labels = get_labels(path="./data/event_schema/event_schema.json", task='role')
    print(len(labels), labels)

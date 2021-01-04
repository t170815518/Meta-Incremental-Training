import os
from random import shuffle
import json


def index_file_to_id_format(file_path):
    entity2id = {}
    rel2id = {}
    entity_cnt = 0
    rel_cnt = 0

    for file_name in ["train.txt", "test.txt"]:
        translated_data = []
        with open(os.path.join(file_path, file_name), "r") as f:
            for line in f:
                e1, rel, e2 = line.replace(" ", "_").strip().split("\t")
                if e1 not in entity2id.keys():
                    entity2id[e1] = entity_cnt
                    entity_cnt += 1
                if e2 not in entity2id.keys():
                    entity2id[e2] = entity_cnt
                    entity_cnt += 1
                if rel not in rel2id.keys():
                    rel2id[rel] = rel_cnt
                    rel_cnt += 1
                e1_id = entity2id[e1]
                e2_id = entity2id[e2]
                rel_id = rel2id[rel]
                translated_data.append((e1_id, rel_id, e2_id))

        with open(os.path.join(file_path, file_name[:-4]), "w") as f:
            for e1, rel, e2 in translated_data:
                f.write("{}\t{}\t{}\n".format(e1, rel, e2))

    for d, file_name in zip([entity2id, rel2id], ["entity2id.txt", "relation2id.txt"]):
        with open(os.path.join(file_path, file_name), "w+") as f:
            for name, index in d.items():
                f.write("{}\t{}\n".format(name, index))


def translate_file(data_dir):
    entity2id = {}
    rel2id = {}
    with open(os.path.join(data_dir, "entity2id.txt"), 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            name = line[0]
            index = line[1]
            entity2id[name] = index
    with open(os.path.join(data_dir, "relation2id.txt"), 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            name = line[0]
            index = line[1]
            rel2id[name] = index
    for read_file_name, write_file_name in zip(["train.txt", "test.txt"], ["train", "test"]):
        triplets = []
        with open(os.path.join(data_dir, read_file_name), 'r') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                e1 = entity2id[line[0]]
                rel = rel2id[line[1]]
                e2 = entity2id[line[2]]
                triplets.append((e1, rel, e2))
        with open(os.path.join(data_dir, write_file_name), 'w+') as f:
            for e1, rel, e2 in triplets:
                f.write("{}\t{}\t{}\n".format(e1, rel, e2))


def split_train_test(path, export_path, test_ratio=0.2):
    test_set = []
    train_set = []

    with open(path, 'r') as f:
        lines = f.readlines()
        print(lines)
        shuffle(lines)
        test_size = round(test_ratio * len(lines))
        test_set = lines[:test_size]
        train_set = lines[test_size:]

    with open(os.path.join(export_path, "train"), 'w+') as f:
        f.writelines(train_set)
    with open(os.path.join(export_path, "test"), 'w+') as f:
        f.writelines(test_set)


def get_additional_information(entity2id_file_path, train_file_path):
    """ :param train_file_path: path to train file in index format """
    info_dict = {}
    with open(entity2id_file_path, 'r') as f:
        for line in f.readlines():
            _, entity_id = line.strip().split('\t')
            info_dict[int(entity_id)] = {"triplets_as_head": []}
    with open(train_file_path, 'r') as f:
        for line in f.readlines():
            head, rel, tail = tuple([int(x) for x in line.strip().split('\t')])  # convert string to integer
            info_dict[head]["triplets_as_head"].append([head, rel, tail])
    with open("graph_info.json", "w+") as f:
        f.write(json.dumps(info_dict))


if __name__ == '__main__':
    # split_train_test("data/alicoco/AliCoCo_v0.2.csv", "data/alicoco")
    index_file_to_id_format("data/FB15k-237")
    # translate_file("data/FB15k-237")
    # get_additional_information("data/FB15k-237/entity2id.txt", "data/FB15k-237/train")

import numpy as np
import os
from collections import defaultdict
import pickle
import dgl
import torch


def load_quadruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), "r") as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), "r") as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), "r") as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


def load_quadruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), "r") as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), "r") as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.array(quadrupleList), np.asarray(times)


def get_data_with_t(data, tim):
    x = data[np.where(data[:, 3] == tim)].copy()
    x = np.delete(x, 3, 1)  # drops 3rd column
    return x


def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm


def get_big_graph(data, num_rels):
    src, rel, dst = data.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    g = dgl.DGLGraph()
    g.add_nodes(len(uniq_v))
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel_o = np.concatenate((rel + num_rels, rel))
    rel_s = np.concatenate((rel, rel + num_rels))
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    g.ndata.update(
        {"id": torch.from_numpy(uniq_v).long().view(-1, 1), "norm": norm.view(-1, 1)}
    )
    g.edata["type_s"] = torch.LongTensor(rel_s)
    g.edata["type_o"] = torch.LongTensor(rel_o)
    g.ids = {}
    idx = 0
    for id in uniq_v:
        g.ids[id] = idx
        idx += 1
    return g


""" ACTUAL CODE """


graph_dict_train = {}

quadruples_train, train_times = load_quadruples("", "train.txt")
quadruples_test, test_times = load_quadruples("", "test.txt")
quadruples_dev, dev_times = load_quadruples("", "valid.txt")
# total_data, _ = load_quadruples('', 'train.txt', 'test.txt')

history_len = 10
num_entities, num_relations = get_total_number("", "stat.txt")

s_his = [[] for _ in range(num_entities)]
s_his_t = [[] for _ in range(num_entities)]
s_history_data = [[] for _ in range(len(quadruples_train))]
s_history_data_t = [[] for _ in range(len(quadruples_train))]
s_his_cache = [[] for _ in range(num_entities)]
s_his_cache_t = [None for _ in range(num_entities)]

o_his = [[] for _ in range(num_entities)]
o_his_t = [[] for _ in range(num_entities)]
o_history_data = [[] for _ in range(len(quadruples_train))]
o_history_data_t = [[] for _ in range(len(quadruples_train))]
o_his_cache = [[] for _ in range(num_entities)]
o_his_cache_t = [None for _ in range(num_entities)]

e = []
relation_id = []
latest_t = 0

# Here, we are creating a graph for each time step in the training data
for tim in train_times:
    data = get_data_with_t(quadruples_train, tim)
    graph_dict_train[tim] = get_big_graph(data, num_relations)

# Create the history data for both the subject and object
for i, quadruple in enumerate(quadruples_train):

    timestep = quadruple[3]
    if latest_t != timestep:

        # For each entity, check if the cache is empty
        for entity_id in range(num_entities):
            # Check if the cache is empty
            if len(s_his_cache[entity_id]) != 0:
                # If the history length is reached, remove the first element
                if len(s_his[entity_id]) >= history_len:
                    s_his[entity_id].pop(0)
                    s_his_t[entity_id].pop(0)

                s_his[entity_id].append(s_his_cache[entity_id].copy())
                s_his_t[entity_id].append(s_his_cache_t[entity_id])
                s_his_cache[entity_id] = []
                s_his_cache_t[entity_id] = None

            if len(o_his_cache[entity_id]) != 0:
                if len(o_his[entity_id]) >= history_len:
                    o_his[entity_id].pop(0)
                    o_his_t[entity_id].pop(0)

                o_his[entity_id].append(o_his_cache[entity_id].copy())
                o_his_t[entity_id].append(o_his_cache_t[entity_id])
                o_his_cache[entity_id] = []
                o_his_cache_t[entity_id] = None
        latest_t = timestep

    source_id = quadruple[0]
    relation_id = quadruple[1]
    object_id = quadruple[2]

    # Subject history
    s_history_data[i] = s_his[source_id].copy()
    s_history_data_t[i] = s_his_t[source_id].copy()

    if len(s_his_cache[source_id]) == 0:
        s_his_cache[source_id] = np.array([[relation_id, object_id]])
    else:
        s_his_cache[source_id] = np.concatenate(
            (s_his_cache[source_id], [[relation_id, object_id]]), axis=0
        )
    s_his_cache_t[source_id] = timestep

    # Object history
    o_history_data[i] = o_his[object_id].copy()
    o_history_data_t[i] = o_his_t[object_id].copy()

    if len(o_his_cache[object_id]) == 0:
        o_his_cache[object_id] = np.array([[relation_id, source_id]])
    else:
        o_his_cache[object_id] = np.concatenate(
            (o_his_cache[object_id], [[relation_id, source_id]]), axis=0
        )
    o_his_cache_t[object_id] = timestep


# Save everything
with open("train_graphs.txt", "wb") as fp:
    pickle.dump(graph_dict_train, fp)

with open("train_history_sub.txt", "wb") as fp:
    pickle.dump([s_history_data, s_history_data_t], fp)
with open("train_history_ob.txt", "wb") as fp:
    pickle.dump([o_history_data, o_history_data_t], fp)


# COPY PASTE FOR DEV DATA

# print(s_history_data[0])
s_history_data_dev = [[] for _ in range(len(quadruples_dev))]
o_history_data_dev = [[] for _ in range(len(quadruples_dev))]
s_history_data_dev_t = [[] for _ in range(len(quadruples_dev))]
o_history_data_dev_t = [[] for _ in range(len(quadruples_dev))]

for i, dev in enumerate(quadruples_dev):
    if i % 10000 == 0:
        print("valid", i, len(quadruples_dev))
    timestep = dev[3]
    if latest_t != timestep:
        for entity_id in range(num_entities):
            if len(s_his_cache[entity_id]) != 0:
                if len(s_his[entity_id]) >= history_len:
                    s_his[entity_id].pop(0)
                    s_his_t[entity_id].pop(0)
                s_his_t[entity_id].append(s_his_cache_t[entity_id])
                s_his[entity_id].append(s_his_cache[entity_id].copy())
                s_his_cache[entity_id] = []
                s_his_cache_t[entity_id] = None
            if len(o_his_cache[entity_id]) != 0:
                if len(o_his[entity_id]) >= history_len:
                    o_his[entity_id].pop(0)
                    o_his_t[entity_id].pop(0)

                o_his_t[entity_id].append(o_his_cache_t[entity_id])
                o_his[entity_id].append(o_his_cache[entity_id].copy())

                o_his_cache[entity_id] = []
                o_his_cache_t[entity_id] = None
        latest_t = timestep
    source_id = dev[0]
    relation_id = dev[1]
    object_id = dev[2]
    s_history_data_dev[i] = s_his[source_id].copy()
    o_history_data_dev[i] = o_his[object_id].copy()
    s_history_data_dev_t[i] = s_his_t[source_id].copy()
    o_history_data_dev_t[i] = o_his_t[object_id].copy()
    if len(s_his_cache[source_id]) == 0:
        s_his_cache[source_id] = np.array([[relation_id, object_id]])
    else:
        s_his_cache[source_id] = np.concatenate(
            (s_his_cache[source_id], [[relation_id, object_id]]), axis=0
        )
    s_his_cache_t[source_id] = timestep

    if len(o_his_cache[object_id]) == 0:
        o_his_cache[object_id] = np.array([[relation_id, source_id]])
    else:
        o_his_cache[object_id] = np.concatenate(
            (o_his_cache[object_id], [[relation_id, source_id]]), axis=0
        )
    o_his_cache_t[object_id] = timestep

    # print(o_his_cache[o])

with open("dev_history_sub.txt", "wb") as fp:
    pickle.dump([s_history_data_dev, s_history_data_dev_t], fp)
with open("dev_history_ob.txt", "wb") as fp:
    pickle.dump([o_history_data_dev, o_history_data_dev_t], fp)

s_history_data_test = [[] for _ in range(len(quadruples_test))]
o_history_data_test = [[] for _ in range(len(quadruples_test))]

s_history_data_test_t = [[] for _ in range(len(quadruples_test))]
o_history_data_test_t = [[] for _ in range(len(quadruples_test))]

for i, test in enumerate(quadruples_test):
    if i % 10000 == 0:
        print("test", i, len(quadruples_test))
    timestep = test[3]
    if latest_t != timestep:
        for entity_id in range(num_entities):
            if len(s_his_cache[entity_id]) != 0:
                if len(s_his[entity_id]) >= history_len:
                    s_his[entity_id].pop(0)
                    s_his_t[entity_id].pop(0)
                s_his_t[entity_id].append(s_his_cache_t[entity_id])

                s_his[entity_id].append(s_his_cache[entity_id].copy())
                s_his_cache[entity_id] = []
                s_his_cache_t[entity_id] = None
            if len(o_his_cache[entity_id]) != 0:
                if len(o_his[entity_id]) >= history_len:
                    o_his[entity_id].pop(0)
                    o_his_t[entity_id].pop(0)

                o_his_t[entity_id].append(o_his_cache_t[entity_id])

                o_his[entity_id].append(o_his_cache[entity_id].copy())
                o_his_cache[entity_id] = []
                o_his_cache_t[entity_id] = None
        latest_t = timestep
    source_id = test[0]
    relation_id = test[1]
    object_id = test[2]
    s_history_data_test[i] = s_his[source_id].copy()
    o_history_data_test[i] = o_his[object_id].copy()
    s_history_data_test_t[i] = s_his_t[source_id].copy()
    o_history_data_test_t[i] = o_his_t[object_id].copy()
    if len(s_his_cache[source_id]) == 0:
        s_his_cache[source_id] = np.array([[relation_id, object_id]])
    else:
        s_his_cache[source_id] = np.concatenate(
            (s_his_cache[source_id], [[relation_id, object_id]]), axis=0
        )
    s_his_cache_t[source_id] = timestep

    if len(o_his_cache[object_id]) == 0:
        o_his_cache[object_id] = np.array([[relation_id, source_id]])
    else:
        o_his_cache[object_id] = np.concatenate(
            (o_his_cache[object_id], [[relation_id, source_id]]), axis=0
        )
    o_his_cache_t[object_id] = timestep
    # print(o_his_cache[o])

with open("test_history_sub.txt", "wb") as fp:
    pickle.dump([s_history_data_test, s_history_data_test_t], fp)
with open("test_history_ob.txt", "wb") as fp:
    pickle.dump([o_history_data_test, o_history_data_test_t], fp)
    # print(train)

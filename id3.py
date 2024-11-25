# python implementation of ID3 algorithm
from collections import defaultdict
import json
import math
import anytree


def load_data_from_file(file: str) -> dict:
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def entropy(data: list, target: str, condition: str = None) -> float:
    if condition:
        values = set([d[condition] for d in data])
        ent = 0
        for v in values:
            sub_data = [d for d in data if d[condition] == v]
            ent += len(sub_data) / len(data) * entropy(sub_data, target)
        return ent
    else:
        values = set([d[target] for d in data])
        ent = 0
        for v in values:
            p = len([d for d in data if d[target] == v]) / len(data)
            ent -= p * math.log2(p)
        return ent


def select_best_feature(data: list, target: str, features: list) -> str:
    ent = entropy(data, target)
    gains = {f: ent - entropy(data, target, f) for f in features}
    return max(gains, key=gains.get)


def build_tree(data: list, target: str, features: list) -> anytree.Node:
    root = anytree.Node("root")
    add_decision_node(root, data, target, features)
    return root


def add_decision_node(node: anytree.Node, data: list, target: str, features: list):
    # if no data left
    if not data:
        anytree.Node("Unknown", parent=node)
        return
    # if all data has the same target value
    target_list = [d[target] for d in data]
    target_set = set(target_list)
    if len(target_set) == 1:
        anytree.Node(target_list[0], parent=node)
        return
    # if there is no feature left but the target values are not the same
    if len(features) == 0:
        # choose the most common target value
        target_value = max(set(target_list), key=target_list.count)
        anytree.Node(target_value, parent=node)
        return
    # choose the best feature, node name is the feature_name : feature_value
    local_features = features[:]
    best_feature = select_best_feature(data, target, local_features)
    local_features.remove(best_feature)

    def group_by(data, feature):
        grouped = defaultdict(list)
        for d in data:
            grouped[d[feature]].append(d)
        return grouped
    
    feature_groups = group_by(data, best_feature)
    for feature_value, sub_data in feature_groups.items():
        child = anytree.Node(f"{best_feature} : {feature_value}", parent=node)
        add_decision_node(child, sub_data, target, local_features)


if __name__ == '__main__':
    table = load_data_from_file('data.json')
    data = table['data']
    target = table['target']
    features = table['features']
    tree = build_tree(data, target, features)
    print(anytree.RenderTree(tree).by_attr())

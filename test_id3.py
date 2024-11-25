from id3 import *


def test_load_data_features():
    table = load_data_from_file('data.json')
    assert table['features'] == ["age", "income", "student", "credit_rating"]
    data = table['data']
    assert data[0]["age"] == "youth"
    assert table["target"] == "buys_computer"


def test_entropy():
    table = load_data_from_file('data.json')
    data = table['data']
    target = table['target']
    assert abs(entropy(data=data, target=target) - 0.94) < 0.01


def test_entropy_condition():
    table = load_data_from_file('data.json')
    data = table['data']
    target = table['target']
    condition = 'age'
    assert abs(entropy(data=data, target=target,
               condition=condition) - 0.69) < 0.01


def test_select_best_feature():
    table = load_data_from_file('data.json')
    data = table['data']
    target = table['target']
    features = table['features']
    assert select_best_feature(data, target, features) == 'age'


def test_build_tree():
    table = load_data_from_file('data.json')
    data = table['data']
    target = table['target']
    features = table['features']
    tree = build_tree(data, target, features)
    print(anytree.RenderTree(tree).by_attr())
    assert tree.name == 'root'

    def get_children(node):
        return set([child.name for child in node.children])

    assert get_children(tree) == {'age : youth',
                                  'age : middle_aged', 'age : senior'}


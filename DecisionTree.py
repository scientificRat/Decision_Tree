""""
create by 黄正跃 2016/10/5
"""
import math


def load_data(filename):
    data_file = open(filename, 'r')
    # read head
    # 属性集（有序）
    attributes = data_file.readline().strip(',label\n').split(',')
    # 所有属性的取值
    attributes_value_dict = {}
    for i in range(len(attributes)):
        attributes_value_dict[attributes[i]] = []
    # read data
    data_set = []  # 数据集(样例集)
    for line in data_file.readlines():
        temp_array = line.strip('\n').split(',')
        if '?' in temp_array:
            continue  # 如果有missing的值，不加载
        example = {}  # 样本(X,y)
        for i in range(len(temp_array) - 1):
            example[attributes[i]] = temp_array[i]
            if temp_array[i] not in attributes_value_dict[attributes[i]]:
                attributes_value_dict[attributes[i]].append(temp_array[i])
        example['label'] = temp_array[-1]  # 标记
        data_set.append(example)
    data_file.close()
    return attributes_value_dict, data_set


#  计算Gini值
def calc_gini_value(data_set):
    dic = {}
    for t_example in data_set:
        if t_example['label'] in dic:
            dic[t_example['label']] += 1
        else:
            dic[t_example['label']] = 1
    total = sum(dic.values())
    temp = 0
    for key in dic:
        pk = dic[key] / total
        temp += pk ** 2
    return 1 - temp


#  计算Gini ——index
def calc_gini_index_value(data_set, divide_attr, attributes_value_dict):
    gini_index = 0
    for attr_value in attributes_value_dict[divide_attr]:
        data_set_v = list(filter(lambda example: example[divide_attr] == attr_value, data_set))
        gini_index += len(data_set_v) * calc_gini_value(data_set_v)
    gini_index /= len(data_set)
    return gini_index


#  计算信息熵
def Ent(data_set):
    dic = {}
    for t_example in data_set:
        if t_example['label'] in dic:
            dic[t_example['label']] += 1
        else:
            dic[t_example['label']] = 1
    total = sum(dic.values())
    temp = 0
    for key in dic:
        pk = dic[key] / total
        if pk == 0:
            continue
        temp += pk * math.log2(pk)
    return -temp


#  计算信息增益
def Gain(data_set, divide_attr, attributes_value_dict):
    temp = 0
    for attr_value in attributes_value_dict[divide_attr]:
        data_set_v = list(filter(lambda example: example[divide_attr] == attr_value, data_set))
        temp += len(data_set_v) * Ent(data_set_v)
    temp /= len(data_set)
    return Ent(data_set) - temp


#  计算信息增益率
def Gain_ratio(data_set, divide_attr, attributes_value_dict):
    temp = 0.0001
    for attr_value in attributes_value_dict[divide_attr]:
        data_set_v = list(filter(lambda example: example[divide_attr] == attr_value, data_set))
        if len(data_set_v) == 0:
            temp += 1
        else:
            temp += len(data_set_v) * math.log2(len(data_set_v) / len(data_set))
    temp /= len(data_set)
    IV = -temp
    return Gain(data_set, divide_attr, attributes_value_dict) / IV


#  使用基尼指数选出最佳划分属性
def get_best_divide_attribute_gini(data_set, attributes_set, attributes_value_dict):
    mini_gini_index = calc_gini_index_value(data_set, attributes_set[0], attributes_value_dict)
    best_divide_attr = attributes_set[0]
    for i in range(1, len(attributes_set)):
        gini_index = calc_gini_index_value(data_set, attributes_set[i], attributes_value_dict)
        if gini_index < mini_gini_index:
            mini_gini_index = gini_index
            best_divide_attr = attributes_set[i]
    return best_divide_attr


#  使用信息增益选出最佳划分属性
def get_best_divide_attribute_grain(data_set, attributes_set, attributes_value_dict):
    max_grain = Gain(data_set, attributes_set[0], attributes_value_dict)
    best_divide_attr = attributes_set[0]
    for i in range(1, len(attributes_set)):
        grain = Gain(data_set, attributes_set[i], attributes_value_dict)
        if grain > max_grain:
            max_grain = grain
            best_divide_attr = attributes_set[i]
    return best_divide_attr


#  使用信息增益率选出最佳划分属性
def get_best_divide_attribute_grain_ratio(data_set, attributes_set, attributes_value_dict):
    max_grain = Gain_ratio(data_set, attributes_set[0], attributes_value_dict)
    best_divide_attr = attributes_set[0]
    for i in range(1, len(attributes_set)):
        grain = Gain_ratio(data_set, attributes_set[i], attributes_value_dict)
        if grain > max_grain:
            max_grain = grain
            best_divide_attr = attributes_set[i]
    return best_divide_attr


""""
建立决策树
"""


def generate_decision_tree(train_data_set, attributes_value_dict, get_best_divide_attribute):
    #  inner function
    def generate_decision_tree_help(data_set, attributes_set):
        def get_max_count_label():
            dic = {}
            for t_example in data_set:
                if t_example['label'] in dic:
                    dic[t_example['label']] += 1
                else:
                    dic[t_example['label']] = 1
            max_count = -1
            for key in dic:
                if dic[key] > max_count:
                    max_count = dic[key]
                    max_label = key
            return max_label

        # 如果data_set为全部相同标记，则返回此标记
        is_same_class = True
        i = 0
        while i < len(data_set) - 1:
            if data_set[i]['label'] != data_set[i + 1]['label']:
                is_same_class = False
                break
            i += 1
        if is_same_class:
            return data_set[i]['label']
        # attributes_set为空,返回data_set中最多的类的标记
        if len(attributes_set) == 0:
            return get_max_count_label()
        # 如果data_set 在属性集上的取值都相同
        is_same = True
        temp_dic = {}
        for attr in attributes_set:
            temp_dic[attr] = data_set[0][attr]
        for i in range(1, len(data_set)):
            for attr in attributes_set:
                if temp_dic[attr] != data_set[i][attr]:
                    is_same = False
                    break
        if is_same:
            return get_max_count_label()
        # 选择最佳划分属性
        divide_attr = get_best_divide_attribute(data_set, attributes_set, attributes_value_dict)
        node = divide_attr, {}  # I define a node is a tuple
        for attr_value in attributes_value_dict[divide_attr]:
            data_set_v = list(filter(lambda example: example[divide_attr] == attr_value, data_set))
            if len(data_set_v) == 0:
                node[1][attr_value] = get_max_count_label()
            else:
                new_attr_set = attributes_set.copy()
                new_attr_set.remove(divide_attr)
                node[1][attr_value] = generate_decision_tree_help(data_set_v, new_attr_set)
        return node

    return generate_decision_tree_help(train_data_set, list(attributes_value_dict.keys()))


def predict(instance, tree):
    if type(tree) != tuple:
        return tree
    attr_value = instance[tree[0]]
    return predict(instance, tree[1][attr_value])

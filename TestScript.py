from DecisionTree import *
import random
import sys
import numpy as np
import matplotlib.pyplot as plt


def print_attributes_dict_info(attr_dict):
    for key in attr_dict:
        print(str(key) + ":" + str(attr_dict[key]))


# k-折交叉检验
def k_fold_cross_validation(k, data_set, attributes_value_dict):
    print("running....")
    # 为保证划分均匀，先打乱
    random.shuffle(data_set)
    #  划分
    data_subsets = [data_set[i:i + k] for i in range(0, len(data_set), k)]
    print('-' * len(data_subsets))
    #  交叉检验
    error_ratio_set_gini = []
    error_ratio_set_grain = []
    error_ratio_set_grain_ratio = []
    for i in range(0, len(data_subsets)):
        #  打印进度条
        print("=", end='')
        sys.stdout.flush()
        train_set = []
        test_set = []
        for j in range(0, len(data_subsets)):
            if i != j:
                train_set += data_subsets[j]
            else:
                test_set = data_subsets[j]
        # 开始检验
        decision_tree_gini = generate_decision_tree(train_set, attributes_value_dict,
                                                    get_best_divide_attribute=get_best_divide_attribute_gini)
        decision_tree_grain = generate_decision_tree(train_set, attributes_value_dict,
                                                     get_best_divide_attribute=get_best_divide_attribute_grain)
        decision_tree_grain_ratio = \
            generate_decision_tree(train_set, attributes_value_dict,
                                   get_best_divide_attribute=get_best_divide_attribute_grain_ratio)
        error_gini = 0
        error_grain = 0
        error_grain_ratio = 0
        for example in test_set:
            infer_gini = predict(example, decision_tree_gini)
            infer_grain = predict(example, decision_tree_grain)
            infer_grain_ratio = predict(example, decision_tree_grain_ratio)
            if infer_gini != example['label']:
                error_gini += 1
            if infer_grain != example['label']:
                error_grain += 1
            if infer_grain_ratio != example['label']:
                error_grain_ratio += 1
        error_gini /= len(test_set)
        error_grain /= len(test_set)
        error_grain_ratio /= len(test_set)
        error_ratio_set_gini.append(error_gini)
        error_ratio_set_grain.append(error_grain)
        error_ratio_set_grain_ratio.append(error_grain_ratio)
    print()
    return error_ratio_set_gini, error_ratio_set_grain, error_ratio_set_grain_ratio


def plot_data(data):
    n_groups = len(data[0])

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.22

    opacity = 0.4
    plt.bar(index, data[0], bar_width, alpha=opacity, color='b', label='Gini')
    plt.bar(index + bar_width, data[1], bar_width, alpha=opacity, color='r', label='Grain')
    plt.bar(index + 2*bar_width, data[2], bar_width, alpha=opacity, color='lightskyblue',
            label='Grain ratio')

    plt.xlabel('Group')
    plt.ylabel('ratio')
    plt.title('k-折下测试集错误率')
    plt.legend()

    plt.tight_layout()
    plt.show()


def start():
    print("决策树生成器 v1.0")
    while True:
        print('输入数据文件路径(输入[0]为car [1]为tic-tac-toe [q]退出 )\n>>>', end='')
        filename = input()
        if filename == '1':
            filename = "data/tic-tac-toe-endgame"
        elif filename == '0':
            filename = "data/CarEvaluation/car"
        elif filename == 'q':
            break
        attributes_value_dict, data_set = load_data(filename)
        print("所有属性及其取值范围：")
        print_attributes_dict_info(attributes_value_dict)
        print("数据集大小：" + str(len(data_set)))
        print("开始20-折交叉检验...")
        rst = k_fold_cross_validation(20, data_set, attributes_value_dict)
        error_ratio_set_gini, error_ratio_set_grain, error_ratio_set_grain_ratio = rst

        def avg(lst):
            return sum(lst) / len(lst)

        av_gini = avg(error_ratio_set_gini)
        av_grain = avg(error_ratio_set_grain)
        av_grain_ratio = avg(error_ratio_set_grain_ratio)
        print("Gini:")
        print("平均错误率\t" + str(av_gini))
        print("平均精度\t" + str(1 - av_gini))
        print("Grain")
        print("平均错误率\t" + str(av_grain))
        print("平均精度\t" + str(1 - av_grain))
        print("Grain Ratio")
        print("平均错误率\t" + str(av_grain_ratio))
        print("平均精度\t" + str(1 - av_grain_ratio))
        plot_data(rst)


start()

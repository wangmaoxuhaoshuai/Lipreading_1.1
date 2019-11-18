from import_sets import *
"""
把数据集分为训练集（训练集和用于测试的训练集）和测试集
"""
if __name__ == '__main__':
    # val = 0.2 # 训练集0.8 测试集0.2
    data = read_csv_file('data/splited_data/0/train.csv')
    # train_num = int((1 - val) * len(data))
    np.random.shuffle(data)
    # # 训练集
    # with open('data/.csv', 'w', newline='', encoding='UTF-8') as f:
    #     for i_data in data[:train_num]:
    #         f_csv = csv.writer(f)
    #         f_csv.writerow(i_data)
    # 训练集（做测试）
    with open('data/validate_train.csv', 'w', newline='', encoding='UTF-8') as f:
        for iter, i_data in enumerate(data[:2000]):
            if iter == 2000:
                break
            f_csv = csv.writer(f)
            f_csv.writerow(i_data)
    #
    # # 测试集
    # with open('data/.csv', 'w', newline='', encoding='UTF-8') as f:
    #     for i_data in data[train_num:]:
    #         f_csv = csv.writer(f)
    #         f_csv.writerow(i_data)

    # load_sequence('/home/z195/home/z195/datasets/lips/lip_train/ff9667b22595f41b15843cca756ca446',False)





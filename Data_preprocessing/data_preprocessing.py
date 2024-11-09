import numpy as np
import tensorflow as tf


def data_reading(path, names, sample_length, feature_index):
    data = []
    for name in names:
        data.append(np.load(path + name + '.npy'))

    data = np.array(data).astype(np.float32)
    # print(data.shape)   # len(names), 49, 40, 20, 40, 70

    data = data[:, :, :, feature_index, :, 0:sample_length].copy()
    # print(data.shape)   # len(names), 49, 40, feature_map_num, 40, sample_length
    data = np.transpose(data, [1, 0, 2, 4, 5, 3])
    # print(data.shape)  # 49, len(names), 40, 40, sample_length, feature_map_num

    label = np.load(path + 'label_sample.npy').astype(np.int32)
    # print(label.shape)

    return data, label


def preprocess(x, y):

    x = np.transpose(x, [0, 2, 1, 3, 4, 5])
    data_h = []
    data_m = []
    for i in range(x.shape[0]):
        if y[i] == 0:
            data_h.extend(x[i])
        else:
            data_m.extend(x[i])

    data_h = np.transpose(data_h, [1, 0, 2, 3, 4])
    data_m = np.transpose(data_m, [1, 0, 2, 3, 4])

    data = [data_h, data_m]
    return data


def repreprocess(x, y):

    data = []
    label_illness = []

    for i in range(x.shape[0]):
        for j in range(x[i].shape[0]):
            temp = x[i][j]
            data.extend(temp)
        label_illness.extend(np.ones(x[i].shape[0] * x[i].shape[1], dtype=int) * y[i])

    data = tf.convert_to_tensor(data)
    label_illness = tf.one_hot(label_illness, 2)
    # print(label.shape)

    return data, label_illness


def model_outputs(pre_y, y, n):
    TP = 0  # 预测为正例，实际为正例
    FP = 0  # 预测为正例，实际为负例
    TN = 0  # 预测为负例，实际为负例
    FN = 0  # 预测为负例，实际为正例

    accuracy_person = 0
    temp = 0

    for i in range(len(y)):
        if y[i] == 1 and pre_y[i] == 1:
            TP += 1     # 真正率
            temp += 1
        if y[i] == 1 and pre_y[i] == 0:
            FN += 1     # 假负率
        if y[i] == 0 and pre_y[i] == 1:
            FP += 1     # 假正率
        if y[i] == 0 and pre_y[i] == 0:
            TN += 1     # 真负率
            temp += 1

        if (i + 1) % (40*n) == 0:
            if temp >= (20*n):
                accuracy_person += 1
            temp = 0

    return accuracy_person, TP, FP, FN, TN


def scores(matrix, i=0):
    acc = (matrix[0, 0] + matrix[1, 1]) / np.sum(matrix) * 100
    if matrix[0, 0] == 0:
        precision = 0
        recall = 0
        F1 = 0
    else:
        precision = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1]) * 100
        recall = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0]) * 100
        F1 = 2 * (precision * recall) / (precision + recall)

    if i == 1:
        print('accuracy: {:.02f}%\tprecision: {:.02f}%\trecall: {:.02f}%\tF1: {:.02f}%\t'
              .format(acc, precision, recall, F1), end='')

    return acc

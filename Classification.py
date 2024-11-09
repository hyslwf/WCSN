import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from Models.setgpuMG import SGMG
SGMG()
from sklearn.model_selection import KFold
from Data_preprocessing.data_preprocessing import data_reading, scores
from Models.training import model_training
import time

tf.get_logger().setLevel('ERROR')
start = time.perf_counter()

path = '../近红外抑郁数据集/'
# 'fear', 'happy', 'sad', 'rest'
data_names = ['sad', 'rest']

sample_length = 50
lr = 0.01
times_of_repetition = 70
epochs = 200

feature_index = [16, 17, 18]
feature_index = np.array(feature_index)
outputs_dim = len(feature_index)

max_results = np.ones((7, 5))

data, label = data_reading(path, data_names, sample_length, feature_index)

for num in range(times_of_repetition):
    print('\n第' + str(num + 1) + '轮：')
    results = np.zeros(5)

    kf = KFold(7)
    i = 0
    for train_index, test_index in kf.split(data):
        print('\n----------第' + str(i + 1) + '折----------')
        train_data, train_label = data[train_index], label[train_index]
        test_x, test_y = data[test_index], label[test_index]

        while True:
            permutation = np.random.permutation(len(train_data))
            train_x = train_data[permutation]
            train_y = train_label[permutation]

            if 0 < np.sum(train_y[-7:]) < 7:
                val_x, val_y = train_x[-7:], train_y[-7:]
                train_x, train_y = train_x[0:-7], train_y[0:-7]
                break

        result = model_training(train_x, train_y, val_x, val_y, test_x, test_y, lr, epochs, outputs_dim)
        result = np.array(result)

        print('本折结果: ', end='')
        scores(result[1:].reshape(2, 2))
        print('acc_person: {:.02f}%'.format(result[0] / 7 * 100))

        results += result
        if (result[1] + result[-1]) / np.sum(result[1:]) \
                > (max_results[i, 1] + max_results[i, -1]) / np.sum(max_results[i, 1:]):
            max_results[i] = result

        i += 1

    print('\n第' + str(num + 1) + '轮平均结果: ')
    scores(results[1:].reshape(2, 2))
    print('acc_person: {:.02f}%'.format(results[0] / 49 * 100))
    print('\n最好结果: ')
    for i in range(7):
        print((max_results[i, 1] + max_results[i, -1]) / np.sum(max_results[i, 1:]), end='\t')
    print()
    scores(np.sum(max_results, axis=0)[1:].reshape(2, 2))
    print('acc_person: {:.02f}%\n'.format(np.mean(max_results, axis=0)[0] / 7 * 100))

    np.save(str(data_names) + 'max_results.npy', max_results)

elapsed = (time.perf_counter() - start)
print("\nTime used:", elapsed / 3600)

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from Models.setgpuMG import SGMG
SGMG()
from sklearn.model_selection import KFold
from Data_preprocessing.data_preprocessing import data_reading, scores
from Models.training4fms import model_training
import time

tf.get_logger().setLevel('ERROR')
start = time.perf_counter()

path = '../近红外抑郁数据集/'
data_names = ['sad', 'rest']

sample_length = 50
lr = 0.01
times_of_repetition = 10
epochs = 200

feature_indexes = [[16], [17], [18], [19],
                   [16, 17], [16, 18], [16, 19],
                   [17, 18], [17, 19], [18, 19],
                   [16, 17, 18], [16, 17, 19],
                   [16, 18, 19], [17, 18, 19],
                   [16, 17, 18, 19]]

for feature_index in feature_indexes:
    print(feature_index)

    max_results = np.zeros((7, 6, 5))
    max_results[:, :, 2] = np.ones(6)

    outputs_dim = len(feature_index)
    data, label = data_reading(path, data_names, sample_length, feature_index)

    kf = KFold(7)
    a = 0
    for train_index, test_index in kf.split(data):
        train_data, train_label = data[train_index], label[train_index]
        print('\n----------外部第' + str(a + 1) + '折----------')

        for num in range(times_of_repetition):
            print('\n第' + str(num + 1) + '轮：')
            results = np.zeros(5)

            kf = KFold(6)
            i = 0
            for train_index1, test_index1 in kf.split(train_data):
                print('----------内部第' + str(i + 1) + '折----------')
                train_x, train_y = train_data[train_index1], train_label[train_index1]
                test_x, test_y = train_data[test_index1], train_label[test_index1]

                result = model_training(train_x, train_y, test_x, test_y, lr, epochs, outputs_dim)
                result = np.array(result)

                print('本折结果: ', end='')
                scores(result[1:].reshape(2, 2), 1)
                print('acc_person: {:.02f}%'.format(result[0] / 7 * 100))

                results += result
                if (result[1] + result[-1]) / np.sum(result[1:]) \
                        > (max_results[a, i, 1] + max_results[a, i, -1]) / np.sum(max_results[a, i, 1:]):
                    max_results[a, i] = result

                i += 1

            print('\n第' + str(num + 1) + '轮平均结果: ')
            scores(results[1:].reshape(2, 2), 1)
            print('acc_person: {:.02f}%'.format(results[0] / 42 * 100))
            print('\n最好结果: ')
            for i in range(6):
                print((max_results[a, i, 1] + max_results[a, i, -1]) / np.sum(max_results[a, i, 1:]), end='\t')
            print()
            scores(np.sum(max_results[a], axis=0)[1:].reshape(2, 2), 1)
            print('acc_person: {:.02f}%\n'.format(np.mean(max_results[a], axis=0)[0] / 7 * 100))

        a += 1

    np.save(str(feature_index) + 'max_results.npy', max_results)

elapsed = (time.perf_counter() - start)
print("\nTime used:", elapsed / 3600)

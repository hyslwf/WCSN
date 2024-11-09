import scipy.io as sio
import numpy as np


path = ''
fear = np.array(sio.loadmat(path + 'fear.mat')['fear'])
happy = np.array(sio.loadmat(path + 'happy.mat')['happy'])
sad = np.array(sio.loadmat(path + 'sad.mat')['sad'])
rest = np.array(sio.loadmat(path + 'rest.mat')['rest'])
label = np.array(sio.loadmat(path + 'label_simple.mat')['y'])

# 转置后顺序为人数、每个人的样本数、特征数、频域宽度、时域宽度
fear = np.transpose(fear, [3, 2, 4, 0, 1])
happy = np.transpose(happy, [3, 2, 4, 0, 1])
sad = np.transpose(sad, [3, 2, 4, 0, 1])
rest = np.transpose(rest, [3, 2, 4, 0, 1])

print(fear.shape, happy.shape, sad.shape, rest.shape, label.shape)

save_path = ''
np.save(save_path+'fear.npy', fear)
np.save(save_path+'happy.npy', happy)
np.save(save_path+'sad.npy', sad)
np.save(save_path+'rest.npy', rest)
np.save(save_path+'label_simple.npy', label)

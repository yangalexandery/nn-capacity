# import cPickle
import numpy as np

# def unpickle(file):
#     with open(file, 'rb') as fo:
#         dict = cPickle.load(fo)
#     return dict
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]

data_batch = []
label_batch = []

for file in files:
	d = unpickle('cifar-10-batches-py/' + file)
	d_c1 = np.reshape(d['data'][:, 0:1024], (d['data'].shape[0], 32, 32))
	d_c2 = np.reshape(d['data'][:, 1024:2048], (d['data'].shape[0], 32, 32))
	d_c3 = np.reshape(d['data'][:, 2048:3072], (d['data'].shape[0], 32, 32))
	l = np.array(d['labels'])
	d = np.zeros((d['data'].shape[0], 32, 32, 3))
	d[:,:,:,0] = d_c1
	d[:,:,:,1] = d_c2
	d[:,:,:,2] = d_c3
	data_batch.append(d)
	label_batch.append(l)

data = np.concatenate(data_batch, axis=0)
labels = np.concatenate(label_batch, axis=0)
print(data.shape)
print(labels.shape)

np.savez('data.npz', data, labels)
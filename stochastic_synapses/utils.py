import numpy as np
from skimage.transform import resize
import torch

from stochastic_synapses.parameters import param

# The data loader module in the network
def load_data(dataset, img_dim1, img_dim2):
	'''
	Load the MNIST dataset without a dataloader
	:param dataset: The dataset which we want: mnist or fmnist
	:param img_dim1: the height of the input image
	:param img_dim2: the width of the input image
	:return: new_train_data: The training data
	:return: train_label_bin: The labels in one hot encoded format
	:return: train_label: The labels in normal numerical format
	:return: new_test_data: The test data
	:return: test_label_bin: The test labels in one hot encoded format
	:return: test_label: The test labels
	'''
	from sklearn import preprocessing
	from tensorflow import keras

	np.set_printoptions(threshold=np.inf)
	if (dataset == 'mnist'):
		mnist = keras.datasets.mnist
		(train_data, train_label), (test_data, test_label) = mnist.load_data()
	else:
		fashion_mnist = keras.datasets.fashion_mnist
		(train_data, train_label), (test_data, test_label) = fashion_mnist.load_data()

	train_data = train_data.astype('float32') / 255
	test_data = test_data.astype('float32') / 255

	train_label = np.reshape(train_label, (np.size(train_data, 0), 1))
	encoder = preprocessing.OneHotEncoder(categories='auto')
	encoder.fit(train_label)
	train_label_bin = encoder.transform(train_label).toarray()
	test_label = np.reshape(test_label, (np.size(test_data, 0), 1))
	encoder.fit(test_label)
	test_label_bin = encoder.transform(test_label).toarray()
	new_train_data = np.zeros((np.size(train_data, 0), img_dim1, img_dim2))
	new_test_data = np.zeros((np.size(test_data, 0), img_dim1, img_dim2))
	if img_dim1 != np.size(train_data, 1):
		for i in range(np.size(train_data, 0)):
			new_train_data[i] = resize(train_data[i], (img_dim1, img_dim2))
		for i in range(np.size(test_data, 0)):
			new_test_data[i] = resize(test_data[i], (img_dim1, img_dim2))
	else:
		# if new_train_data is None:
		new_train_data = train_data
		new_test_data = test_data
	# print (train_label_bin.shape,new_train_data.shape,new_test_data.shape)
	return new_train_data, train_label_bin, train_label, new_test_data, test_label_bin, test_label


def load_dataloader():
	'''
	Load MNIST from the dataloader
	:return: train_loader: The training set loader
	:return: test_loader: The test set loader
	'''
	import torch
	from torchvision import datasets
	import torchvision.transforms as transforms

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])

	# choose the training and test datasets
	train_data = datasets.MNIST(root='data', train=True,
	                            download=True, transform=transform)
	test_data = datasets.MNIST(root='data', train=False,
	                           download=True, transform=transform)

	# prepare data loaders
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=1,
	                                           num_workers=0)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=1,
	                                          num_workers=0)
	return train_loader, test_loader



def generate_split(_taskid_, train_data, train_label):
	'''
	Split the MNIST dataset into different tasks
	:param taskID: the array presenting the classes for the current task
	:param train_data: Full training dataset
	:param train_label: The corresponding train labels
	:return: updated_train_data - The split training data
	:return: updated_train_label - The split training label
	'''
	return train_data[np.isin(train_label, _taskid_)], train_label[np.isin(train_label, _taskid_)] % param['n_classes']


def apply_transformation(_arr_):
	'''
	Adjusts the number of samples to the number of
	:param _arr_: The input array to be reshaped
	:return: modified_arr: Sliced down array to batch divisible metric
	'''
	val = _arr_.shape[0] % param['batch_size']
	if val != 0:
		return _arr_[:-val]
	else:
		return _arr_


def reformat_data(taskID, train_data, train_label):
	'''
	Reformat the MNIST data to Pytorch tensors
	:param taskID: the array presenting the classes for the current task
	:param train_data: ndarray training dataset
	:param train_label: The corresponding train labels
	:return: updated_train_data - The training data tensor
	:return: updated_train_label - The training label tensor
	'''
	if param['cuda']:
		_type_in_ = torch.cuda.FloatTensor
		_type_label_ = torch.cuda.LongTensor
	else:
		_type_in_ = torch.FloatTensor
		_type_label_ = torch.LongTensor
	split_train_data, split_train_label = generate_split(taskID, train_data, train_label.reshape(train_label.shape[0]))
	split_train_data = apply_transformation(split_train_data)
	split_train_label = apply_transformation(split_train_label)
	split_train_data = np.reshape(split_train_data,
	                              (int(split_train_data.shape[0] / param['batch_size']), param['batch_size'],
	                               split_train_data.shape[1] * split_train_data.shape[2]))
	split_train_label = np.reshape(split_train_label,
	                               (int(split_train_label.shape[0] / param['batch_size']), param['batch_size']))
	print(split_train_data.shape, split_train_label.shape)
	if isinstance(split_train_data, np.ndarray):
		split_train_data = torch.from_numpy(split_train_data).type(_type_in_)
		split_train_label = torch.from_numpy(split_train_label).type(_type_label_)
	return split_train_data, split_train_label


def get_mask(_arr_, val, flag):
	mask_arr = torch.zeros_like(_arr_)
	row, col = torch.where(_arr_ > val) if flag else torch.where(_arr_ < val)
	mask_arr[row, col] = 1
	return mask_arr

def update_probabilities(g,p):
	'''
	Update the probability values of the network
	:param g: The gradients of the selected layer
	:param p: The initial probability values of the weights
	:return: p: The updated probability
	'''
	# Compute the probability change values
	up_val = (1 - p) * param['p_up']
	down_val = (1 - p) * param['p_down']

	# First select p values that are < pfreeze
	p_mask = p.lt(param['p_freeze'])
	# Select the gradients within those whose value is > glim
	g_mask = torch.abs(g).gt(param['g_lim'])
	# print ((g_mask!=False).sum(), g_mask.shape, torch.max(g))

	# Update the p value for those gradients
	composite_mask = torch.mul(p_mask, g_mask)
	p[composite_mask] += up_val[composite_mask]

	# Reduce the probabilites for the gradients < glim
	non_composite_mask = torch.mul(p_mask, ~g_mask)
	p[non_composite_mask] -= down_val[non_composite_mask]

	# Update p as max(p_min, min(p,1))
	val = p[p_mask]
	val[val>1] = 1
	val[val<param['p_min']] = param['p_min']
	p[p_mask] = val

	return p

def convert_onehot(label):
	from sklearn import preprocessing

	label = np.reshape(label, (np.size(label, 0), 1))
	encoder = preprocessing.OneHotEncoder(categories='auto')
	encoder.fit(label)
	return encoder.transform(label).toarray()

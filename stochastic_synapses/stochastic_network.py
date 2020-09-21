import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from stochastic_synapses.parameters import param
from stochastic_synapses.utils import load_data, load_dataloader, update_probabilities, \
	reformat_data

if (param['dataloader']):
	trainloader, testloader = load_dataloader()
else:
	train_data, train_label_bin, train_label, test_data, test_label_bin, test_label = load_data(param['dataset'], 28,
	                                                                                            28)
print(train_data.shape, train_label.shape)


class base_model(nn.Module):
	'''
	The basic MLP model
	'''

	def __init__(self):
		super(base_model, self).__init__()
		self.fc1 = nn.Linear(784, 200)
		self.fc2 = nn.Linear(200, 200)
		self.fc3 = nn.Linear(200, param['n_classes'])

	def forward(self, x):
		x = F.relu(self.fc1(x.view(x.shape[0], -1)))
		x = F.relu(self.fc2(x))
		x = F.softmax(self.fc3(x))
		return x


class stochastic_model(nn.Module):
	'''
	The model with the stochastic synapses
	Variables starting with m are the trainable weight magnitudes.
	Variables starting with p are the weight probabilities.
	Variables starting with b are the bernoulli distributions of the probabilities
	'''

	def __init__(self):
		super(stochastic_model, self).__init__()
		self.m1 = torch.nn.Parameter(torch.randn(param['input_size'], param['hidden_size']), requires_grad=True)
		self.p1 = torch.nn.Parameter(torch.empty(param['input_size'], param['hidden_size']).fill_(0.25),
		                             requires_grad=False)
		self.b1 = torch.nn.Parameter(torch.bernoulli(self.p1), requires_grad=False)
		self.m2 = torch.nn.Parameter(torch.randn(param['hidden_size'], param['hidden_size']), requires_grad=True)
		self.p2 = torch.nn.Parameter(torch.empty(param['hidden_size'], param['hidden_size']).fill_(0.25),
		                             requires_grad=False)
		self.b2 = torch.nn.Parameter(torch.bernoulli(self.p2), requires_grad=False)
		self.m3 = torch.nn.Parameter(torch.randn(param['hidden_size'], param['n_classes']), requires_grad=True)
		self.p3 = torch.nn.Parameter(torch.empty(param['hidden_size'], param['n_classes']).fill_(0.25),
		                             requires_grad=False)
		self.b3 = torch.nn.Parameter(torch.bernoulli(self.p3), requires_grad=False)


	def forward(self, x):
		x = x.view(x.shape[0], -1)
		x = F.relu(x.mm(torch.div(torch.mul(self.m1, self.b1), (self.p1))))
		x = F.relu(x.mm(torch.div(torch.mul(self.m2, self.b2), (self.p2))))
		x = F.log_softmax(x.mm(torch.div(torch.mul(self.m3, self.b3), (self.p3))))
		return x


# Instantiating the model
if param['cuda']:
	model = stochastic_model().cuda()
else:
	model = stochastic_model()
# Loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=param['lr'])

# Arrange the task information for the network
taskID = np.arange(0, param['n_tasks'] * param['n_classes'], 1).reshape(param['n_tasks'], param['n_classes'])
if param['cuda']:
	_type_in_ = torch.cuda.FloatTensor
	_type_label_ = torch.cuda.LongTensor
else:
	_type_in_ = torch.FloatTensor
	_type_label_ = torch.LongTensor

# The accuracy matrix for the task
_acc_ = np.zeros((param['n_tasks'], param['n_tasks']))
''' 
# def grad_multiplier_hook(grad):
# 	grad_val = grad.clone()
# 	grad_val = torch.mul(grad_val, )
# print (param['p1'])
'''
# Storing the initial probability parameters
probability_lookup_list = []
for name, params in model.named_parameters():
	if name.__contains__('p'):
		probability_lookup_list.append(params)


def train_model(model, train_image, label):
	'''
	Train the binary net
	:param model: The specified model to be trained
	:param train_image: The selected training sample
	:param label: The corresponding label
	:return: loss: the accumulated loss of the model
	:return: output: The predicted output of the model
	'''
	# Generic forward pass
	gradient_lookup_list = []
	output = model(train_image)
	loss = criterion(output, label)
	optimizer.zero_grad()
	loss.backward()

	# Product of the gradients with the probability factor
	for name, param in model.named_parameters():
		if name.__contains__('m'):
			gradient_lookup_list.append(param.grad)
			param.grad = torch.mul(param.grad, (1-probability_lookup_list[int(name[1]) - 1].data))
	# Update the weights
	optimizer.step()

	# Update the network probabilities
	for name, param in model.named_parameters():
		if name.__contains__('p'):
			param.data = update_probabilities(gradient_lookup_list[int(name[1]) - 1].data, param.data)
			probability_lookup_list[int(name[1]) - 1] = param.data
	return loss, output


def calc_accuracy(model, test_images, test_labels):
	'''
	Perform inference on the model
	:param model: The specified model to be trained
	:param test_images: The test images in the model
 	:param test_labels: The test labels in the model
	:return: acc: The accuracy of the model
	'''
	output = model(test_images)
	return (output.argmax(dim=1).long() == test_labels).float().mean()

mean_accuracy = np.zeros((param['n_runs']))
mean_task_acc = np.zeros((param['n_runs'], param['n_tasks']))
for run in range(param['n_runs']):
	print ("Starting the evaluation run {} ...".format(run))
	for name, params in model.named_parameters():
		if name.__contains__('m'):
			nn.init.normal_(params.data, 0, 1)
		if name.__contains__('p'):
			nn.init.constant_(params.data, 0.25)
	print ("Done")
	for task in range(param['n_tasks']):
		print("Starting training on task {} {} ....".format(task + 1, taskID[task]))
		train_data_tensor, train_label_tensor = reformat_data(taskID[task], train_data, train_label)
		running_loss = 0.0
		for epochs in range(param['n_epochs']):
			count = 0.0
			for input_sample in tqdm(range(train_data_tensor.shape[0])):
				loss, output = train_model(model, train_data_tensor[input_sample], train_label_tensor[input_sample])
				count += (output.argmax(dim=1).long() == train_label_tensor[input_sample]).float().mean()
				running_loss += loss.item()

				if input_sample % 20 == 19:  # print every 20 mini-batches
					print('[%d, %5d] loss: %.3f  accuracy: %.2f' %
					      (epochs + 1, input_sample + 1, running_loss / 20, count / input_sample))
					running_loss = 0.0

			print("The training accuracy in this epoch --> {}".format(count / train_data_tensor.shape[0]))


		for subtask in range(task + 1):
			print("Testing on subtask ---> {}".format(subtask))
			test_data_tensor, test_label_tensor = reformat_data(taskID[subtask], test_data, test_label)
			for input_sample in tqdm(range(test_data_tensor.shape[0])):
				_acc_[task][subtask] += calc_accuracy(model, test_data_tensor[input_sample],
				                                      test_label_tensor[input_sample])
			print("Accuracy in this subtask --> {}".format(_acc_[task][subtask] / test_data_tensor.shape[0]))
			_acc_[task][subtask] /= test_data_tensor.shape[0]

	print("The accuracy matrix for the Split task \n {}".format(_acc_))

	print("The mean accuracy on the Split MNIST task is {}".format(np.mean(_acc_[-1])))
	mean_accuracy[run] = np.mean(_acc_[-1])
	mean_task_acc[run] = _acc_[-1]

print("The mean accuracy over the runs ---> \n {}".format(mean_accuracy))
print("The mean and std deviation in the network is --> \n Mean : {} \n STD : {}".format(np.mean(mean_accuracy),
                                                                                         np.std(mean_accuracy)))

print("The mean task accuracy across runs --> \n {}".format(mean_task_acc))
print("The mean and std for tasks is --> \n Mean : {} \n STD : {}".format(np.mean(mean_task_acc, axis=0),
                                                                          np.std(mean_task_acc, axis=0)))

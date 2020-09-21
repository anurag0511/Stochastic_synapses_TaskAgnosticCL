global param

param = {
	'dataset': 'mnist',     # mnist or fmnist dataset
	'cuda': 0,              # GPU (1) or CPU (0)
	'dataloader': 0,        # Load dataset from dataloader
	'n_tasks' : 5,          # Number of splits in the task
	'n_classes' : 2,        # Classes per task
	'n_epochs' : 10,        # Number of training cycles
	'n_runs': 1,            # Averaged over the number of runs
	'input_size': 784,      # The input size
	'hidden_size': 200,     # The hidden layer size
	'batch_size':1,         # Batch size
	'lr': 1e-3,             # Learning rate
	'loss': 'ce',           # New loss options will be added
	'p_freeze': 0.9353,     # The upper limit of transmission probability
	'g_lim': 1e-3,          # The lower limit of gradient value to be considered for probability update
	'p_up': 0.0520,         # The increase in transmission probability for the plastic neurons
	'p_down': 0.0516,       # The decrease in transmission probability for the important synapses
	'p_min': 0.25,          # The initial bernoulli distribution value for the transmission probabilities
	# Additional parameters for the local learning rules
	'beta1': 0.022,
	'beta2': 0.326,
	'beta3': 1.658,
	'base_lr': 1e-3,
	'uptake': 1
}

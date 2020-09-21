global param

param = {
	'dataset': 'mnist',
	'cuda': 0,
	'dataloader': 0,
	'n_tasks' : 5,
	'n_classes' : 2,
	'n_epochs' : 10,
	'n_runs': 1,
	'input_size': 784,
	'hidden_size': 200,
	'batch_size':1,
	'lr': 1e-3,
	'loss': 'ce',
	'p_freeze': 0.9353,
	'g_lim': 1e-3,
	'p_up': 0.0520,
	'p_down': 0.0516,
	'p_min': 0.25,
	'beta1': 0.022,
	'beta2': 0.326,
	'beta3': 1.658,
	'layer_sizes': [784, 500, 2],
	'layers': 2,
	'activation_flag': 1,
	'base_lr': 1e-3,
	'uptake': 1
}

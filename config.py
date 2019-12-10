class DefaultConfig(object):
	model = 'densenet'
	load_model_path =  ''
	data_root = '..'
	train_data_list = './data/trainSet.csv'
	valid_data_list = './data/validSet.csv'
	test_data_list = './data/testSet.csv'
	classes= ['Cardiomegaly','Atelectasis','Pleural Effusion','Consolidation','Edema']
	batch_size = 16
	check_freq = 2000
	result_file = 'result.csv'
	max_epoch = 10
	lr = .0001
	betas = (.9,.999)
	eps= 1e-08
	lr_decay = .95
	weight_decay = 1e-5
opt =  DefaultConfig()

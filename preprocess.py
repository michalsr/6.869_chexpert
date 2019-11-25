import pandas as pd
from config import opt

def preprocess():
	train_set()
def train_set():
	dataset_name = './data/train.csv'
	trainSet = load_dataset(dataset_name)
	trainSet.to_csv('./data/tranSet.csv',header=False,index=False,sep=' ')
def load_dataset(dataset_name):
	dataset = pd.read_csv(dataset_name)
	class_names = opt.classes
	columns = ['Path'] + class_names
	dataset = dataset[columns].fillna(0)
	dataset= dataset.replace(-1,1)
if __name__ == '__main__':
	preprocess()

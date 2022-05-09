import os

current_dir = os.getcwd()
data_dir = os.path.join(current_dir, 'data')
train_file = os.path.join(data_dir, 'train.csv')
test_file = os.path.join(data_dir, 'test.csv')
result_dir = os.path.join(current_dir, 'result')

os.system('Rscript ./intrees_baseline.R %s %s %s' % (train_file, test_file, result_dir))

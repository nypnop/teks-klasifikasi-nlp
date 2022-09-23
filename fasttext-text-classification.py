import pandas as pd
from gensim.utils import simple_preprocess
import fasttext

print("===============")
print("Loading the data training and data testing")

data_train = pd.read_csv('data_worthcheck/train.csv')[['text_a', 'label']].rename(columns= { 'text_a': 'text', 'label': 'class'})
data_test = pd.read_csv('data_worthcheck/test.csv')[['text_a', 'label']].rename(columns= { 'text_a': 'text', 'label': 'class'})

print("===============")
print("Showing head and info of the data")
data_train.info()
data_train.head()
data_test.info()
data_test.head()

data_train.iloc[:, 0] = data_train.iloc[:, 0].apply(lambda x: ' '.join(simple_preprocess(x)))
data_test.iloc[:, 0] = data_test.iloc[:, 0].apply(lambda x: ' '.join(simple_preprocess(x)))

data_train.iloc[:, 1] = data_train.iloc[:, 1].apply(lambda x: '__label__' + x)
data_test.iloc[:, 1] = data_test.iloc[:, 1].apply(lambda x: '__label__' + x)

data_train = data_train[['class', 'text']]
data_test = data_test[['class', 'text']]

# Looking at the DataFrames
data_train.head(5), data_test.tail(4)

data_train.to_csv('train.txt', index = False, sep = ' ', header = None)

data_test.to_csv('test.txt', index = False, sep = ' ', header = None)

# Training the fastText classifier
model = fasttext.train_supervised('train.txt', wordNgrams = 2, epoch=20)

# Evaluating performance on the entire test file
print(model.test('test.txt'))                      

# Predicting on a single input
print(model.predict(data_test.iloc[2, 0]))


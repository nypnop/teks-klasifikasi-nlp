import pandas as pd
from gensim.utils import simple_preprocess
import fasttext

from nltk.corpus import stopwords

print("===============")
print("Checkpoint 1 : Read the data training and data testing")

data_train = pd.read_csv('data_worthcheck/train.csv')[['text_a', 'label']].rename(columns= { 'text_a': 'text', 'label': 'class'})
data_test = pd.read_csv('data_worthcheck/test.csv')[['text_a', 'label']].rename(columns= { 'text_a': 'text', 'label': 'class'})

print("===============\n\n")

print("Checkpoint 2 : Showing head and info of the data\n")
print("data_train :")
print(data_train.info())
print(data_train.head(), "\n\n")
print("data_test :")
print(data_test.info())
print(data_test.head())

print("===============\n\n")
print("Checkpoint 3 data pre-processing")
stop_words = set(stopwords.words('indonesian'))

def text_clean(s: str) -> str:
    cleaned_text = ' '.join(word for word in simple_preprocess(s, deacc= True)if not word in stop_words)
    return cleaned_text

data_train.iloc[:, 0] = data_train.iloc[:, 0].apply(text_clean)
data_test.iloc[:, 0] = data_test.iloc[:, 0].apply(text_clean)

data_train.iloc[:, 1] = data_train.iloc[:, 1].apply(lambda x: '__label__' + x)
data_test.iloc[:, 1] = data_test.iloc[:, 1].apply(lambda x: '__label__' + x)

data_train = data_train[['class', 'text']]
data_test = data_test[['class', 'text']]


data_train.to_csv('train.txt', index = False, sep = ' ', header = None)

data_test.to_csv('test.txt', index = False, sep = ' ', header = None)

print("===============\n\n")
print("Checkpoint 5 : Data training and prediction")
# Training the fastText classifier
model = fasttext.train_supervised('train.txt', lr=1.0, dim=50,wordNgrams = 2, epoch=25)
print("\nSummarize")
def print_results(N, p, r):
    print("Total Data : \t" + str(N))
    print("Precision Score :\t{:.3f}".format(p))
    print("Recall Score :\t{:.3f}".format(r))

# Evaluating performance on the entire test file
print_results(*model.test('test.txt'))   
  
print("\nPredicting word")
# Predicting on a single input
print(data_test.iloc[1, 1])
print(model.predict(data_test.iloc[1, 1],k=-1))


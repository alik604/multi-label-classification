# %% [markdown]
# # This is a minmal code fork of `Machine learning.ipynb`
# With reduced output. Full output are in `Machine learning.ipynb`. Tt is on the "min scores 3" dateset, as it is much faster to run and there is not the point of failure of sharing models. 

# %%
# test_size = 0.2 # default and reported 
test_size = 0.5 # speed if you are in a hurry.

# %%
import random
import math
import sys
import gc
import re
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, SparsePCA, LatentDirichletAllocation, TruncatedSVD # aka LSA
from sklearn.preprocessing import *
from sklearn.preprocessing import MultiLabelBinarizer


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import * 


from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier # will not work for multi-label
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.tree import ExtraTreeClassifier as ET
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import torch
torch.cuda.is_available()

# %%
questions = pd.read_csv('./data/questions_preprocessed_min3.csv', index_col=0)
tags = pd.read_csv('./data/tags_preprocessed_min3.csv', index_col=0)

# %%
mlb = MultiLabelBinarizer()

list_to_encode = tags.Tag.values
mlb.fit(list_to_encode.reshape(-1,1))

encoded = dict() 
for i in tqdm(tags.index.unique()): 
    input_ = tags.loc[i, 'Tag']
    if (type(input_) is str):
        input_ = np.array(input_)
    else:   
        input_ = input_.values
    ouput_ = mlb.transform(input_.reshape(-1,1))
    ouput_ = list(np.sum(ouput_, axis=0))
    encoded[i] = ouput_

# %%
list_of_questions = []
list_of_tags = []
for i in tqdm(questions['Formatted'].index.unique()):
    list_of_questions.append(questions['Formatted'][i])
    list_of_tags.append(encoded[i])
print(f'length is {len(list_of_questions)}')

# %%
# some of list_of_questions are nan. Im not sure how to deal with this.
bad_idx = []
for idx, i in enumerate(list_of_questions):
    if type(i) is not str:
        if math.isnan(i):
            list_of_questions[idx] = list_of_questions[idx+1]
            list_of_tags[idx] = list_of_tags[idx+1] 

# %%
print(f'Sample question: {list_of_questions[0]}\n')
print(f'Number of tags: {sum(list_of_tags[0])}')

# %% [markdown]
# # Experiment 1: Encoding with Tokenizer, padding sequence

# %%
X_train, X_test, y_train, y_test = train_test_split(list_of_questions, list_of_tags, test_size=test_size , random_state=42)
y_train = np.asarray(y_train)
for i in range(len(y_train)):
    y_train[i] = np.array(y_train[i]) 
y_test = np.asarray(y_test)
for i in range(len(y_test)):
    y_test[i] = np.array(y_test[i])

# %%
tokenizer = Tokenizer(char_level=False, num_words=None, filters='!"()*+,-./:;<=>?@[\\]^_`{|}~\t\n') # #$%&
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
vocab_size = len(tokenizer.word_index) + 1

maxlen = 167 # code to get mean ommited in this file
X_train = pad_sequences(X_train, padding='post', truncating='post', maxlen=maxlen)
X_test = pad_sequences(X_test,   padding='post', truncating='post', maxlen=maxlen)


# %%
def print_score(y_test, y_pred, clf=None):
    with warnings.catch_warnings():
        # print("Clf: ", clf.__class__.__name__)
        print(f"Accuracy score: {accuracy_score(y_true=y_test, y_pred=y_pred)}")
        print(f"Recall score: {recall_score(y_true=y_test, y_pred=y_pred, average='weighted')}")
        print(f"Precision score: {precision_score(y_true=y_test, y_pred=y_pred, average='weighted')}")
        f1 = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')
        print(f"F1 score: {f1}")
        return f1

# %% [markdown]
# ## Experiment 1 assessment - sklearn

# %%
clfs = []
# clfs.append(DecisionTreeClassifier(random_state=42))
clfs.append(ExtraTreesClassifier(n_estimators=15, random_state=42, n_jobs=-1))
clfs.append(RandomForestClassifier(n_estimators=15, random_state=42, n_jobs=-1))
# clfs.append(RadiusNeighborsClassifier(n_jobs=-1)) # takes a while
# clfs.append(KNeighborsClassifier(n_neighbors=10, n_jobs=-1)) # untested too long

for clf in clfs:
    start = time.time()
    _ = gc.collect()
    _ = clf.fit(X_train, y_train)
    print("Clf: ", clf.__class__.__name__)
    F1 = print_score(y_test, clf.predict(X_test), clf)
    print(f'Train score: {clf.score(X_train, y_train)}')
    print(f'Test score:  {clf.score(X_test, y_test)}')

    print(f'Time taken for {clf.__class__.__name__} was {time.time()-start:.2f}\n\n')

# %%
start = time.time()

clf = MLPClassifier(hidden_layer_sizes=(200, 100), random_state=42, max_iter=60)

clf = clf.fit(X_train, y_train)
print("Clf: ", clf.__class__.__name__)
F1 = print_score(y_test, clf.predict(X_test), clf)
print(f'Train score: {clf.score(X_train, y_train)}')
print(f'Test score:  {clf.score(X_test, y_test)}')
    
print(f'Time taken for {clf.__class__.__name__} was {time.time()-start:.2f}')

plt.figure(figsize=(8, 3))
plt.plot(clf.loss_curve_)

# %% [markdown]
# ### In depth Eval of MLPClassifier, via argmax of `clf.predict_proba`

# %%
y_pred = clf.predict_proba(X_test)

y_pred_same_number_of_tags = [] # if average tags is 2.5, we are at a disadvantage if we do 2 or 3 tags.
for i in range(len(y_pred)):  
    best = y_pred[i].argsort()
    y_pred_ = np.zeros(y_pred.shape[1])
    number_of_tags = np.sum(y_test[i])
    y_pred_[best[-number_of_tags:]] = 1
    y_pred_same_number_of_tags.append(y_pred_)
    
_ = print_score(y_test, y_pred_same_number_of_tags) 
print("")

for i in range(5): # how many examples to print
    idx = np.random.randint(0, len(y_pred))
    print("y_pred:",', '.join(mlb.inverse_transform(y_pred_same_number_of_tags[idx].reshape(1,-1))[0]))
    print("y_test:",', '.join(mlb.inverse_transform(y_test[idx].reshape(1,-1))[0]))
    print('\n')

# %% [markdown]
# ## Experiment 1 assessment - Basic LSTM

# %%
vocab_size = len(tokenizer.word_index) + 1
target_size = len(mlb.classes_)
embedding_dim = 300

# %%
## Build glove embdedding matrix
embeddings_dictionary = dict()
# glove_file = open('.\data\glove.42B.300d.txt', encoding="utf8")
glove_file = open('C:\\glove.42B.300d.txt', encoding="utf8")
for line in glove_file: # longer, by far
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

num_words_in_embedding = 0
# embedding_matrix = np.zeros((vocab_size, embedding_dim))
embedding_matrix = np.random.normal(scale=0.6, size=((vocab_size, embedding_dim)))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        num_words_in_embedding += 1
        embedding_matrix[index] = embedding_vector
        
print(f'portion of words in embedding: {num_words_in_embedding/vocab_size:.4f}')

# %%
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from keras.layers.embeddings import Embedding
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Input
# del model, opt, lr

loss = []
acc = []
val_loss = []
val_acc = []
checkpoint_file = 'ModelCheckpoint.h5'

# %%
output_dim = mlb.classes_.shape[0]
model = Sequential() ;_=gc.collect()
model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False))
model.add((Bidirectional(LSTM(64, return_sequences=True))))
model.add(Dropout(0.05)) # technically not appropriate for a LSTM. recc_dropout is not on CuDNN
model.add(LSTM(128))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(output_dim, activation='sigmoid')) # , activation='softmax')

opt = Adam(learning_rate=0.003)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc']) # categorical_crossentropy
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.00001)
cp = ModelCheckpoint(filepath=checkpoint_file, monitor='val_acc', save_best_only=True)

# %%
# model = keras.models.load_model("keras_lstm.h5")
# model.layers[0].trainable = True
history = model.fit(X_train, y_train, validation_split=0.05, epochs=10, batch_size=256, verbose=1, callbacks=[lr, cp])
loss.extend(history.history['loss'])
acc.extend(history.history['acc'])
val_loss.extend(history.history['val_loss'])
val_acc.extend(history.history['val_acc'])

# %%
def test_keras_model(print_sample=False, X_test=X_test, model=model, y_test=y_test, topN=2):  
    y_pred = model.predict(X_test)
    y_pred_same_number_of_tags = [] # if average tags is 2.5, we are at a disadvantage if we do 2 or 3 tags.
    
    for i in range(len(y_pred)):  
        best = y_pred[i].argsort()
        y_pred_ = np.zeros(y_pred.shape[1])
        number_of_tags = np.sum(y_test[i])
        y_pred_[best[-number_of_tags:]] = 1
        y_pred_same_number_of_tags.append(y_pred_)
        
    _ = print_score(y_test, y_pred_same_number_of_tags) 

    # if print_sample:
    for i in range(print_sample):
        idx = np.random.randint(0, len(y_pred))
        print("y_pred:",', '.join(mlb.inverse_transform(y_pred_same_number_of_tags[idx].reshape(1,-1))[0]))
        print("y_test:",', '.join(mlb.inverse_transform(y_test[idx].reshape(1,-1))[0]))
        print('\n')

test_keras_model(print_sample=3)

# %%
_=gc.collect()
model.layers[0].trainable = True
history = model.fit(X_train, y_train, validation_split=0.05, epochs=5, batch_size=256, verbose=1, callbacks=[lr, cp])
# model.save("keras_lstm_min_3_score.h5")

loss.extend(history.history['loss'])
acc.extend(history.history['acc'])
val_loss.extend(history.history['val_loss'])
val_acc.extend(history.history['val_acc'])

# furture taining is omited from this file, I decreased LR and batch size

# %% [markdown]
# #### This will print the top five most likely tags, and then the 4 metrics, in a manner such the the network preds the same number of tags as the ground truth

# %%
model.load_weights(checkpoint_file)
y_pred = model.predict(X_test)

topN = 5
y_pred_top = np.zeros_like(y_pred)
for i in range(y_pred.shape[0]):
    best = y_pred[i].argsort()
    y_pred_ = np.zeros(y_pred.shape[1])
    y_pred_[best[-topN:]] = 1
    y_pred_top[i] = y_pred_

for i in range(5): # how many examples to print
    idx = np.random.randint(0, len(y_pred))
    print("y_pred:",', '.join(mlb.inverse_transform(y_pred_top[idx].reshape(1,-1))[0]))
    print("y_test:",', '.join(mlb.inverse_transform(y_test[idx].reshape(1,-1))[0]))
    print('\n')
    

y_pred_same_number_of_tags = [] # if average tags is 2.5, we are at a disadvantage if we do 2 or 3 tags.
for i in range(len(y_pred)):  
    best = y_pred[i].argsort()
    y_pred_ = np.zeros(y_pred.shape[1])
    number_of_tags = np.sum(y_test[i])
    y_pred_[best[-number_of_tags:]] = 1
    y_pred_same_number_of_tags.append(y_pred_)
    
print(f'LSTM model scores')
_ = print_score(y_test, y_pred_same_number_of_tags) 

# %% [markdown]
# ```
# on the all scores dataset "questions_preprocessed.csv"
# 2963/2963 [==============================] - 85s 29ms/step - loss: 0.0199 - acc: 0.6533 - val_loss: 0.0242 - val_acc: 0.6135 - lr: 5.0000e-04
# 
# Accuracy score: 0.6504909819639279
# Recall score: 0.7615028687766056
# Precision score: 0.7574494222705919
# F1 score: 0.7544918807953225
# 
# 
# on the min 3 scores dataset "questions_preprocessed_min3.csv"
# 
# 1902/1902 [==============================] - 47s 25ms/step - loss: 0.0195 - acc: 0.6716 - val_loss: 0.0313 - val_acc: 0.5739 - lr: 1.5000e-04
# 
# Accuracy score: 0.6123965651834504
# Recall score: 0.7221690839173125
# Precision score: 0.7139613396235852
# F1 score: 0.714711189087093
# 
# 
# ```


# %%
plt.figure(figsize=(12,8))
plt.title('LSTM w/ GloVe Loss BCE')
plt.plot(loss, label='train loss')
plt.plot(val_loss, label='val loss')
plt.legend()
plt.show()

# %% [markdown]
# # Expirement 2: Encoding with TfidfVectorizer
# > the following stats are for the `tags_preprocessed.csv` dataset
print(f'Expirement 2: Encoding with TfidfVectorizer\n\n\n\n')
# %%
stop_words = None # there are maybe 300 stop words, and thousands of other features. i'm inclined to keep them around
min_df=318 # 3378 features

vectorizer = TfidfVectorizer(strip_accents='unicode', ngram_range=(1, 1), stop_words=stop_words, min_df=min_df)
docs       = vectorizer.fit_transform(list_of_questions)
features   = vectorizer.get_feature_names()
print(f'The number of features is {len(features)}')

# %%
print(f'The number of samples is {docs.shape[0]} \nThe number of features is {docs.shape[1]}')

X_train, X_test, y_train, y_test = train_test_split(docs, list_of_tags, test_size=test_size, random_state=42)

y_train = np.asarray(y_train)
for i in range(len(y_train)):
    y_train[i] = np.array(y_train[i])
y_test = np.asarray(y_test)
for i in range(len(y_test)):
    y_test[i] = np.array(y_test[i])

# %%
clfs = []
# clfs.append(DecisionTreeClassifier(random_state=42))
clfs.append(ExtraTreesClassifier(n_estimators=10, random_state=42, n_jobs=3))
clfs.append(RandomForestClassifier(n_estimators=15, random_state=42, n_jobs=3))
# clfs.append(RadiusNeighborsClassifier(n_jobs=-1))

# clfs.append(KNeighborsClassifier(n_neighbors=10, n_jobs=-1)) # untested

for clf in clfs:
    start = time.time()
    _ = gc.collect()
    _ = clf.fit(X_train, y_train)
    
    print("Clf: ", clf.__class__.__name__)
    f1 = print_score(y_test, clf.predict(X_test))
    print(f'Train score: {clf.score(X_train, y_train)}')
    print(f'Test score:  {clf.score(X_test, y_test)}')

    print(f'Time taken for {clf.__class__.__name__} was {time.time()-start:.2f}\n\n')

# %%
print('Caution MLPClassifier can take long to train. I suggest max_iter=30 for performance, less for speed')
start = time.time()

clf = MLPClassifier(hidden_layer_sizes=(512, 100), random_state=42, max_iter=15)

clf = clf.fit(X_train, y_train)

print("Clf: ", clf.__class__.__name__)
f1 = print_score(y_test, clf.predict(X_test))
print(f'Train score: {clf.score(X_train, y_train)}')
print(f'Test score:  {clf.score(X_test, y_test)}')
    
print(f'Time taken for {clf.__class__.__name__} was {time.time()-start:.2f}')

plt.figure(figsize=(6,3))
plt.plot(clf.loss_curve_)

# This is very slow. 30 mins
# Clf:  MLPClassifier
# Accuracy score: 0.33423887587822015
# Recall score: 0.5944094165414121
# Precision score: 0.6495997459270161
# F1 score: 0.6191668671471201
# Train score: 0.9959952536339365
# Test score:  0.33423887587822015
# Time taken for MLPClassifier was 1801.59

# %%
del clfs
_=gc.collect()

# %%
y_pred = clf.predict(X_test[:])
for i in range(5):
    idx = np.random.randint(0, len(y_pred))
    print("y_pred:",', '.join(mlb.inverse_transform(y_pred[idx].reshape(1,-1))[0]))
    print("y_test:",', '.join(mlb.inverse_transform(y_test[idx].reshape(1,-1))[0]))
    print('\n')


# %%
y_pred_same_number_of_tags = [] # if average tags is 2.5, we are at a disadvantage if we do 2 or 3 tags.
for i in range(len(y_pred)):  
    best = y_pred[i].argsort()
    y_pred_ = np.zeros(y_pred.shape[1])
    number_of_tags = np.sum(y_test[i])
    y_pred_[best[-number_of_tags:]] = 1
    y_pred_same_number_of_tags.append(y_pred_)
_ = print_score(y_test, y_pred_same_number_of_tags) 


# Accuracy score: 0.41033567525370807
# Recall score: 0.5303246208202386
# Precision score: 0.6853284607898511
# F1 score: 0.585585014891463

# %%
# without ensuring the same number of tags, we are at a disadvantage if we do 2 or 3 tags.

print("Train set")
_ = print_score(y_train, clf.predict(X_train)) 

print("Test set")
_ = print_score(y_test, y_pred) 

# Train set
# Accuracy score: 0.9959952536339365
# Recall score: 0.9989239633364962
# Precision score: 0.998521101023201
# F1 score: 0.9987211981852843
# Test set
# Accuracy score: 0.33423887587822015
# Recall score: 0.5944094165414121
# Precision score: 0.6495997459270161
# F1 score: 0.6191668671471201

# %%




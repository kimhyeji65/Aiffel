#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import urllib.request
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import re
from konlpy.tag import Okt
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import os
import tensorflow as tf


# 데이터를 읽어봅시다. 
train_data = pd.read_table('~/aiffel/sentiment_classification/data/ratings_train.txt')
test_data = pd.read_table('~/aiffel/sentiment_classification/data/ratings_test.txt')



# In[2]:


train_data.head()


# # 2. data_loader 
# 
# 1)데이터의 중복 제거 
# 
# 2)NaN 결측치 제거
# 
# 3)한국어 토크나이저로 토큰화
# 
# 4)불용어(Stopwords) 제거
# 
# 5)사전word_to_index 구성
# 
# 6)텍스트 스트링을 사전 인덱스 스트링으로 변환
# 
# 7)X_train, y_train, X_test, y_test, word_to_index 리턴
# 

# In[3]:


from konlpy.tag import Mecab
tokenizer = Mecab()
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']


# In[4]:


def load_data(train_data, test_data, num_words=10000):
    train_data.drop_duplicates(subset=['document'], inplace=True)
    train_data = train_data.dropna(how = 'any') 
    test_data.drop_duplicates(subset=['document'], inplace=True)
    test_data = test_data.dropna(how = 'any') 
    
    X_train = []
    for sentence in train_data['document']:
        temp_X = tokenizer.morphs(sentence) # 토큰화
        temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
        X_train.append(temp_X)

    X_test = []
    for sentence in test_data['document']:
        temp_X = tokenizer.morphs(sentence) # 토큰화
        temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
        X_test.append(temp_X)
    
    words = np.concatenate(X_train).tolist()
    counter = Counter(words)
    counter = counter.most_common(10000-4)
    vocab = ['', '', '', ''] + [key for key, _ in counter]  # _ 없을 경우 사용
    word_to_index = {word:index for index, word in enumerate(vocab)}
    
    
        
    def wordlist_to_indexlist(wordlist):
        return [word_to_index[word] if word in word_to_index else word_to_index[''] for word in wordlist]
        
    X_train = list(map(wordlist_to_indexlist, X_train))
    X_test = list(map(wordlist_to_indexlist, X_test))
        
    return X_train, np.array(list(train_data['label'])), X_test, np.array(list(test_data['label'])), word_to_index
    
X_train, y_train, X_test, y_test, word_to_index = load_data(train_data, test_data) 



# In[5]:


index_to_word = {index:word for word, index in word_to_index.items()}


# In[6]:


# 문장 1개를 활용할 딕셔너리와 함께 주면, 단어 인덱스 리스트 벡터로 변환해 주는 함수입니다. 
# 단, 모든 문장은 <BOS>로 시작하는 것으로 합니다. 
def get_encoded_sentence(sentence, word_to_index):
    return [word_to_index['<BOS>']]+[word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in sentence.split()]

# 여러 개의 문장 리스트를 한꺼번에 단어 인덱스 리스트 벡터로 encode해 주는 함수입니다. 
def get_encoded_sentences(sentences, word_to_index):
    return [get_encoded_sentence(sentence, word_to_index) for sentence in sentences]

# 숫자 벡터로 encode된 문장을 원래대로 decode하는 함수입니다. 
def get_decoded_sentence(encoded_sentence, index_to_word):
    return ' '.join(index_to_word[index] if index in index_to_word else '<UNK>' for index in encoded_sentence[1:])  #[1:]를 통해 <BOS>를 제외

# 여러 개의 숫자 벡터로 encode된 문장을 한꺼번에 원래대로 decode하는 함수입니다. 
def get_decoded_sentences(encoded_sentences, index_to_word):
    return [get_decoded_sentence(encoded_sentence, index_to_word) for encoded_sentence in encoded_sentences]


# # 3. 모델구성을 위한 데이터 분석 및 가공
# 1)데이터셋 내 문장 길이 분포
# 
# 2)적절한 최대 문장 길이 지정
# 
# 3)keras.preprocessing.sequence.pad_sequences 을 활용한 패딩 추가
# 

# In[7]:


print(tf.__version__)
imdb = keras.datasets.imdb

# IMDb 데이터셋 다운로드 
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
print("훈련 샘플 개수: {}, 테스트 개수: {}".format(len(X_train), len(X_test)))


# In[8]:


print(X_train[0])  # 1번째 리뷰데이터
print('라벨: ', y_train[0])  # 1번째 리뷰데이터의 라벨
print('1번째 리뷰 문장 길이: ', len(X_train[0]))
print('2번째 리뷰 문장 길이: ', len(X_train[1]))


# In[9]:


word_to_index = imdb.get_word_index()
index_to_word = {index:word for word, index in word_to_index.items()}
print(index_to_word[1])     # 'the' 가 출력됩니다. 
print(word_to_index['the'])  # 1 이 출력됩니다.


# In[10]:


#실제 인코딩 인덱스는 제공된 word_to_index에서 index 기준으로 3씩 뒤로 밀려 있습니다.  
word_to_index = {k:(v+3) for k,v in word_to_index.items()}

# 처음 몇 개 인덱스는 사전에 정의되어 있습니다
word_to_index["<PAD>"] = 0
word_to_index["<BOS>"] = 1
word_to_index["<UNK>"] = 2  # unknown
word_to_index["<UNUSED>"] = 3

index_to_word[0] = "<PAD>"
index_to_word[1] = "<BOS>"
index_to_word[2] = "<UNK>"
index_to_word[3] = "<UNUSED>"

index_to_word = {index:word for word, index in word_to_index.items()}

print(index_to_word[1])     # '<BOS>' 가 출력됩니다. 
print(word_to_index['the'])  # 4 이 출력됩니다. 
print(index_to_word[4])     # 'the' 가 출력됩니다.


# In[11]:


total_data_text = list(X_train) + list(X_test)
# 텍스트데이터 문장길이의 리스트를 생성한 후
num_tokens = [len(tokens) for tokens in total_data_text]
num_tokens = np.array(num_tokens)
# 문장길이의 평균값, 최대값, 표준편차를 계산해 본다. 
print('문장길이 평균 : ', np.mean(num_tokens))
print('문장길이 최대 : ', np.max(num_tokens))
print('문장길이 표준편차 : ', np.std(num_tokens))


# In[12]:


# 예를들어, 최대 길이를 (평균 + 2*표준편차)로 한다면,  
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
maxlen = int(max_tokens)
print('pad_sequences maxlen : ', maxlen)
print('전체 문장의 {}%가 maxlen 설정값 이내에 포함됩니다. '.format(np.sum(num_tokens < max_tokens) / len(num_tokens)))


# In[13]:


X_train = keras.preprocessing.sequence.pad_sequences(X_train,
                                                        value=word_to_index["<PAD>"],
                                                        padding='post', # 혹은 'pre'
                                                        maxlen=maxlen)

X_test = keras.preprocessing.sequence.pad_sequences(X_test,
                                                       value=word_to_index["<PAD>"],
                                                       padding='post', # 혹은 'pre'
                                                       maxlen=maxlen)

print(X_train.shape)
print(X_test.shape)


# # 4. 모델구성 및 validation set 구성

# In[22]:


from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model


# In[23]:


# validation set 20000건 분리
X_val = X_train[:20000]   
y_val = y_train[:20000]

# validation set을 제외한 나머지 15000건
partial_X_train = X_train[10000:]  
partial_y_train = y_train[10000:]

print(partial_X_train.shape)
print(partial_y_train.shape)


# In[25]:


#모델구성
vocab_size = 10000
word_vector_dim = 150

#LSTM
LSTM = keras.Sequential()
LSTM.add(keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,)))
LSTM.add(keras.layers.LSTM(150))
LSTM.add(keras.layers.Dense(10, activation='relu'))
LSTM.add(keras.layers.Dense(1, activation='sigmoid'))

LSTM.summary()

# 1-D CNN
CNN = keras.Sequential()
CNN.add(keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,))) 
CNN.add(keras.layers.Conv1D(16, 7, activation='relu'))
CNN.add(keras.layers.MaxPooling1D(5))
CNN.add(keras.layers.Conv1D(16, 7, activation='relu'))
CNN.add(keras.layers.GlobalMaxPooling1D())
CNN.add(keras.layers.Dense(16, activation='relu'))
CNN.add(keras.layers.Dense(1, activation='sigmoid'))

CNN.summary()


# In[26]:


#모델학습

epochs = 5

LSTM.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history_LSTM = LSTM.fit(partial_X_train,
                    partial_y_train,
                    epochs=epochs,
                    batch_size=512,
                    validation_data=(X_val, y_val),
                    verbose=1)
              
epochs = 3

CNN.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


history_CNN = CNN.fit(partial_X_train,
                    partial_y_train,
                    epochs=epochs,
                    batch_size=512,
                    validation_data=(X_val, y_val),
                    verbose=1)


# In[27]:


#모델평가

# LSTM
print("LSTM")
results_LSTM = LSTM.evaluate(X_test, y_test, verbose=2)
print(results_LSTM)


# 1-D CNN
print("1-D CNN")
results_CNN = CNN.evaluate(X_test, y_test, verbose=2)
print(results_CNN)


# # 5. 그래프 시각화

# In[28]:


#LSTM 
history_dict = history_LSTM.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 그림을 초기화
plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[29]:


#1-D CNN

history_dict = history_CNN.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#그림 초기화
plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# # 6. Embedding 레이어 분석

# In[30]:


from gensim.models.keyedvectors import Word2VecKeyedVectors


embedding_LSTM = LSTM.layers[0]
weights_LSTM = embedding_LSTM.get_weights()[0]

embedding_CNN = CNN.layers[0]
weights_CNN = embedding_CNN.get_weights()[0]


#LSTM
word2vec_file_path_LSTM = os.getenv('HOME')+'/aiffel/sentiment_classification/word2vec_lstm.txt'
f = open(word2vec_file_path_LSTM, 'w')
f.write('{} {}\n'.format(vocab_size-4, word_vector_dim))

vectors = LSTM.get_weights()[0]
for i in range(4,vocab_size):
    f.write('{} {}\n'.format(index_to_word[i], ' '.join(map(str, list(vectors[i, :])))))
f.close()

#1-D CNN
word2vec_file_path_CNN = os.getenv('HOME')+'/aiffel/sentiment_classification/word2vec_cnn.txt'
f = open(word2vec_file_path_CNN, 'w')
f.write('{} {}\n'.format(vocab_size-4, word_vector_dim))

vectors = CNN.get_weights()[0]
for i in range(4,vocab_size):
    f.write('{} {}\n'.format(index_to_word[i], ' '.join(map(str, list(vectors[i, :])))))
f.close()



# In[31]:


word_vectors_LSTM = Word2VecKeyedVectors.load_word2vec_format(word2vec_file_path_LSTM, binary=False)
vector_LSTM = word_vectors_LSTM['computer']
vector_LSTM


# In[32]:


word_vectors_CNN = Word2VecKeyedVectors.load_word2vec_format(word2vec_file_path_CNN, binary=False)
vector_CNN = word_vectors_CNN['computer']
vector_CNN


# In[73]:


word_vectors_LSTM.similar_by_word("movie")


# In[74]:


word_vectors_CNN.similar_by_word("movie")


# # 7. 한국어 Word2Vec 임베딩 활용하여 성능개선

# In[33]:


import gensim
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors


# In[34]:


word2vec_path = os.getenv('HOME')+'/aiffel/sentiment_classification/ko.bin'
word2vec = gensim.models.Word2Vec.load(word2vec_path)
vector = word2vec['컴퓨터']
vector


# In[35]:


word2vec.similar_by_word("인사")


# In[37]:


word2vec.similar_by_word("영화")


# In[49]:


# 모델 설계
from tensorflow.keras.initializers import Constant

vocab_size = 10000    # 어휘 사전의 크기입니다(10,000개의 단어)
word_vector_dim = 300  # 워드 벡터의 차원수(변경가능 파라미터)
embedding_matrix = np.random.rand(vocab_size, word_vector_dim)

# embedding_matrix에 Word2Vec 워드 벡터를 단어 하나씩마다 차례차례 카피한다.
for i in range(4,vocab_size):
    if index_to_word[i] in word2vec:
        embedding_matrix[i] = word2vec[index_to_word[i]]


# In[44]:






# 모델 구성
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 
                                 word_vector_dim, 
                                 embeddings_initializer=Constant(embedding_matrix),  # 카피한 임베딩을 여기서 활용
                                 input_length=maxlen, 
                                 trainable=True))   # trainable을 True로 주면 Fine-tuning
model.add(keras.layers.LSTM(16))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()


# In[48]:


# 학습의 진행
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
              
epochs=20  # 몇 epoch를 훈련하면 좋을지 결과를 보면서 바꾸어 봅시다. 

history = model.fit(X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=700,
                    validation_data=(X_val, y_val),
                    verbose=1)


# In[52]:


results = model.evaluate(X_test,  y_test, verbose=2)

print(results)


# # 결과
# 
# LSTM은 0.511, 1-D CNN은 0.847 이 나왔다.
# 
# 
# 한국어 Word2Vec에서 LSTM을 사용 했을 때는 LSTM 은 0.8804로 성능이 조금 더 향상 되었음을 볼 수 있었다.

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # 1. 가위바위보 분류기 만들기

# # 데이터 불러오기 + Resize하기
# (minist 숫자 손글씨 Dataset이 28 x28 이기 때문에 이미지도 28 x28 만들기)
# 
# 

# In[6]:


get_ipython().system('pip install pillow   ')
from PIL import Image
import glob
import os
def resize_images(img_path):
    images=glob.glob(img_path + "/*.jpg")

    target_size=(28,28)
    for img in images:
        old_img=Image.open(img)
        new_img=old_img.resize(target_size,Image.ANTIALIAS)
        new_img.save(img, "JPEG")

image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/scissor"
resize_images(image_dir_path)
image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/rock"
resize_images(image_dir_path)
image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/paper"
resize_images(image_dir_path)

print("가위, 바위, 보  이미지 resize 완료!")


# # 학습 및 정규화
# 이미지들의 정보를 행렬에 담아준 뒤 지도학습을 하기위해 라벨링 시키고, 0-1 사이값으로 정규화 해준다.
# 

# In[3]:


# load_data 함수
import numpy as np

def load_data(img_path, number_of_data=300):  # 가위바위보 이미지 개수 총합에 주의
    # 가위 : 0, 바위 : 1, 보 : 2
    img_size=28
    color=3
    #이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels=np.zeros(number_of_data,dtype=np.int32)

    idx=0
    for file in glob.iglob(img_path+'/scissor/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   #가위 :0
        idx=idx+1

    for file in glob.iglob(img_path+'/rock/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1  
    
    for file in glob.iglob(img_path+'/paper/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1
        
    print("학습데이터(x_train)의 이미지 개수는", idx,"입니다.")
    return imgs, labels

image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper"
(x_train, y_train)=load_data(image_dir_path)
x_train_norm = x_train/255.0   # 입력은 0~1 사이의 값으로 정규화

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))


# # 딥러닝 네트워크 설계
# 학습시킬 모델 만들기, 파라미터를 갱신시키는 optimizer 종류 중 adam 사용

# In[36]:


# 딥러닝 네트워크 설계
import tensorflow as tf
from tensorflow import keras
import numpy as np

channel_1=16
channel_2=32
dense=32
train_epoch=10


model=keras.models.Sequential()
model.add(keras.layers.Conv2D(channel_1, (3,3), activation='relu', input_shape=(28,28,3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(channel_2, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(dense, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))
model.summary()




# # Train
# model을 compile 한 후 fit 함수 사용해서 학습시킨다

# In[35]:


# 모델 학습
model.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

model.fit(x_train_norm, y_train, epochs=10)


# # Test
# 
# 

# In[25]:



from PIL import Image
import glob
import os

def resize_images(img_path):
  images=glob.glob(img_path + "/*.jpg")

  target_size=(28,28)
  for img in images:
      old_img=Image.open(img)
      new_img=old_img.resize(target_size,Image.ANTIALIAS)
      new_img.save(img, "JPEG")

image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test/scissor"
resize_images(image_dir_path)   # 테스트 가위 이미지 resize

image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test/rock"
resize_images(image_dir_path)   # 테스트 바위 이미지 resize

image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test/paper"
resize_images(image_dir_path)   # 테스트 보 이미지 resize


# In[26]:


image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test"
(x_test, y_test)=load_data(image_dir_path)
print("x_train shape: {}".format(x_test.shape))
print("y_train shape: {}".format(y_test.shape))


# In[31]:


# 모델 테스트

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose = 2)

print("test_loss : {}".format(test_loss))
print("test_accuracy : {}".format(test_accuracy))


# accuracy가 33% 밖에 나오지 않는다. 
# 잘 못 추론한 데이터를 확인해 보자.

# In[32]:


predicted_result = model.predict(x_test)  
predicted_labels = np.argmax(predicted_result, axis=1)


# In[33]:


import random
wrong_predict_list=[]
 # i번째 test_labels과 y_test이 다른 경우
for i, _ in enumerate(predicted_labels):   
    if predicted_labels[i] != y_test[i]:
        wrong_predict_list.append(i)

# wrong_predict_list 에서 랜덤하게 5개 뽑기
samples = random.choices(population=wrong_predict_list, k=5)

for n in samples:
    print("예측확률분포: " + str(predicted_result[n]))
    print("라벨: " + str(y_test[n]) + ", 예측결과: " + str(predicted_labels[n]))
    plt.imshow(x_test[n], cmap=plt.cm.binary)
    plt.show()


# # 결론:
# train 한 개수도 부족했지만, 이미지 크기를 28 x 28 로 변경하면서 해상도가 알아보지 못 할 정도가 됐다.
# 하이퍼파라미터 값을 바꿔가면서 다시 해봐야 할 것 같다.
# 

# In[ ]:





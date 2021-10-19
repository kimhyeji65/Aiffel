#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import os
import tarfile
import urllib

from matplotlib import pyplot as plt
import tensorflow as tf


# In[6]:


import os
img_path = os.getenv('HOME')+'/aiffel/human_segmentation/images/my_image.jpeg'  # 본인이 선택한 이미지의 경로에 맞게 바꿔 주세요. 
img_orig = cv2.imread(img_path) 
print (img_orig.shape)


# In[7]:


class DeepLabModel(object):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    # __init__()에서 모델 구조를 직접 구현하는 대신, tar file에서 읽어들인 그래프구조 graph_def를 
    # tf.compat.v1.import_graph_def를 통해 불러들여 활용하게 됩니다. 
    def __init__(self, tarball_path):
        self.graph = tf.Graph()
        graph_def = None
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
                break
        tar_file.close()

        with self.graph.as_default():
    	    tf.compat.v1.import_graph_def(graph_def, name='')

        self.sess = tf.compat.v1.Session(graph=self.graph)

    # 이미지를 전처리하여 Tensorflow 입력으로 사용 가능한 shape의 Numpy Array로 변환합니다.
    def preprocess(self, img_orig):
        height, width = img_orig.shape[:2]
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = cv2.resize(img_orig, target_size)
        resized_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        img_input = resized_rgb
        return img_input
        
    def run(self, image):
        img_input = self.preprocess(image)

        # Tensorflow V1에서는 model(input) 방식이 아니라 sess.run(feed_dict={input...}) 방식을 활용합니다.
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [img_input]})

        seg_map = batch_seg_map[0]
        return cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR), seg_map


# In[8]:


# define model and download & load pretrained weight
_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'

model_dir = os.getenv('HOME')+'/aiffel/human_segmentation/models'
tf.io.gfile.makedirs(model_dir)

print ('temp directory:', model_dir)

download_path = os.path.join(model_dir, 'deeplab_model.tar.gz')
if not os.path.exists(download_path):
    urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + 'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
                   download_path)

MODEL = DeepLabModel(download_path)
print('model loaded successfully!')


# In[9]:


LABEL_NAMES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
]
len(LABEL_NAMES)


# In[10]:


img_resized, seg_map = MODEL.run(img_orig)
print (img_orig.shape, img_resized.shape, seg_map.max())


# In[11]:


img_show = img_resized.copy()
seg_map = np.where(seg_map == 15, 15, 0) # 예측 중 사람만 추출
img_mask = seg_map * (255/seg_map.max()) # 255 normalization
img_mask = img_mask.astype(np.uint8)
color_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)
img_show = cv2.addWeighted(img_show, 0.6, color_mask, 0.35, 0.0)

plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()


# In[12]:


img_mask_up = cv2.resize(img_mask, img_orig.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
_, img_mask_up = cv2.threshold(img_mask_up, 128, 255, cv2.THRESH_BINARY)

ax = plt.subplot(1,2,1)
plt.imshow(img_mask_up, cmap=plt.cm.binary_r)
ax.set_title('Original Size Mask')

ax = plt.subplot(1,2,2)
plt.imshow(img_mask, cmap=plt.cm.binary_r)
ax.set_title('DeepLab Model Mask')

plt.show()


# In[13]:


img_orig_blur = cv2.blur(img_orig, (13,13)) #(13,13)은 blurring  kernel size를 뜻합니다. 
plt.imshow(cv2.cvtColor(img_orig_blur, cv2.COLOR_BGR2RGB))
plt.show()


# In[14]:


img_mask_color = cv2.cvtColor(img_mask_up, cv2.COLOR_GRAY2BGR)
img_bg_mask = cv2.bitwise_not(img_mask_color)
img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask)
plt.imshow(cv2.cvtColor(img_bg_blur, cv2.COLOR_BGR2RGB))
plt.show()


# In[15]:


img_concat = np.where(img_mask_color==255, img_orig, img_bg_blur)
plt.imshow(cv2.cvtColor(img_concat, cv2.COLOR_BGR2RGB))
plt.show()


# # 사진 문제점
# 경계가 불분명 하기 때문에 블러가 핸드폰까지 처리가 되었다. 

# In[26]:


plt.figure(figsize=(5,5))

ax = plt.subplot(1,1,1)

img_concat_resized = cv2.resize(img_concat, (500, 500))
plt.imshow(cv2.cvtColor(img_concat_resized, cv2.COLOR_BGR2RGB))
ax.set_title('DeepLab')


# In[37]:


problem = cv2.imread(os.path.join(img_path, "Problem.jpeg")) 
plt.figure(figsize=(10, 10))
#plt.imshow(cv2.cvtColor(problem, cv2.COLOR_BGR2RGB))
#이미지를 넣었는데, 이미지가 없다고 나옴 


# In[38]:


#가우시안 블러 이용 : 경계선 약화

seg_map = np.where(seg_map == 15, 15, 0) # 예측 중 사람만 추출
img_mask = seg_map * (255/seg_map.max()) # 255 normalization
img_mask = img_mask.astype(np.uint8)


# 원본 사진, Mask 크기 맞추기
img_mask_up = cv2.resize(img_mask, img_orig.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
# 원본 사진처럼 3 channel 만들기
img_mask_c = np.repeat(np.expand_dims(img_mask_up, axis=2), 3, axis=2).astype(np.uint8)



# In[40]:


# 원본 이미지 가우시안 블러 처리
img_orig_blur = cv2.GaussianBlur(img_orig, (25,25),0)
# mask image를 가우시안 블러 처리
img_gb_mask = (cv2.GaussianBlur(img_mask_c, (101, 101), 25, 25)/255).astype(np.float32)


# In[31]:


# 마스크 + 이미지 
img_image_blur = img_gb_mask*img_orig.astype(np.float32)
# 1-img_gb_mask : 블러 처리한 원본에서 배경만 뽑아내기 위해 
img_bg_mask = (1-img_gb_mask)*img_orig_blur.astype(np.float32)
out = (img_image_blur+img_bg_mask).astype(np.uint8)


# In[33]:




fig = plt.figure(figsize=(20, 20)) 

ax = plt.subplot(3,1,1)
plt.imshow(cv2.cvtColor(img_image_blur.astype(np.uint8),cv2.COLOR_BGR2RGB))
ax.set_title('img blur')

ax = plt.subplot(3,1,2)
plt.imshow(cv2.cvtColor(img_bg_mask.astype(np.uint8),cv2.COLOR_BGR2RGB))
ax.set_title('img background')

ax = plt.subplot(3,1,3)
plt.imshow(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))
ax.set_title('img background')

plt.show()


# # 크로마키 배경 제작

# In[86]:


img_path = os.getenv('HOME')+ '/aiffel/human_segmentation/images/my_image.jpeg' 
img_orig = cv2.imread(img_path)
img_orig = cv2.resize(img_orig, (335, 512)) # 풍경 사이즈와 맞게 resize
print (img_orig.shape)


# In[88]:


img_resized, seg_map = MODEL.run(img_orig)
print (img_orig.shape, img_resized.shape, seg_map.max())


# In[89]:


img_show = img_resized.copy()
seg_map = np.where(seg_map == 15, 15, 0) # 예측 중 사람만 추출
img_mask = seg_map * (255/seg_map.max()) # 255 normalization
img_mask = img_mask.astype(np.uint8)
color_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)
img_show = cv2.addWeighted(img_show, 0.6, color_mask, 0.35, 0.0)

plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()


# In[90]:


img_mask_up = cv2.resize(img_mask, img_orig.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
_, img_mask_up = cv2.threshold(img_mask_up, 128, 255, cv2.THRESH_BINARY)

ax = plt.subplot(1,2,1)
plt.imshow(img_mask_up, cmap=plt.cm.binary_r)
ax.set_title('Original Size Mask')

ax = plt.subplot(1,2,2)
plt.imshow(img_mask, cmap=plt.cm.binary_r)
ax.set_title('DeepLab Model Mask')

plt.show()


# In[112]:


img_mask_color = cv2.cvtColor(img_mask_up, cv2.COLOR_GRAY2BGR)
img_bg_mask = cv2.bitwise_not(img_mask_color)
img_bg = cv2.bitwise_and(img_orig, img_bg_mask)

plt.imshow(img_bg)
plt.show()


# In[124]:


bg_path = os.getenv('HOME')+'/aiffel/human_segmentation/images/background.jpeg'  
bg_orig = cv2.imread(bg_path)
bg_orig = cv2.resize(img_orig, (335, 512))
print (bg_orig.shape)
img_concat = np.where(img_bg==0, img_bg, bg_orig)
plt.imshow(img_concat)
plt.show()


# In[119]:


cromakey = np.where(img_mask_color==255, img_orig, img_concat)
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(cromakey, cv2.COLOR_BGR2RGB))
plt.show()


# In[122]:


cat_resized = cv2.resize(
    img_orig, (bg_orig.shape[1], img_orig.shape[0]))
serengeti_img_resized = cv2.resize(
    bg_orig, (bg_orig.shape[1], img_orig.shape[0]))
cat_img_mask_resized = cv2.resize(
    img_mask_color, (bg_orig.shape[1], img_orig.shape[0]))

cat_serengeti_concat = np.where(
    cat_img_mask_resized == 255, cat_resized, serengeti_img_resized)

plt.imshow(cv2.cvtColor(cat_serengeti_concat, cv2.COLOR_BGR2RGB))
plt.show()


# # 결과
# 1. ValueError: operands could not be broadcast together with shapes
# 
#     크로마키를 할 때 이미지끼리 배열이 달랐던 것 같다. 
#     broadcast가 되지 못했다는 의미이며, 조건이 맞을 경우에 모양이 다른 배열끼리 연산이 가능하다는 얘기이다.
#     단, 1차원 경우에 가능
# 
# 2. Segmentation에서 이미지 Mask가 정확이 되어야 하는데 안돼서 블러처리가 모호하게 되는 것 같다.
# 
# 3.가우시안 블러를 사용하면서 경계가 자연스럽다. 하지만 여전히 폰에 블러처리가 되어있다.

# In[ ]:





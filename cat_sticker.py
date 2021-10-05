#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


# 이미지 불러오기
img_path = os.getenv('HOME') + '/aiffel/camera_sticker/images/image.png'
img_bgr = cv2.imread(img_path)
# 출력용 이미지 보관
img_show = img_bgr.copy()       
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.show()


# In[4]:


# detector를 선언하여 bounding box 추출
detector_hog = dlib.get_frontal_face_detector()
 # 얼굴 bounding box 좌표
dlib_rects = detector_hog(img_rgb, 1)  


# In[5]:


# 얼굴 bounding box 
for dlib_rect in dlib_rects:
    l = dlib_rect.left()
    t = dlib_rect.top()
    r = dlib_rect.right()
    b = dlib_rect.bottom()
cv2.rectangle(img_show, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA)

img_show_rgb =  cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
plt.imshow(img_show_rgb)
plt.show()


# In[6]:



# 저장한 landmark 모델을 불러오기
model_path = os.getenv('HOME')+'/aiffel/camera_sticker/models/shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(model_path)


# In[7]:


# parts() 함수로 개별 위치에 접근 => (x,y)형태로 변환
list_landmarks = []
for dlib_rect in dlib_rects:
    points = landmark_predictor(img_rgb, dlib_rect)
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))
    list_landmarks.append(list_points)
    
print(len(list_landmarks[0])) 
# 얼굴에서 68개 랜드마크를 모두 검출하면 68 출력한다

for landmark in list_landmarks:
    for idx, point in enumerate(list_points):
        cv2.circle(img_show, point, 2, (0, 255, 255), -1)

img_show_rgb = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
plt.imshow(img_show_rgb)
plt.show()


# In[8]:



# 코의 좌표 확인하고, 스티커의 위치 지정
for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
    #print(landmark[30])  #(522, 438)
    
    x = landmark[33][0] - 3
    y = landmark[33][1] - 3
    w = dlib_rect.width()
    h = dlib_rect.width()
    
    print ('(x,y) : (%d,%d)'%(x,y))
    print ('(w,h) : (%d,%d)'%(w,h))


# In[4]:


# 스티커 이미지 사이즈 수정

sticker_path = os.getenv('HOME')+'/aiffel/camera_sticker/images/cat.png'
img_sticker = cv2.imread(sticker_path)
img_sticker = cv2.resize(img_sticker, (w , h ))
print (img_sticker.shape)

rows, cols = img_sticker.shape[:2]

# 이미지의 중심점을 기준으로 90도 회전 하면서 0.5배 Scale
'''
M= cv2.getRotationMatrix2D((cols/2, rows/2), 45, 0.5)

dst = cv2.warpAffine(img_sticker, M,(cols, rows))

cv2.imshow('Original', img_sticker)
cv2.imshow('Rotation', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# 이미지 시작점이 left-top 이니깐, 스티커 이미지의 x,y좌표 조정하기 
refined_x = x - w // 2  # left
refined_y = y - h //2 # top

print ('(x,y) : (%d,%d)'%(refined_x, refined_y))


# In[ ]:





# In[1]:


# 스티커 이미지 영역 잡기
sticker_area = img_show[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]


# In[140]:


# 스티커를 이미지에 적용
img_show[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]     = np.where(img_sticker == 255, sticker_area, img_sticker).astype(np.uint8)
plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()


# In[142]:



#bounding box와 landmark를 제거
sticker_area = img_bgr[refined_y:img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
img_bgr[refined_y:img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] =     np.where(img_sticker==0,sticker_area,img_sticker).astype(np.uint8)
plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
plt.show()


# #  #다시 봐야 하는  것 :
# # 얼굴 각도에 따른 스티커 조정
# cv2.getRotationMatrix2D( ) 라는 method로 스티커 회전 시키기 
# # 왜 스티커가 계속 여러 개 떠 있는 지 이유를 찾기

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # 1) Processing

# In[2]:


import pandas as pd
import os
rating_file_path=os.getenv('HOME') + '/aiffel/recommendata_iu/data/ml-1m/ratings.dat'
ratings_cols = ['user_id', 'movie_id', 'ratings', 'timestamp']
ratings = pd.read_csv(rating_file_path, sep='::', names=ratings_cols, engine='python', encoding = "ISO-8859-1")
orginal_data_size = len(ratings)
ratings.head()


# In[3]:


# 3점 이상만 남깁니다.
ratings = ratings[ratings['ratings']>=3]
filtered_data_size = len(ratings)

print(f'orginal_data_size: {orginal_data_size}, filtered_data_size: {filtered_data_size}')
print(f'Ratio of Remaining Data is {filtered_data_size / orginal_data_size:.2%}')


# In[4]:


# ratings 컬럼의 이름을 counts로 바꿉니다.
ratings.rename(columns={'ratings':'counts'}, inplace=True)


# In[5]:


ratings['counts']


# In[6]:


# 영화 제목을 보기 위해 메타 데이터를 읽어옵니다.
movie_file_path=os.getenv('HOME') + '/aiffel/recommendata_iu/data/ml-1m/movies.dat'
cols = ['movie_id', 'title', 'genre'] 
movies = pd.read_csv(movie_file_path, sep='::', names=cols, engine='python', encoding='ISO-8859-1')
movies.head()


# In[7]:


# 영화를 title로 입력할 것이기때문에 title이란 컬럼을 불러와야합니다.
ratings = pd.merge(ratings, movies[['title', 'movie_id']], on='movie_id', how='left')
ratings


# # 2) 분석해 봅시다.
# 
# 1.ratings에 있는 유니크한 영화 개수
# 
# 2.ratings에 있는 유니크한 사용자 수
# 
# 3.가장 인기 있는 영화 30개(인기순)

# In[8]:


# ratings에 있는 유니크한 영화 개수
ratings['movie_id'].nunique()


# In[9]:


# rating에 있는 유니크한 사용자 수
ratings['user_id'].nunique()


# In[10]:


# 가장 인기 있는 영화 30개(인기순)


movie_count = ratings.groupby('movie_id')['user_id'].count()
movie_count.sort_values(ascending=False).head(30)


# # 3) 내가 선호하는 영화를 5가지 골라서 ratings에 추가해 줍시다.

# In[11]:



# 내가 선호하는 영화를 5가지 골라서 ratings에 추가
my_favorite = ['Platoon', 'Crying Game, The', 'Welcome to the Dollhouse' , 'E.T. the Extra-Terrestrial', 'James and the Giant Peach']
 
# 'hyeji'이라는 user_id가 위 영화를 5회씩 봤다고 가정하겠습니다.
my_list = pd.DataFrame({'user_id': ['hyeji']*5, 'title': my_favorite, 'count':[5]*5})

if not ratings.isin({'user_id':['hyeji']})['user_id'].any():   # user_id에 'hyeji'이라는 데이터가 없다면
    ratings = ratings.append(my_list)                         # 위에 임의로 만든 my_list 데이터를 추가해 줍니다. 

ratings.tail(10)       


# In[12]:


# indexing
userid_unique = ratings['user_id'].unique()
movie_unique = ratings['title'].unique()

# user, movie indexing 하는 코드 idx는 index의 약자입니다.
user_to_idx = {v:k for k,v in enumerate(userid_unique)}
movie_to_idx = {v:k for k,v in enumerate(movie_unique)}


# In[13]:


# 인덱싱이 잘 되었는지 확인
print(user_to_idx['hyeji'])    
print(movie_to_idx['Platoon'])


# In[14]:



# indexing을 통해 데이터 컬럼 내 값을 바꾸는 코드
# dictionary 자료형의 get 함수는 https://wikidocs.net/16 을 참고하세요.

# user_to_idx.get을 통해 user_id 컬럼의 모든 값을 인덱싱한 Series를 구해 봅시다. 
# 혹시 정상적으로 인덱싱되지 않은 row가 있다면 인덱스가 NaN이 될 테니 dropna()로 제거합니다. 
temp_user_data = ratings['user_id'].map(user_to_idx.get).dropna()
if len(temp_user_data) == len(ratings):   # 모든 row가 정상적으로 인덱싱되었다면
    print('user_id column indexing OK!!')
    ratings['user_id'] = temp_user_data   # data['user_id']을 인덱싱된 Series로 교체해 줍니다. 
else:
    print('user_id column indexing Fail!!')

# artist_to_idx을 통해 artist 컬럼도 동일한 방식으로 인덱싱해 줍니다. 
temp_movie_data = ratings['title'].map(movie_to_idx.get).dropna()
if len(temp_movie_data) == len(ratings):
    print('movie_id column indexing OK!!')
    ratings['movie_id'] = temp_movie_data
else:
    print('movie_id column indexing Fail!!')

ratings


# # 4) CSR matrix를 직접 만들어 봅시다.

# In[15]:


# csr_matrix
from scipy.sparse import csr_matrix

num_user = ratings['user_id'].nunique()
num_movie = ratings['movie_id'].nunique()

csr_data = csr_matrix((ratings['count'], (ratings.user_id, ratings.movie_id)), shape= (num_user, num_movie))
csr_data


# # 5) als_model = AlternatingLeastSquares 모델을 직접 구성하여 훈련시켜 봅시다.

# In[16]:


from implicit.als import AlternatingLeastSquares
import os
import numpy as np

# implicit 라이브러리에서 권장하고 있는 부분입니다. 학습 내용과는 무관합니다.
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['MKL_NUM_THREADS']='1'


# In[17]:


# Implicit AlternatingLeastSquares 모델의 선언
als_model = AlternatingLeastSquares(factors=100, regularization=0.01, use_gpu=False, iterations=15, dtype=np.float32)


# In[18]:


# als 모델은 input으로 (item X user 꼴의 matrix를 받기 때문에 Transpose해줍니다.)
csr_data_transpose = csr_data.T
csr_data_transpose


# In[ ]:


# 모델 훈련
als_model.fit(csr_data_transpose)


# # 6) 내가 선호하는 5가지 영화 중 하나와 그 외의 영화 하나를 골라 훈련된 모델이 예측한 나의 선호도를 파악해 보세요.

# In[43]:


hyeji, Platoon = user_to_idx['hyeji'], movie_to_idx['Platoon']
hyeji_vector, Platoon_vector = als_model.user_factors[hyeji], als_model.item_factors[Platoon]

print('슝=3')


# In[44]:


# 내적하는 코드
np.dot(hyeji_vector, Platoon_vector)


# In[46]:


#모델이 Crying Game, The에 대한 선호도를 어떻게 예측할지 한 번 보겠습니다
movie_i = movie_to_idx['Crying Game, The']
movie_i_vector = als_model.item_factors[movie_i]
np.dot(hyeji_vector, movie_i_vector)


# # 7) 내가 좋아하는 영화와 비슷한 영화를 추천받아 봅시다.

# In[54]:


#좋아하는 영화
favorite_movie = 'Platoon'
movie_id = movie_to_idx[favorite_movie]
similar_movie = als_model.similar_items(movie_id, N=15)
similar_movie


# In[55]:


#artist_to_idx 를 뒤집어, index로부터 artist 이름을 얻는 dict를 생성합니다. 
idx_to_movie = {v:k for k,v in movie_to_idx.items()}
[idx_to_movie[i[0]] for i in similar_movie]


# In[56]:


# 좋아하는 영화와 비슷한 영화 함수
def get_similar_movie(movie_name: str):
    movie_id = movie_to_idx[movie_name]
    similar_movie = als_model.similar_items(movie_id)
    similar_movie = [idx_to_movie[i[0]] for i in similar_movie]
    return similar_movie

print("슝=3")


# # 8) 내가 가장 좋아할 만한 영화들을 추천받아 봅시다.

# In[ ]:


user = user_to_idx['hyeji']
# recommend에서는 user*item CSR Matrix를 받습니다.
movie_recommended = als_model.recommend(user, csr_data, N=20, filter_already_liked_items=True)
movie_recommended


# In[ ]:


[idx_to_movie[i[0]] for i in movie_recommended]


# # 결과 
# 모델 훈련 시키는 과정에서 NaN encountered in factors 가 자꾸 나는데, 시간이 날 때 오류를 고쳐야겠다.

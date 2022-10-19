from keras.preprocessing.text import Tokenizer
from matplotlib.pyplot import text
import sklearn

test='나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'
token=Tokenizer()
token.fit_on_texts([test]) #리스트 형태로 받는다 (여러개 넣을 수 있다는 뜻) /fit에서 인텍스가 생성된거임


print(token.word_index) #토큰은 어절을 인덱스로 나누고 많이 나온 순서로 프린트
# {'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}

# 밥을(6) 이 나는(3)의 두배의 가치를 가지는건 아니지만 나는(3)의 3+3=6 이 되기 때문에 이럴때는 원핫인코딩을 해줘야 한다.

x=token.texts_to_sequences([test])
# print(x) 
#[[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]]

from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
# x=to_categorical(x)
# print(x)
# [[[0. 0. 0. 1. 0. 0. 0. 0. 0.]     
#   [0. 0. 0. 0. 1. 0. 0. 0. 0.]     
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]     
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]     
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]     
#   [0. 0. 0. 0. 0. 0. 1. 0. 0.]     
#   [0. 0. 0. 0. 0. 0. 0. 1. 0.]     
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]     
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]     
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]     
#   [0. 0. 0. 0. 0. 0. 0. 0. 1.]]]  
# 각 각의 어절들이 열로 
# ohe=OneHotEncoder
# x=ohe.fit_transform(x.reshape)
# print(x)

# print(x.shape) (1, 11, 9) 
#               ㄴ한문장, 11어절, 9열
import numpy as np
x_new=np.array(x)
# print(x_new.shape) #(1, 11)
# print(x_new)

ohe=OneHotEncoder()
x_new=ohe.fit_transform(x_new)
print(x_new)
print(x_new.shape) 














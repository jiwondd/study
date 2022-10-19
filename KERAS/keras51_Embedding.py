from keras.preprocessing.text import Tokenizer
import numpy as np

# 1. 데이터
docs=['너무 재밋어요','참 최고에요','참 잘 만든 영화예요',
      '추천하고 싶은 영화입니다.','한 번 더 보고 싶네요','글쎄요',
      '별로에요', '생각보다 지루해요','연기가 어색해요',
      '재미없어요','너무 재미없다','참 재밋네요','민수가 못 생기긴 했어요',
      '안결 혼해요']

# 긍정1, 부정 0
labels=np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0]) #(14, )

token=Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재밋어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화예요': 7, 
# '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, 
# '싶네요': 15, '글쎄요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, 
# '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밋네요': 24, '민수가': 25, '못': 26, 
# '생기긴': 27, '했어요': 28, '안결': 29, '혼해요': 30}

x=token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], 
# [20, 21], [22], [2, 23], [1, 24], [25, 26, 27, 28], [29, 30]]  <-쉐입이 다 다르다. 그럼 우짜노?
#  길이가 큰거에 맞춰서 나머지 0으로 채우기 / 만약 길이가 너무 길다면? 일정 자릿수에 맞춰서 긴거는 자르고 짧은거는 늘이고
#  ㄴ 보통은 앞부분에 채우는게 낫다.
from keras.preprocessing.sequence import pad_sequences
pad_x=pad_sequences(x, padding='pre',maxlen=5 ) 
#                      ㄴ0은 앞에 채울게0 ㄴ5글자까지 만들게
#                        뒤는 post
print(pad_x)
# [[ 0  0  0  2  3]
#  [ 0  0  0  1  4]
#  [ 0  1  5  6  7]
#  [ 0  0  8  9 10]
#  [11 12 13 14 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0  0 17]
#  [ 0  0  0 18 19]
#  [ 0  0  0 20 21]
#  [ 0  0  0  0 22]
#  [ 0  0  0  2 23]
#  [ 0  0  0  1 24]
#  [ 0 25 26 27 28]
#  [ 0  0  0 29 30]]
print(pad_x.shape) #(14, 5) -> 덴스 혹은 3차원으로 바꿔서 LSTM이나 

word_size=len(token.word_index)
print('word_size:',word_size) #단어 사전의 개수 : 30

print(np.unique(pad_x,return_counts=True)) 
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
# array([37,  3,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1], dtype=int64))


# 2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding

model=Sequential()  #인풋 쉐이프는 (14,5) <- 원 핫 안되어있는 순수한 수치의 쉐이프
#                단어사전의개수ㄱ 아웃풋노드의개수ㄱ  한 열의 길이ㄱ (모를때는안넣어도됨)
#  인풋딤(단어사전의개수)에 아무숫자나 넣어도 돌아가기는 하는데 그래도 안되요 , input len도 마찬가지!
# model.add(Embedding(input_dim=31, output_dim=10, input_length=5))
# model.add(Embedding(input_dim=31, output_dim=10))
# model.add(Embedding(31,10)) 
# model.add(Embedding(31,10,5)) 앞에 두개는 연산할때 쓰는건데 뒤에 5는 연산용이 아니기때문에 이렇게 넣으면 에러!
model.add(Embedding(31, 3, input_length=5)) 
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))
model.summary()

'''
Model: "sequential"         10개짜리 5묶음
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 10)             310
_________________________________________________________________
lstm (LSTM)                  (None, 32)                5504
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 5,847
Trainable params: 5,847
Non-trainable params: 0
_________________________________________________________________
PS C:\study>

'''

# 3. 컴파일 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(pad_x,labels,epochs=3,batch_size=16)

# 4. 평가, 예측
acc=model.evaluate(pad_x,labels)[1]
print('acc:',acc)

# acc=model.evaluate(pad_x,labels)
# print('acc:',acc[1])

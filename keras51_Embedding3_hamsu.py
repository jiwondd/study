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

# [실습하기] 아래 문장의 긍정과 부정을 예측해!!!
x_pred='나는 형권이가 정말 재미없다 정말 정말'
x_pred=[x_pred]

token=Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재밋어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화예요': 7, 
# '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, 
# '싶네요': 15, '글쎄요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, 
# '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밋네요': 24, '민수가': 25, '못': 26, 
# '생기긴': 27, '했어요': 28, '안결': 29, '혼해요': 30}

token=Tokenizer()
token.fit_on_texts(x_pred)
print(token.word_index)
# {'정말': 1, '나는': 2, '형권이가': 3, '재미없다': 4}  

x=token.texts_to_sequences(docs)
x_pred=token.texts_to_sequences(x_pred)

# print(x)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], 
# [20, 21], [22], [2, 23], [1, 24], [25, 26, 27, 28], [29, 30]]  <-쉐입이 다 다르다. 그럼 우짜노?
#  길이가 큰거에 맞춰서 나머지 0으로 채우기 / 만약 길이가 너무 길다면? 일정 자릿수에 맞춰서 긴거는 자르고 짧은거는 늘이고
#  ㄴ 보통은 앞부분에 채우는게 낫다.

from keras.preprocessing.sequence import pad_sequences
pad_x=pad_sequences(x, padding='pre',maxlen=5) 
#                      ㄴ0은 앞에 채울게0 ㄴ5글자까지 만들게
pad_pred=pad_sequences(x_pred, padding='pre',maxlen=5, truncating='post') 

print(pad_x)
print(pad_x.shape)

word_size=len(token.word_index)
print('word_size:',word_size) #단어 사전의 개수 : 30

print(np.unique(pad_x,return_counts=True)) 

# 2. 모델
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, LSTM, Embedding, Input

# model=Sequential() 
# model.add(Embedding(31, 3, input_length=5)) 
# model.add(LSTM(32))
# model.add(Dense(1,activation='sigmoid'))
# model.summary()

input=Input(shape=(5, ))
Emb=Embedding(31,3)(input)
LSTM1=LSTM(32)(Emb)
dense1=Dense(1, activation='sigmoid')(LSTM1)
model=Model(inputs=input,outputs=dense1)
model.summary()


# 3. 컴파일 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(pad_x,labels,epochs=100,batch_size=16)

# 4. 평가, 예측
acc=model.evaluate(pad_x,labels)[1]
x_pred=pad_sequences(x_pred, padding='pre',maxlen=5, truncating='post') 


x_pred=model.predict(x_pred)
print('acc:',acc)

if x_pred >=0.5:
    print('재밋다')
else :
    print('노잼')
    
# acc: 0.5714285969734192
# 노잼

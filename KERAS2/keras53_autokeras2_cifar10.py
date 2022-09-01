from tracemalloc import start
import autokeras as ak
import tensorflow as tf
import keras
import time

# 1. 데이터
(x_train,y_train),(x_test,y_test)=\
    keras.datasets.cifar10.load_data()
    
# 2. 모델
model=ak.ImageClassifier(
    overwrite=True,
    max_trials=2
)    

# 3. 컴파일 훈련
start=time.time()
model.fit(x_train,y_train,epochs=5)
end=time.time()

# 4. 평가, 예측
y_pred=model.predict(x_test)
results=model.evaluate(x_test,y_test)
print('걸린시간:',round(end-start,4))
print('result:',results)

# 걸린시간: 2706.6742
# result: [0.8762439489364624, 0.6973999738693237]
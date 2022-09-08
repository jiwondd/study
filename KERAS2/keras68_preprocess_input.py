from keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from keras.applications.regnet import preprocess_input, decode_predictions
import numpy as np

model=ResNet50(weights='imagenet')
img_path='d:/study_data/_data/dog/chiwawa.jpg'

img=image.load_img(img_path,target_size=(224,224))
# print(img) <PIL.Image.Image image mode=RGB size=224x224 at 0x1D6C320C8E0>

# 이미지를 수치화
x=image.img_to_array(img)
# print('=======image.img_to_array(img)=======')
# print(x,'\n',x.shape)  (224, 224, 3)
x=np.expand_dims(x,axis=0) #  <- 얘랑 똑같  x=x.reshape(1,224,224,3)
# print('=======np.expand_dims(x,axis=0)=======')
# print(x,'\n',x.shape) # (1, 224, 224, 3)

x=preprocess_input(x)
# print('======preprocess_input(x)=======')
# print(x,'\n',x.shape) # (1, 224, 224, 3)

pred=model.predict(x)
print(pred, '\n',pred.shape)

print('결과:',decode_predictions(pred,top=5)[0])

# 결과: [('n02086910', 'papillon', 0.36338863), ('n02085620', 'Chihuahua', 0.33511364), 
#        ('n02112018', 'Pomeranian', 0.14023957), ('n02104365', 'schipperke', 0.029506162), 
#        ('n02087046', 'toy_terrier', 0.026006388)]
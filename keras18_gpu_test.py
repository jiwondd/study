import numpy as np
import tensorflow as tf

print(tf.__version__)

gpus=tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus):
    print('지피유돈다룰루')
else:
    print('지피유안도라유유')
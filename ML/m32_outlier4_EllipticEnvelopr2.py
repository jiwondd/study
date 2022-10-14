import numpy as np
from sklearn.covariance import EllipticEnvelope

aaa=np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
             [100,200,-30,400,500,600,-7000,800,900,1000,210,420,350]])

# print(aaa.shape) #(2, 13)
aaa=np.transpose(aaa)
# print(aaa.shape) #(13, 2)

abc=aaa[:,0]
abc2=aaa[:,1]
abc=abc.reshape(-1,1)
abc2=abc2.reshape(-1,1)
# print(abc.shape) #(13, 1)
# print(abc2.shape) #(13, 1)

outliers=EllipticEnvelope(contamination=.1) 

outliers.fit(abc)
results1=outliers.predict(abc)
print(results1)

outliers.fit(abc2)
results2=outliers.predict(abc2)
print(results2)

# [-1  1  1  1  1  1  1  1  1  1  1  1 -1]
# [ 1  1  1  1  1  1 -1  1  1 -1  1  1  1]
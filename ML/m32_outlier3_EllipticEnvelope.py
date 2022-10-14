import numpy as np
from sklearn.covariance import EllipticEnvelope
outliers=EllipticEnvelope(contamination=.1) #10프로를 이상치로 잡아라 / .2 =20프로
aaa=np.array([-10,2,3,4,5,6,700,8,9,10,11,12,50])
print(aaa.shape) #(13,)
aaa=aaa.reshape(-1,1)
print(aaa.shape) #(13, 1)

outliers.fit(aaa)
results=outliers.predict(aaa)
print(results)
# [ 1  1  1  1  1  1 -1  1  1  1  1  1 -1]


# def outliers(data_out):
#     quartile_1, q2, quartile_3=np.percentile(data_out,
#                                        [25,50,75])
#     print("1사분위:",quartile_1)
#     print("q2:",q2)
#     print("3사분위:",quartile_3)
#     iqr=quartile_3-quartile_1
#     print("ipr : " ,iqr) 
#     lower_bound=quartile_1-(iqr*1.5) 
#     upper_bound=quartile_3+(iqr*1.5) 
#     return np.where((data_out>upper_bound)|
#                     (data_out<lower_bound))

# outliers_loc=outliers(aaa)
# print("이상치의 위치 : ",outliers_loc)
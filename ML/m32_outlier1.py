
import numpy as np
aaa=np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50])

def outliers(data_out):
    quartile_1, q2, quartile_3=np.percentile(data_out,
                                       [25,50,75])
    print("1사분위:",quartile_1)
    print("q2:",q2)
    print("3사분위:",quartile_3)
    iqr=quartile_3-quartile_1
    print("ipr : " ,iqr) 
    lower_bound=quartile_1-(iqr*1.5) 
    upper_bound=quartile_3+(iqr*1.5) 
    return np.where((data_out>upper_bound)|
                    (data_out<lower_bound))

outliers_loc=outliers(aaa)
print("이상치의 위치 : ",outliers_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()

# aaa=np.array([1,2,-20,4,5,6,7,8,20,10])
# 1사분위: 2.5
# q2: 5.5
# 3사분위: 7.75
# ipr :  5.25
# 이상치의 위치 :  (array([2, 8], dtype=int64),)

# aaa=np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50])
# 1사분위: 4.0
# q2: 7.0
# 3사분위: 10.0
# ipr :  6.0
# 이상치의 위치 :  (array([ 0, 12], dtype=int64),)
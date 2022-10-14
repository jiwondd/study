
import numpy as np
aaa=np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
             [100,200,-30,400,500,600,-7000,800,900,1000,210,420,350]])
aaa=np.transpose(aaa)

abc=aaa[:,0]
abc2=aaa[:,1]

# print(aaa.shape) (13, 2)
# print(aaa)
# [[  -10   100] 
#  [    2   200] 
#  [    3   -30] 
#  [    4   400] 
#  [    5   500] 
#  [    6   600] 
#  [    7 -7000] 
#  [    8   800] 
#  [    9   900] 
#  [   10  1000] 
#  [   11   210] 
#  [   12   420] 
#  [   50   350]]

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

outliers_loc1=outliers(abc)
print("이상치의 위치 : ",outliers_loc1)

outliers_loc2=outliers(abc2)
print("이상치의 위치 : ",outliers_loc2)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()


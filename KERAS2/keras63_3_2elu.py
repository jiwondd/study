import numpy as np
import matplotlib.pyplot as plt


a=0.1

def elu(x):
    return (x>=0)*x+(x<0)*a*(np.exp(x)-1)

elu2=lambda x: (x>=0)*x+(x<0)*a*(np.exp(x)-1)

x=np.arange(-5,5,0.1)
y=elu2(x)

plt.plot(x,y)
plt.grid()
plt.show()
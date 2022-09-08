import numpy as np
import matplotlib.pyplot as plt

alpha=1.67326324
scale=1.05070098

def selu(x) :
    return scale * (np.maximum(0,x)+np.minimum(0,alpha*(np.exp(x)-1)))
selu2=lambda x: scale * (np.maximum(0,x)+np.minimum(0,alpha*(np.exp(x)-1)))

x=np.arange(-5,5,0.1)
y=selu2(x)

plt.plot(x,y)
plt.grid()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

def Leaky_relu(x):
    return np.np.maximum(0.1*x,x)
Leaky_relu2=lambda x:np.maximum(0.1*x,x)

x=np.arange(-5,5,0.1)
y=Leaky_relu2(x)

plt.plot(x,y)
plt.grid()
plt.show()
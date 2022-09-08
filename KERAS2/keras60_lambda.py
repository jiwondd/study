from numpy import gradient


gradient1=lambda x :2*x-4

def gradient2(x):
    temp=2*x-4
    return temp

x=3

print(gradient1(x))
print(gradient2(x))

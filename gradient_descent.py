import matplotlib.pyplot as plt
import numpy as np
x=soap=[4, 4.5, 5, 5.5, 6, 6.5, 7]
y=Y=sud=[33, 42, 45, 51, 53, 61, 62]

cou=0
b0=0
b1=0
eta=0.001
b0_new=0
b1_new=0
while True:
    s1=0
    s2=0
    for i in range(7):
        s1+=(y[i]-(b0+b1*x[i]))
        s2+=(x[i]*(y[i]-(b0+b1*x[i])))
    b0_new=b0+eta*s1
    b1_new=b1+eta*s2
    if(b0==b0_new and b1==b1_new):
        print(cou)
        break
    b0=b0_new
    b1=b1_new
    print(f"{b0_new}\t{b1_new}")
    cou+=1
print('slope ',b1_new)
print('intercept',b0_new)

x = np.linspace(0,len(soap))
y = b0_new+b1_new*x
plt.plot(x, y, '-0')

Xt=[[1, 1, 1, 1, 1, 1, 1],soap]
print(Xt)

print("\nX Transpose: ")
X=np.transpose(Xt)
print(X)


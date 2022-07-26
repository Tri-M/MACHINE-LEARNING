import numpy as np   
import matplotlib.pyplot as plt

soaps=[4, 4.5, 5, 5.5, 6, 6.5, 7]
Y=sud=[33, 42, 45, 51, 53, 61, 62]

p=plt.scatter(soaps,sud)
Xt=[[1, 1, 1, 1, 1, 1, 1],soaps]
print(Xt)

print("\nX^T: ")
X=np.transpose(Xt)
print(X)


print("\nX^T X: ")
XtX=np.dot(Xt,X)
print(XtX)


print("\nX^T X^-1: ")
XtXi=np.linalg.inv(XtX)
print(XtXi)


print("\nX^TY")
XtY=np.dot(Xt,Y)
print(XtY)


print("\nB0 B1")
B=np.dot(XtXi,XtY)
print(B)

print("\nLinear equation : ")
if (B[0]<0):
    print("y="+str(B[1])+"x"+str(B[0]))
else:
    print("y="+str(B[1])+"x+"+str(B[0]))


x = np.linspace(0,7)
y = B[1]*x+B[0]
plt.plot(x, y, '-0', label='y=2x+1')

SSE=0
SE=0
for i in range(len(sud)):
    SE+=sud[i]-(B[1]*soaps[i]+B[0])
    SSE+=(sud[i]-(B[1]*soaps[i]+B[0]))**2 #Quantify the error
print("Errors sum: "+str(SE)+"\nSSE: "+str(SSE))

Y_ = np.mean(Y)
print("\nMean (Y_): "+str(Y_))

S=0
for i in sud:
    S+=i-Y_
print("Sum of Yi-Y_: "+str(S))

SST=0
for i in sud:
    SST+=(i-Y_)**2
print("SST: "+str(SST))

SSR=SST-SSE
print("SSR: "+str(SSR))
import matplotlib.pyplot as plt
import numpy as np
x=soap=[4, 4.5, 5, 5.5, 6, 6.5, 7]
y=Y=sud=[33, 42, 45, 51, 53, 61, 62]

# p=plt.scatter(soap,sud)
ax = plt.axes(projection='3d')

ct=0
b0=0
b1=0
s1=0
s2=0
b0_new=0
b1_new=0
eta=0.001
while True:
    s1=0
    s2=0
   
    for i in range(7):
        s1+=(y[i]-(b0+b1*x[i]))
        s2+=(x[i]*(y[i]-(b0+b1*x[i])))
    b0_new=b0+eta*s1
    b1_new=b1+eta*s2
    if(b0==b0_new and b1==b1_new):
        print(ct)
        break
    b0=b0_new
    b1=b1_new
   
   
   
    print(f"{b0_new}\t{b1_new}")
    ct+=1

print('slope ',b1_new)
print('intercept',b0_new)


x = np.linspace(0,len(soap))
y = b0_new+b1_new*x
# plt.plot(x, y, '-0')

Xt=[[1, 1, 1, 1, 1, 1, 1],soap]
print(Xt)

print("\nX Transpose: ")
X=np.transpose(Xt)
print(X)


print("\nX Transpose X: ")
XtX=np.dot(Xt,X)
print(XtX)


print("\nX Transpose X Inverse: ")
XtXi=np.linalg.inv(XtX)
print(XtXi)


print("\nX Transpose Y")
XtY=np.dot(Xt,Y)
print(XtY)


print("\nB matrix b0 b1")
B=np.dot(XtXi,XtY)
print(B)

print("\nLinear equation of the model: ")
if (B[0]<0):
    print("y="+str(B[1])+"x"+str(B[0]))
else:
    print("y="+str(B[1])+"x+"+str(B[0]))

SSE=0
SE=0
SElist=[]
for i in range(len(sud)):
    SE+=sud[i]-(B[1]*soap[i]+B[0])
    SSE+=(sud[i]-(B[1]*soap[i]+B[0]))**2 
    SElist.append(SE**2)
print("Sum of errors: "+str(SE)+"\nSSE: "+str(SSE))

Ybar = np.mean(Y)
print("\nMean (Ybar): "+str(Ybar))

S=0
for i in sud:
    S+=i-Ybar
print("Sum of Yi-Ybar: "+str(S))

SST=0
for i in sud:
    SST+=(i-Ybar)**2
print("SST: "+str(SST))

SSR=SST-SSE
print("SSR: "+str(SSR))
print("Print squared errors :",SElist)
ax.plot3D(soap,sud,SElist, 'red')
# ax.scatter3D(soap,sud,SElist,cmap='Greens');

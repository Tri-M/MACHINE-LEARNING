import numpy as np

def cost(b,m,data_points):
    r_sq=0
    
    for i in range(len(data_points)):
        x=data_points[i,0]
        y=data_points[i,1]
        
        r_sq+=(y-(m*x+b))**2
    return r_sq/(2*float(len(data_points)))

def stepGradient(b,m,data_points,eta):
    m_grad=0
    b_grad=0
    N=float(len(data_points))
    
    for i in range(len(data_points)):
        x=data_points[i,0]
        y=data_points[i,1]
        b_grad+=-(1/N)*(y-((m*x+b)))
        m_grad+=-(1/N)*(x*(y-((m*x)+b)))
        b_new=b-(eta*b_grad)
        m_new=m-(eta*m_grad)
    return [b_new,m_new]

def batchGD(data_points,b_initial,m_initial,eta,iterations):
    b=b_initial
    m=m_initial
    for i in range(iterations):
        b,m=stepGradient(b, m, np.array(data_points), eta)
    return [b,m]

def y_predicted(x):
    b,m=main()
    y_pred=m*x+b
    print(f"\nThe regression predictor of x ={x} is y={y_pred}")

def main():
    data_points=np.genfromtxt("soap.csv",delimiter=",")
    eta=0.1
    m_initial=0
    b_initial=0
    iterations=100
    initial_cost=cost(b_initial,m_initial,data_points)
    print(f"\nStaring gradient descent at y 'intercept' = {b_initial},'slope'={m_initial} , 'starting cost' ={initial_cost}")
    print("______________________________")
    [b,m]=batchGD(data_points, b_initial, m_initial, eta, iterations)
    end_cost=cost(b,m,data_points)
    print(f"\ngradient descent for {iterations} iterations we got  y 'intercept' = {b},'slope'={m} ,'ending cost' ={end_cost}")
    return b,m

if __name__ == "__main__":
    y_predicted(7)
        
    

        
        
  

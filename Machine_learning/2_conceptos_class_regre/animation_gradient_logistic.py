#Milton Orlando Sarria
#USC - Cali
import numpy as np
import matplotlib.pyplot as plt

###
def createData(N=[500,500]):
    #generate artificial data points "toy data"
    mvalues=np.array([[0.7,3],[3,0.3]])
    std = [0.8,0.6]
    X=np.array([])
    Y=np.array([])

    for ii in range(len(N)):
        x= std[ii]*np.random.randn(N[ii],2)+mvalues[ii]
        if ii==0:
            X= x
        else:
            X=np.vstack((X,x))
        y=np.ones(N[ii])*ii; 
        Y= np.append(Y,y)
    
    Y=Y.astype(int)    

    return X,Y
######################################
def func_h_x(theta,x):
    """Compute h_theta(x)"""
    z=np.dot(x,theta)
    score=1/(1+np.exp(-z))
    
    return score      
    
#######################################################################
### crear datos
X,Y=createData()
labels=np.unique(Y)

#####  determinar una fncion inicial 
x = np.arange(-5.0, 8.0, 3)
x = np.vstack([x, np.ones_like(x)])
x=x.T
#parametros
theta=np.array([-1.0,-1.0,-1.0])

alpha=1e-4
Y0=(-theta[0]*x[:,0]-theta[2])/theta[1]

#configurar grafica
plt.show() 
axes = plt.gca() 

axes.set_ylim(-4, +6) 

#graficar datos usando un color diferente para cada grupo
color='rb'
for ii in labels:
        key = 'o'+color[ii]  
        lin,=axes.plot(X[Y==ii,0],X[Y==ii,1],key)
        

line1, = axes.plot(x[:,0],Y0,'-k')
########################################
input('Presione ENTER para iniciar')
#iniciar el proceso de captura de datos
print('Para finalizar la cierre la ventana')
continuar = True

X = np.vstack([X.T, np.ones_like(Y)]).transpose()

plt.draw()
plt.pause(1) 
#maxiter=1000
#for it in range(maxiter):
while continuar:
    
    hx =func_h_x(theta,X)
    
    theta[2] = theta[2] + alpha * np.sum((Y-hx)*X[:,2])
    theta[1] = theta[1] + alpha * np.sum((Y-hx)*X[:,1])
    theta[0] = theta[0] + alpha * np.sum((Y-hx)*X[:,0])
    print(theta)
    y_hat=(-theta[0]*x[:,0]-theta[2])/theta[1]

    line1.set_xdata(x[:,0]) 
    line1.set_ydata(y_hat) 
    
    plt.draw() 
    plt.pause(1) 
    #determinar si la ventana continua abierta
    existe_plot = plt.get_fignums()
    if len(existe_plot)==0:
        continuar = False
   
        
    


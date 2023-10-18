#Milton Orlando Sarria
#USC - Cali
import numpy as np
import matplotlib.pyplot as plt

### cargar datos
file_name='datosML/linear.data'
X=np.loadtxt(file_name,delimiter=',') 
x=X[:,0]
y=X[:,1]

#####  determinar una pendiente inicial de forma aleatoria
theta1= np.random.rand()*20
theta0=-np.random.rand()*10
Y0=theta1*x+theta0

alpha=1e-5 #tasa de aprendizaje

#configurar grafica
plt.show() 
axes = plt.gca() 

axes.set_ylim(0, +100) 
line0, = axes.plot(x, y, 'or') 
line1, = axes.plot(x, Y0, 'k-') 
########################################
input('Presione ENTER para iniciar')
#iniciar el proceso de captura de datos
print('Para finalizar la cierre la ventana')
continuar = True

#maxiter=1000
#for it in range(maxiter):
while continuar:
    g0=0
    g1=0
    for i in range(x.size):
        hx=theta1*x[i]+theta0
        g1=g1+(hx-y[i])*x[i]
        g0=g0+(hx-y[i])*1
        
    theta1=theta1-alpha*g1
    theta0=theta0-alpha*g0
    Y_hat=theta1*x+theta0
    
    line1.set_xdata(x) 
    line1.set_ydata(Y_hat) 
    
    plt.draw() 
    plt.pause(10e-1) 
    #determinar si la ventana continua abierta
    existe_plot = plt.get_fignums()
    if len(existe_plot)==0:
        continuar = False
     
        
    


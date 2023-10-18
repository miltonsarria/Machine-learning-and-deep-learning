#Milton Orlando Sarria
#USC - Cali
#visualizar datos provenientes de arduino de forma grafica
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

import time

def update(i):
    x0=1;
    alpha=0.1;
    maxiter=10;
    x=np.linspace(0,8,100)
    y=(x-3)*(x-5)
    x_ =[]
    y_ =[] 
    x_.append(x0)
    y_.append((x0-3)*(x0-5)) 
    j=1
    while j<i:
        df = 2*x0-8
        x0 = x0-alpha*df
        x_.append(x0)
        y_.append((x0-3)*(x0-5)) 
        j=j+1
    ax.clear()
    ax.plot(x,y,lw=4) 
    ax.plot(x_,y_,'ro',markersize=10) 


#inicializar la grafica
fig = plt.figure()
ax  = fig.add_subplot(1,1,1)
ax.set_autoscaley_on(True)
ax.set_ylim(-3, 10)

continuar = input('press ENTER')

ani = FuncAnimation(fig, update, interval=2000) #actualizar la grafica cada 500 ms
plt.show()

    
    

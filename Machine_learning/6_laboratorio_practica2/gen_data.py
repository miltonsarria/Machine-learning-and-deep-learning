#create artificial data in 2-D
import numpy as np

def createData2D(N=[500,500,500]):
    #generar puntos de forma artificial para 3 clases
    #definir valor medio y desviacion
    #np.random.seed(1234)
    m = np.array([[0.7,5],[2,0.5],[4,4]])
    s = np.array([[0.5,2],[1,0.5],[1,1]])
    X=np.array([])
    Y=np.array([])

    for ii in range(len(N)):
        x1 = np.random.normal(m[ii,0], s[ii,0], N[ii])
        x2 = np.random.normal(m[ii,1], s[ii,1], N[ii])
        x = np.array([x1, x2]).T 
   
        if ii==0:
          X= x
        else:
          X=np.vstack((X,x))
        y=np.ones(N[ii])*ii; 
        Y= np.append(Y,y)
    
    Y=Y.astype(int)    

    return X,Y
##################################################################    
def createData3D(N=[500,500,500]):
    #generar puntos de forma artificial para 3 clases
    #definir valor medio y desviacion
    np.random.seed(1234)
    m = np.array([[0.7,3,3],[2,2,0.5],[4,4,0]])
    s = np.array([[0.5,2,1],[1,0.3,1],[1,1,0.5]])
    X=np.array([])
    Y=np.array([])

    for ii in range(len(N)):
        x1 = np.random.normal(m[ii,0], s[ii,0], N[ii])
        x2 = np.random.normal(m[ii,1], s[ii,1], N[ii])
        x3 = np.random.normal(m[ii,2], s[ii,2], N[ii])
        x = np.array([x1, x2, x3]).T 
        if ii==0:
          X= x
        else:
          X=np.vstack((X,x))
        y=np.ones(N[ii])*ii; 
        Y= np.append(Y,y)
    
    Y=Y.astype(int)    

    return X,Y

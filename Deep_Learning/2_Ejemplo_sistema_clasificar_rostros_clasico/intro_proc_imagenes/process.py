import numpy as np 
#####
#Milton Orlando Sarria Paja
#USC
#####
############################
### clas to do whitening
class whiten():
    def __init__(self):
        self.Mu     = np.array([])
        self.sigma  = np.array([])
        self.eigVals= np.array([])
        self.eigVecs= np.array([])
    ##### compute parameters
    def fit(self,X):
        self.get_sigma(X)
        
        eigVals, eigVecs = np.linalg.eig(self.sigma)
        self.eigVecs     = np.real(eigVecs)
        self.eigVals     = np.real(eigVals)
        return
    def get_center(self,X):
        if (self.Mu.size==0): 
            self.Mu =np.mean(X, axis = 0)
        X = X - self.Mu
        return X
    
    def get_sigma(self,X):
        X = self.get_center(X)
        self.sigma = np.cov(X, rowvar=False, bias=True)
        return
    #apply transform
    def transform(self,X):
        X = self.get_center(X)
        # Aplicar los vectores propios a los datos (rotar)
        X = X.dot(self.eigVecs)
        # Re-escalar los datos
        X = X / np.sqrt(self.eigVals + 1e-5)
        return  X
####################################################
####################################################
## class to perform  zca
class zca():
    def __init__(self):
        self.Mu     = np.array([])
        self.sigma  = np.array([])
        self.U      = np.array([])
        self.S      = np.array([])
        self.V      = np.array([])
    ##### compute parameters
    def fit(self,X):
        print("[INFO] training..")
        self.get_sigma(X)
        self.U,self.S,self.V = np.linalg.svd(self.sigma)
        print("[INFO] done!")
        return
    def get_center(self,X):
        if (self.Mu.size==0): 
            self.Mu =np.mean(X, axis = 0)
        X = X - self.Mu
        return X
    
    def get_sigma(self,X):
        X = self.get_center(X)
        self.sigma = np.cov(X, rowvar=False, bias=True)
        return
    #apply transform
    def transform(self,X, epsilon = 0.1):
        
        X = self.get_center(X)
        # Aplicar la transformacion
        X = self.U.dot(np.diag(1.0/np.sqrt(self.S + epsilon))).dot(self.U.T).dot(X.T).T
        return X
  

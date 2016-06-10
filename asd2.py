import matplotlib.pyplot as plt
import numpy as np
import math

####Inicialización####
dt = .01
h = 50
######################

class nodos:
    def __init__(self, tipo):
        self.tipo = tipo
        self.X = np.array([0,0,0])
        self.F = np.array([0,0,0])
        self.m = 0.1
        self.DOF = np.array([1,1,1])
        self.a = np.array([[0,0,0]])
        self.aold = np.array([0,0,0])
        self.v = np.array([[0,0,0]])
        self.Posicion = np.array([])
        
    def hidrostatica(self):#Resultados de hidrostatica en toneladas
        if self.X[2]>h:
            return 0
        elif (h-self.X[2])>0 and (h-self.X[2])<=1.2:#Boya es de 1,2[m] de diámetro
            x = h-self.X[2]
            funcion = (2.002797406892284*x**10)-(12.36393599196526*x**9)+(32.99328336198505*x**8)-(49.72756897200634*x**7)+(46.35953154660547*x**6)-(27.49175164909684*x**5)+(10.26214713244815*x**4)-(3.366114742055231*x**3)+(2.200152254189497*x**2)-(0.01442824074082993*x)
            return funcion
        else:
            return 0.922
            
class Elementos:
    def __init__(self,nodo1,nodo2,longitud):
        self.nodo1=int(nodo1)
        self.nodo2=int(nodo2)
        self.longitud=longitud
        self.E = ListaCaracteristicaElementos[i].Datos[0]
        Area = ListaCaracteristicaElementos[i].Datos[1]**2*np.pi*0.25*1000**-2
        self.A0 = Area
        self.F = 0.0
        self.Fx = 0.0
        self.Fy = 0.0
        self.Fz = 0.0
        self.R = Nodos[self.nodo2].X-Nodos[self.nodo1].X
        self.L= math.sqrt(self.R[0]**2+self.R[1]**2+self.R[2]**2)
        self.k = self.E*self.A0/self.L
        self.HistorialF = np.array([0.])
              
    def Phi(self):
        self.R=Nodos[self.nodo2].X-Nodos[self.nodo1].X
        if self.R[0]<0:
            return math.atan(self.R[1]/self.R[0]) + math.pi
        elif self.R[0]==0:
            return 0
        else:
            return math.atan(self.R[1]/self.R[0])
        
    def Theta(self):
        self.R=Nodos[self.nodo2].X-Nodos[self.nodo1].X
        return math.asin(self.R[2]/np.linalg.norm(self.R))
    
    def Fuerza(self):
        
        ##############################################
        #CALCULAR LONGITUD DESDE CONDICIÓN INICIAL!!!
        #############################################
        self.R=Nodos[self.nodo2].X-Nodos[self.nodo1].X
        
        if np.linalg.norm(self.R)>self.L:
            self.F=(np.linalg.norm(self.R)-self.L)*self.k # F = k(X1-X0), con X0 la longitud inicial del resorte.
            return self.F
        else:
            self.F = 0.0
            return self.F
        self.HistorialF = np.concatenate((self.HistorialF, [self.F]), axis = 0)

    def Plotear(self):
        plt.plot([Nodos[self.nodo1].X[0],Nodos[self.nodo2].X[0]],[Nodos[self.nodo1].X[1],Nodos[self.nodo2].X[1]])
        plt.plot([Nodos[self.nodo1].X[0],Nodos[self.nodo2].X[0]],[Nodos[self.nodo1].X[1],Nodos[self.nodo2].X[1]], 'ro')
        
    def Plotear3D(self):
        ax.scatter([Nodos[self.nodo1].X[0],Nodos[self.nodo2].X[0]],[Nodos[self.nodo1].X[1],Nodos[self.nodo2].X[1]],[Nodos[self.nodo1].X[2],Nodos[self.nodo2].X[2]])
        ax.plot([Nodos[self.nodo1].X[0],Nodos[self.nodo2].X[0]],[Nodos[self.nodo1].X[1],Nodos[self.nodo2].X[1]],[Nodos[self.nodo1].X[2],Nodos[self.nodo2].X[2]])
     
def Plotear3D():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    
    for i in np.arange(0,len(ArchivoElementos), 1):
        ax.plot([Nodos[ListaElementos[i].nodo1].X[0],Nodos[ListaElementos[i].nodo2].X[0]],[Nodos[ListaElementos[i].nodo1].X[1],Nodos[ListaElementos[i].nodo2].X[1]],[Nodos[ListaElementos[i].nodo1].X[2],Nodos[ListaElementos[i].nodo2].X[2]])
        ax.scatter([Nodos[ListaElementos[i].nodo1].X[0],Nodos[ListaElementos[i].nodo2].X[0]],[Nodos[ListaElementos[i].nodo1].X[1],Nodos[ListaElementos[i].nodo2].X[1]],[Nodos[ListaElementos[i].nodo1].X[2],Nodos[ListaElementos[i].nodo2].X[2]],color='blue')
    for i in np.arange(0,len(ArchivoNodos), 1):
        if np.array_equal(Nodos[i].DOF,np.array([0,2,1])):
            ax.scatter(Nodos[i].X[0], Nodos[i].X[1], Nodos[i].X[2], color = 'yellow')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim3d(0,100)
    x_surf=np.arange(-150, 200, 50)                # generate a mesh
    y_surf=np.arange(-150, 200, 50)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    plt.show()

#~CREACIÓN DE NODOS         
Nodos=[] #"Array" de nodos, definidos como objetos en clase nodos.
ArchivoNodos=np.loadtxt("nodos.txt") #Almacenamiento en memoria de nodos de archivo
for i in np.arange(0,len(ArchivoNodos),1):
    Nodos.append(nodos(1))
    Nodos[i].X=np.copy(ArchivoNodos[i,:])
    Nodos[i].Posicion = np.copy([Nodos[i].X])
    print Nodos[i].X

###################################
#PREPARACIÓN DE PLOTEO EN 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
###################################


#~CREACIÓN DE ELEMENTOS  
ListaElementos=[] #"Lista de elementos" como objetos de tipo Elementos.
ArchivoElementos=np.loadtxt("elementos.txt")
for i in np.arange(0,len(ArchivoElementos),1):
    ListaElementos.append(Elementos(ArchivoElementos[i,0],ArchivoElementos[i,1],ArchivoElementos[i,2]))
    ListaElementos[i].Plotear3D()
    
#################################
###CARACTERISTICA DE ELEMENTOS###
#################################  
ListaCaracteristicaElementos=[] 
ArchivoCaracteristicaElementos=np.loadtxt("caracteristica_elementos.txt")
for i in np.arange(0,len(ArchivoCaracteristicaElementos),1):
    if len(ArchivoElementos) != len(ArchivoCaracteristicaElementos):
        print "Caracteristica de Elementos definidos en forma inconsistente"
        break
for i in np.arange(0,len(ArchivoCaracteristicaElementos),1):
    ListaCaracteristicaElementos.append(Elementos(ArchivoCaracteristicaElementos[i,0],ArchivoCaracteristicaElementos[i,1], ArchivoCaracteristicaElementos[i,2]))
    ListaCaracteristicaElementos[i].Datos = np.copy(ArchivoCaracteristicaElementos[i,:])
    print ListaCaracteristicaElementos[i].Datos
#OTRA VEZ LA PREGUNTA... QUE PASA CON LOS QUE NO SON LINEAS DE TENSION
    
# GRAFICAR CONDICIÓN INICIAL; ELEMENTOS EN 3D
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

ArchivoCB=np.loadtxt("CB.txt")
for i in np.arange(0,len(ArchivoCB),1):
    if len(ArchivoCB) != len(ArchivoNodos):
        print "Condiciones de borde definidas en forma inconsistente"
        break
    Nodos[i].DOF = np.copy(ArchivoCB[i,:])


#~DEFINICIÓN DE CARGAS EXTERNAS
Cargas=np.loadtxt("cargas.txt")

def Solver(maxiter=500):
    
    ######################
    ########SOLVER########
    ######################
    contador = 0
    MaxIteraciones = maxiter
    hidrostaticavieja=np.zeros(len(ArchivoNodos))
    acel=np.array([.0,.0,.0])
    ######################
    
    while contador < MaxIteraciones:
        contador=contador+1
        FuerzaElemento = np.array([0.0,0.0,0.0])
        for i in np.arange(0,len(ArchivoNodos),1):
            Nodos[i].F=[0,0,0]
            Nodos[i].F=np.copy(Nodos[i].F)+Cargas[i,:]*(1/(1+math.exp(-0.1*(float(contador)-50)))) #Aplicación de cargas externas sobre función 'S' (sigmoide) que converge aproximadamente en 100 ciclos
            Nodos[i].F=np.copy(Nodos[i].F)+np.array([0,0,Nodos[i].m*(-9.81)]) #Aplicación de peso
            
            if np.array_equal(Nodos[i].DOF,np.array([0,2,1])): #Aplicación de empuje de boyas
                Nodos[i].F=np.copy(Nodos[i].F)+[0,0,0.5*Nodos[i].hidrostatica()]+[0,0,0.5*hidrostaticavieja[i]]
                hidrostaticavieja[i] = Nodos[i].hidrostatica()
                
        for i in np.arange(0,len(ArchivoElementos),1): #Aplicación de cargas por tensión de elementos
            
            Fx = ListaElementos[i].Fuerza()*math.cos(ListaElementos[i].Phi())*math.cos(ListaElementos[i].Theta())
            Fy = ListaElementos[i].Fuerza()*math.sin(ListaElementos[i].Phi())*math.cos(ListaElementos[i].Theta())
            Fz = ListaElementos[i].Fuerza()*math.sin(ListaElementos[i].Theta())
            ListaElementos[i].HistorialF = np.concatenate((ListaElementos[i].HistorialF,[ListaElementos[i].Fuerza()]), axis = 0)
            FuerzaElemento = np.array([Fx,Fy,Fz]) 
            
            Nodos[ListaElementos[i].nodo1].F=np.copy(Nodos[ListaElementos[i].nodo1].F)+FuerzaElemento
            
            Nodos[ListaElementos[i].nodo2].F=np.copy(Nodos[ListaElementos[i].nodo2].F)-FuerzaElemento
            
        for i in np.arange(0,len(ArchivoNodos),1): #Iterar para encontrar nueva posición 
            acel = np.array([np.copy(Nodos[i].F)])
            Nodos[i].a = np.concatenate((Nodos[i].a, acel), axis = 0)
            vel = [sum(Nodos[i].a)*dt]
            Nodos[i].v = np.concatenate ((Nodos[i].v, vel), axis = 0)
    
            if np.array_equal(Nodos[i].DOF,[0,0,1])==False:
                Nodos[i].X=np.copy(Nodos[i].X)+(Nodos[i].a[-1]+Nodos[i].a[-2])*0.5*dt**2
            Nodos[i].Posicion = np.concatenate((Nodos[i].Posicion, [Nodos[i].X]), axis = 0)
    plt.show()
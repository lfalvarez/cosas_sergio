import numpy as np

h = 50
boya = np.array([0., 0., 48])

def hidrostatica():
    if boya[2]>h:
        return 0
        print hidrostatica
    elif boya[2]<=h and boya[2]>(h-2):
        return (h-boya[2])*500#-.5*((self.Posicion[-1,2]-self.Posicion[-2,2])/dt)**2
        print hidrostatica
    else:
        return 1000
        print hidrostatica
   
#Recordar cambiar print por return
#tener en cuenta sistema de unidades
#Desplazamientos estan en toneladas
#cambiar boya[2] por self.X[2]
def hidrostatica2():
    if boya[2]>h:
        print 0
    elif (h-boya[2])>0 and (h-boya[2])<=1.2:
        x = h-boya[2]
        funcion = (2.002797406892284*x**10)-(12.36393599196526*x**9)+(32.99328336198505*x**8)-(49.72756897200634*x**7)+(46.35953154660547*x**6)-(27.49175164909684*x**5)+(10.26214713244815*x**4)-(3.366114742055231*x**3)+(2.200152254189497*x**2)-(0.01442824074082993*x)
        print funcion
    else:
        return 0.922
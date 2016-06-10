import scipy
import matplotlib.pyplot as plt
import numpy as np


x = scipy.array([0, 0.12, 0.24, 0.36, 0.48, 0.6, 0.72, 0.84, 0.96, 1.08, 1.2])
y = scipy.array([0, 0.026, 0.096, 0.2, 0.326, 0.464, 0.601, 0.727, 0.831, 0.901, 0.927])
result = scipy.poly1d([0.0]) #setting result = 0

for i in range(0,len(x)): #number of polynomials L_k(x).
    temp_numerator = scipy.poly1d([1.0]) # resets temp_numerator such that a new numerator can be created for each i.
    denumerator = 1.0 #resets denumerator such that a new denumerator can be created for each i.
    for j in range(0,len(x)):
        if i != j:
            temp_numerator *= scipy.poly1d([1.0,-x[j]]) #finds numerator for L_i
            denumerator *= x[i]-x[j] #finds denumerator for L_i
    result += (temp_numerator/denumerator) * y[i] #linear combination
 
print("The result is: ")
print(result)
 
x_val = np.arange(min(x),max(x)+1, 0.1) #generates x values we would like to evaluate.
plt.xlabel('x'); plt.ylabel('p(x)')
plt.grid(True)
for i in range(0,len(x)):
    plt.plot([x[i]],[y[i]],'ro') #plot the points
plt.plot(x_val, result(x_val)) #result(x_val) gives the value of our Lagrange polynomial.
plt.axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
plt.show()
# -*- coding: Windows-1252 -*-
 
import sys
print sys.stdout.encoding
import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 0.12, 0.24, 0.36, 0.48, 0.6, 0.72, 0.84, 0.96, 1.08, 1.2])
y = np.array([0, 0.0257, 0.0958, 0.1991, 0.3246, 0.4611, 0.5976, 0.7231, 0.8265, 0.8965, 0.9222])
plt.figure(figsize=(10, 6), dpi=80)
plt.plot(x, y)
plt.title(u"Función Hidrostática")
plt.xlabel('Calado [m]')
plt.ylabel('Desplazamiento [t]')
plt.grid(True)
plt.show()

plt.savefig("grafico.pdf")

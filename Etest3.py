import numpy as np
from EfieldClass import *

#Plotting two paralell lines with opposite charge, similar to a capacitor

A = chrgplotter()

A.makeline(np.array([0, 0]), np.array([6, 0]), 1e-6)
A.makeline(np.array([0, 6]), np.array([6, 6]), -1e-6)

x = np.linspace(-1, 7, 20)
y = x
A.plotEfield(x, y)
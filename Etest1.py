import numpy as np
from EfieldClass import *

#Plotting two point charges with opposite charge

A = chrgplotter()

A.makepoint(2, 3, 1e-16)
A.makepoint(-2, -3, -1e-16)

x = np.linspace(-4, 4, 20)
y = x
A.plotEfield(x, y)

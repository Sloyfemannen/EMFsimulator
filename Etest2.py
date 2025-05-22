import numpy as np
from EfieldClass import *

#Plotting 2D circular shell with even charge distribution

A = chrgplotter()

A.makeshell(2)

x = np.linspace(-6, 6, 20)
y = x
A.plotEfield(x, y)
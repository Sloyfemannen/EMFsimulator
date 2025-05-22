import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

epsilon_0 = 8.85418782e-12
const = 1 / (4 * np.pi * epsilon_0)

class pointch():

    def __init__(self, x: float, y: float, charge: float):
        self.x = x
        self.y = y
        self.charge = charge

    def coulomb(self, x, y):
        r = np.array([x - self.x, y - self.y])
        return const * self.charge * r / np.linalg.norm(r)**3
    
    def Efield(self, x: np.array, y: np.array):
        X = x - self.x
        Y = y - self.y
        U = const * self.charge * X / np.sqrt(X**2 + Y**2)**2
        V = const * self.charge * Y / np.sqrt(X**2 + Y**2)**2
        return [U, V]
    
    def Vfield(self, x: np.array, y: np.array):
        X = x - self.x
        Y = y - self.y
        V = const * self.charge / np.sqrt(X**2 + Y**2)
        return V

class chrgplotter():
    
    def __init__(self):
        self.chobjs = []
            
    def makepoint(self, x: float, y: float, charge: float):
        self.chobjs.append(np.array([pointch(x, y, charge)]))
        self.charges = np.concatenate(self.chobjs)
    
    def makeline(self, p1: np.array, p2: np.array, charge=1e-6, n=1000):
        chdensity = charge / n
        xline = np.linspace(p1[0], p2[0], n)
        yline = np.linspace(p1[-1], p2[-1], n)
        line = np.array([pointch(i, j, chdensity) for i, j in zip(xline, yline)])
        self.chobjs.append(line)
        self.charges = np.concatenate(self.chobjs)
    
    def makeshell(self, r: float, charge = 1e-6, posx=0, posy=0):
        t = np.linspace(0, 2*np.pi, 1000)
        chdensity = charge / len(t)
        q1, q2 = r * np.cos(t), r * np.sin(t)
        charges = np.array([pointch(x - posx, y - posy, chdensity) for x, y in zip(q1, q2)])
        self.chobjs.append(charges)
        self.charges = np.concatenate(self.chobjs)
    
    def makeball(self, r, charge = 1e-6, posx=0, posy=0):
        d = np.linspace(-r, r, 50)
        area = []
        for i in d:
            for j in d:
                if np.sqrt(i**2 + j**2)**2 <= r:
                    area.append(np.array([i, j]))
        area = np.array(area)
        chdensity = charge / len(area)
        area = area.transpose()
        charges = np.array([pointch(x - posx, y - posy, chdensity) for x, y in zip(area[0], area[1])])
        self.chobjs.append(charges)
        self.charges = np.concatenate(self.chobjs)
    
    def calcEfield(self, x, y):
        return sum(np.array([A.Efield(x, y) for A in self.charges]))
    
    def calcVfield(self, x, y):
        return sum(np.array([A.Vfield(x, y) for A in self.charges]))

    def plotEfield(self, x: np.array, y: np.array):
        
        x_i, y_i = np.meshgrid(x, y, indexing='xy')
        
        fisx = x[-1] - x[0]
        fisy = y[-1] - y[0]
        plt.figure(figsize=(fisx, fisy))
        
        for c in self.chobjs:
            chargepoints = np.array([np.array([chrg.x, chrg.y]) for chrg in c])
            chargepoints = chargepoints.transpose()
            plt.plot(chargepoints[0], chargepoints[1], '.')
            
    
        u = self.calcEfield(x_i, y_i)
        plt.quiver(x_i, y_i, u[0], u[1], color="blue")
        plt.show()
    
    def plotVfield(self, x, y):
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        
        x_i, y_i = np.meshgrid(x, y, indexing='xy')
        V = self.calcVfield(x_i, y_i)
        
        ax.plot_surface(x_i, y_i, V, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        #ax.set_zlim(-6000, 2000)
        #ax.zaxis.set_major_locator(LinearLocator(10))
        plt.show()

    def twodEplot(self, start, end):

        xline = np.linspace(start[0], end[0], 1000)
        yline = np.linspace(start[1], end[1], 1000)
        line  = np.linspace(np.linalg.norm(start), np.linalg.norm(end), 1000)

        fieldline = self.calcEfield(xline, yline)
        fieldline = np.sqrt(fieldline[0]**2 + fieldline[1]**2)

        plt.plot(line, fieldline)
        plt.show()
        
    def twodVplot(self, start, end):

        xline = np.linspace(start[0], end[0], 1000)
        yline = np.linspace(start[1], end[1], 1000)
        line  = np.linspace(np.linalg.norm(start), np.linalg.norm(end), 1000)

        fieldline = self.calcVfield(xline, yline)
        #fieldline = np.sqrt(fieldline[0]**2 + fieldline[1]**2)

        plt.plot(line, fieldline)
        plt.show()
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.optimize import minimize
from typing import Tuple
import math


P = nx.Graph() #one Graph is for real joints and connections, the other is for theoretical ones such as center points and radius 
C = nx.Graph() 

class Center:
    """Since we only want one rotation point, a Singleton is used here"""
    _instance = None 

    def __new__(cls, name: str, posX: int, posY: int, rotatingPoint, angle : float = None, radius : int = None):
        if cls._instance is None:
            cls._instance = super(Center, cls).__new__(cls)
            cls._instance.name = name
            cls._instance.posX = posX
            cls._instance.posY = posY
            cls._instance.radius = radius
            cls._instance.rotatingPoint = rotatingPoint
            C.add_node(cls._instance.name, pos=(cls._instance.posX, cls._instance.posY))
            
            if C.degree(cls._instance.name) <= 1:
                C.add_edge(cls._instance.name, rotatingPoint.name)
                C.nodes[rotatingPoint.name]['pos'] = (rotatingPoint.posX, rotatingPoint.posY)
            else:
                print("Kann nicht mehr als ein Knoten hinzugefügt werden")
            
        return cls._instance
    
    def __init__(self, name: str, posX: int, posY: int, rotatingPoint, angle : float = None, radius : int = None ):
        if not hasattr(self, 'initialized'):
            self.name = name
            self.posX = posX
            self.posY = posY
            self.radius = radius
            self.angle = angle
            self.rotatingPoint = rotatingPoint
            
            C.add_node(self.name, pos=(self.posX, self.posY))
            self.initialized = True
    

    def rotate_point(self, degree: float):
        if self.angle is None:
            self.angle = math.atan2(self.rotatingPoint.posY - self.posY, self.rotatingPoint.posX - self.posX)

        if self.radius is None:
            self.radius = math.sqrt((self.posX - self.rotatingPoint.posX) ** 2 + (self.posY - self.rotatingPoint.posY) ** 2)

        # Convert degree to radians
        rad = math.radians(degree)
        # Calculate new angle
        newAngle = self.angle + rad
        # Calculate new position
        newPosX = self.posX + self.radius * math.cos(newAngle)
        newPosY = self.posY + self.radius * math.sin(newAngle)
        # Update position of rotatingPoint
        self.rotatingPoint.update_position(newPosX, newPosY)
        # Update the angle
        self.angle = newAngle

                
        
    

class Point():
    """These are points that are not center of a circle"""
    allPoints = [] #creates a list which is shared between all objects

    def __init__(self, name: str, posX: int, posY: int, isFixed: bool):
        
        self.name = name
        self.posX = posX
        self.posY = posY
        self.isFixed = isFixed
        self.vec = np.array([[self.posX], [self.posY]])
        self.connectedPoints = []
        Point.allPoints.append(self) #adds Point object to this list
        P.add_node(self.name, pos=(self.posX, self.posY))
    
    def get_position(self):
        return self.posX, self.posY

    def update_position(self, newPosX, newPosY):
        if self.isFixed == False:
            self.posX = newPosX
            self.posY = newPosY
            P.nodes[self.name]['pos'] = (self.posX, self.posY)
            self.vec = np.array([[self.posX], [self.posY]])  # Update the vec attribute

    def add_connection(self, *points):
        for point in points:
            if isinstance(point, Point) and point not in self.connectedPoints:
                self.connectedPoints.append(point)
                P.add_edge(self.name, point.name)

    def get_connection(self):
        return list(P.neighbors(self.name))


class Calculation:

    lVecList = []

    @classmethod
    def create_xVec(cls, points: list):
        cls.xVec = np.empty((0,0))
        """This creates a column vector similar to x in script"""
        for point in points:
            vecs = [point.vec for point in points]
            cls.xVec = np.vstack(vecs)
        print(f"xVec:\n {cls.xVec}")

    @classmethod
    def create_AMatrix(cls, points: list):
        cls.AMatrix = np.zeros((int(P.number_of_edges()) * 2, int(P.number_of_nodes()) * 2))
        #print(f"{[point.name for point in points]}")  # Ausgabe der Namen der Punkte
        edges = list(P.edges())
        #print(f"{edges}")
        #print(f"Points: {len(points)}, Connections: {len(edges)}")
        #Creates a dictionary, that maps the names to according indicies!!!1!11!
        #Essential to create the matrix
        nodeIndices = {point.name: i for i, point in enumerate(points)}
        row = 0
        for edge in edges:
            #splits edge in start and end point
            start, end = edge
            cls.AMatrix[row, nodeIndices[start] * 2] = 1
            cls.AMatrix[row+1, nodeIndices[start] * 2 + 1] = 1
            cls.AMatrix[row, nodeIndices[end] * 2] = -1
            cls.AMatrix[row+1, nodeIndices[end] * 2 + 1] = -1
            row += 2
        print(f"AMatrix:\n {cls.AMatrix}")
    
    @classmethod
    def create_lVec(cls):
        # Reset the lVec attribute
        cls.lVec = np.empty((0,1))  # Initialize with 1 column, so you can add
        cls.LVec = np.empty((0,0))
        # Recalculate LVec
        cls.LVec = cls.AMatrix @ cls.xVec  # Care about order, matrices are not commutative
        numRows = cls.LVec.shape[0] // 2
        cls.LVec = cls.LVec.reshape(numRows, 2)
        print(f"LVec:\n {cls.LVec}")

        for i in range(0, int(cls.LVec.shape[0])-1): 
            # Calculating the distance between the two points 
            numVecX = np.array([math.sqrt(cls.LVec[i,0]**2 + cls.LVec[i, 1]**2)]) 
            cls.lVec = np.vstack((cls.lVec, numVecX))
            
            numVecY = np.array([math.sqrt(cls.LVec[i+1,0]**2 + cls.LVec[i+1,1]**2)]) 
            cls.lVec = np.vstack((cls.lVec, numVecY)) 

        print(f"lVec:\n {cls.lVec}")
        if len(cls.lVecList) >= 2:
            cls.lVecList.pop(0)
        cls.lVecList.append(cls.lVec)

    @classmethod
    def calculate_error(cls):
        if len(cls.lVecList) >= 2:
            cls.eVec = np.empty((0,0))
            firstlVec = cls.lVecList[0]
            secondlVec = cls.lVecList[1]
            differences = secondlVec - firstlVec
            cls.eVec = differences
            print(f"Differences:\n{cls.eVec}")

            def error_function(params):
                new_positions = params.reshape((-1, 2))
                total_error = np.sum(np.abs(new_positions - cls.eVec))
                return total_error

            initial_guess = cls.eVec.flatten()
            result = minimize(error_function, initial_guess, method='BFGS')
            optimized_params = result.x.reshape(cls.eVec.shape)
            print(f"Optimized parameters: \n{optimized_params}")
            
        else:
            print("Not enough lVecs in lVecList to calculate differences.")
            
            

        


            


p0Vec = Point("A",0, 0, True)
p1Vec = Point("B",10, 35, False)
p2Vec = Point("C",-25, 10, False)
centerVec = Center("center", -30, 0, p2Vec,angle = None, radius=None)

p0Vec.add_connection(p1Vec)
p1Vec.add_connection(p2Vec)


Calculation.create_xVec(Point.allPoints)
Calculation.create_AMatrix(Point.allPoints)
Calculation.create_lVec()

centerVec.rotate_point(10)
print("\n Winkel erhöht \n")

Calculation.create_xVec(Point.allPoints)
Calculation.create_AMatrix(Point.allPoints)
Calculation.create_lVec()

Calculation.calculate_error()

"""
def update(num, p0Vec):

    ax.clear()
    centerVec.rotate_point(1)
    #p0Vec.update_position(p0Vec.posX+1, p0Vec.posY)
    pos = (nx.get_node_attributes(C, 'pos'))
    pos.update(nx.get_node_attributes(P, 'pos'))
    nx.draw(C, pos, with_labels=True, node_size=350, node_color="red", font_size=10, font_weight="bold", edge_color="red", style="dashed")
    nx.draw(P, pos, with_labels=True, node_size=350, node_color="skyblue", font_size=10, font_weight="bold")
    

fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, update, frames=range(800), fargs=(p0Vec,), interval=300, repeat=False)

plt.show()
"""
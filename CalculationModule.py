import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import Tuple
import math


P = nx.Graph() #one Graph is for real joints and connections, the other is for theoretical ones such as center points and radius 
C = nx.Graph() 

class Center():
    """Since we only want one rotation point, maybe consider a Singleton?"""
   
    def __init__(self, name: str, posX: int, posY: int):
        self.name = name
        self.posX = posX
        self.posY = posY
        C.add_node(self.name, pos=(self.posX, self.posY))
    
    def add_rotating_point(self, point):
        if C.degree(self.name) <= 1:
            C.add_edge(self.name, point.name)
        else:
            print("Kann nicht mehr als ein Knoten hinzugefügt werden")

    def rotate_point(self, degree: float):
        neighbors = list(C.neighbors(self.name))  # Gets the connected points
        if neighbors:
            neighbor = neighbors[0]  # Take the only neighbor
            neighborPos = C.nodes[neighbor]['pos']  # Gets the position of the connected point
            distance = math.sqrt((self.posX - neighborPos[0]) ** 2 + (self.posY - neighborPos[1]) ** 2)
            print(f"Distanz von {self.name} zu {neighbor}: {distance}")
        else:
            print(f"{self.name} hat keinen Nachbarn")

class Point(Center):
    """These are points that are not center of a circle"""
    allPoints = [] #creates a list which is shared between all objects

    def __init__(self, name: str, posX: int, posY: int, isFixed: bool):
        super().__init__(name, posX, posY)
    
        self.isFixed = isFixed
        self.vec = np.array([[self.posX], [self.posY]])
        self.connectedPoints = []
        Point.allPoints.append(self) #adds Point object to this list
        P.add_node(self.name, pos=(self.posX, self.posY))

    def update_position(self, newPosX, newPosY):
        self.posX = newPosX
        self.posY = newPosY
        P.nodes[self.name]['pos'] = (self.posX, self.posY)

    def add_connection(self, *points):
        for point in points:
            if isinstance(point, Point) and point not in self.connectedPoints:
                self.connectedPoints.append(point)
                P.add_edge(self.name, point.name)

    def get_connection(self):
        return list(P.neighbors(self.name))


class Calculation:

    xVec = np.empty((0, 0))  # Empty array with 0 rows and 0 columns
    AMatrix = np.empty((0,0))
    LVec = np.empty((0,0))
    lVec = np.empty((0,1)) #Initialize with 1 column, so you can add 

    @classmethod
    def create_xVec(cls, points: list):
        """This creates a column vector similar to x in script"""
        for point in points:
            vecs = [point.vec for point in points]
            cls.xVec = np.vstack(vecs)
        print(cls.xVec)

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
        #print(cls.AMatrix)
    
    @classmethod
    def create_lVec(cls):
        cls.LVec = cls.AMatrix @ cls.xVec #care about order, matrices are not commutative
        #print(cls.LVec)
        # Converts LVec into a 2-column vector
        numRows = cls.LVec.shape[0] // 2
        cls.LVec = cls.LVec.reshape(numRows, 2)
        #print(cls.LVec)

        for i in range(0, int(cls.LVec.shape[0])-1):
            #calculating the distance between the two points
            numVecX = np.array([math.sqrt(cls.LVec[i,0]**2 + cls.LVec[i+1,0]**2)])
            cls.lVec = np.vstack((cls.lVec, numVecX))
            numVecY = np.array([math.sqrt(cls.LVec[i,1]**2 + cls.LVec[i+1,1]**2)])
            
            cls.lVec = np.vstack((cls.lVec, numVecY))
        #print(cls.lVec)
        

p0Vec = Point("A",0, 0, True)
p1Vec = Point("B",10, 35, False)
p2Vec = Point("C",-25, 10, False)

centerVec = Center("center", -30, 0)
centerVec.add_rotating_point(p2Vec)


Calculation.create_xVec(Point.allPoints)
p0Vec.add_connection(p1Vec)
p1Vec.add_connection(p2Vec)
Calculation.create_AMatrix(Point.allPoints)
Calculation.create_lVec()
centerVec.rotate_point(10)


def update(num, p0Vec):

    ax.clear()
    p0Vec.update_position(p0Vec.posX + 1, p0Vec.posY)
    pos = (nx.get_node_attributes(C, 'pos'))
    pos.update(nx.get_node_attributes(P, 'pos'))
    nx.draw(C, pos, with_labels=True, node_size=700, node_color="red", font_size=10, font_weight="bold", edge_color="red", style="dashed")
    nx.draw(P, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_weight="bold")
    

fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, update, frames=range(20), fargs=(p0Vec,), interval=200, repeat=False)

plt.show()

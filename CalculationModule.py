import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.optimize import least_squares
import math
import pandas as pd

class Center:
    """Since we only want one rotation point, a Singleton is used here"""
    _instance = None

    def __new__(cls, name: str, posX: int, posY: int, rotatingPoint, angle: float = None):
        if cls._instance is None:
            cls._instance = super(Center, cls).__new__(cls)
            cls._instance.name = name
            cls._instance.posX = posX
            cls._instance.posY = posY
            cls._instance.rotatingPoint = rotatingPoint
            
        return cls._instance

    def __init__(self, name: str, posX: int, posY: int, rotatingPoint, angle: float = None):
        if not hasattr(self, 'initialized'):
            self.name = name
            self.posX = posX
            self.posY = posY
            self.angle = angle
            self.rotatingPoint = rotatingPoint
            self.radius = math.sqrt((self.posX - self.rotatingPoint.posX) ** 2 + (self.posY - self.rotatingPoint.posY) ** 2)
            self.initialized = True

    def rotate_point(self, degree: float):
        if self.angle is None:
            self.angle = math.atan2(self.rotatingPoint.posY - self.posY, self.rotatingPoint.posX - self.posX)
            
        else:
            newX = self.radius * math.cos(self.angle)
            newY = self.radius * math.sin(self.angle)
            self.rotatingPoint.update_position(newX, newY)
        
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

class Point:
    """These are points that are not center of a circle"""
    allPoints = []

    def __init__(self, name: str, posX: int, posY: int, isFixed: bool):
        self.name = name
        self.posX = posX
        self.posY = posY
        self.isFixed = isFixed
        self.vec = np.array([[self.posX], [self.posY]])
        self.connectedPoints = []
        Point.allPoints.append(self)  # adds Point object to this list

    def get_position(self):
        return self.posX, self.posY

    def update_position(self, newPosX, newPosY):
        if not self.isFixed:
            self.posX = newPosX
            self.posY = newPosY
            self.vec = np.array([[self.posX], [self.posY]])  # Update the vec attribute

    def add_connection(self, *points):
        for point in points:
            if isinstance(point, Point) and point not in self.connectedPoints:
                self.connectedPoints.append(point)

    def get_connection(self):
        return self.connectedPoints


class Calculation:

    lVecList = []

    @classmethod
    def create_xVec(cls, points: list):
        cls.xVec = np.empty((0,0))
        for point in points:
            vecs = [point.vec for point in points]
            cls.xVec = np.vstack(vecs)

    @classmethod
    def create_AMatrix(cls, points: list):
        num_edges = sum(len(point.connectedPoints) for point in points)
        num_nodes = len(points)
        cls.AMatrix = np.zeros((num_edges * 2, num_nodes * 2))

        nodeIndices = {point.name: i for i, point in enumerate(points)}
        row = 0
        for point in points:
            for connectedPoint in point.get_connection():
                start, end = point.name, connectedPoint.name
                cls.AMatrix[row, nodeIndices[start] * 2] = 1
                cls.AMatrix[row+1, nodeIndices[start] * 2 + 1] = 1
                cls.AMatrix[row, nodeIndices[end] * 2] = -1
                cls.AMatrix[row+1, nodeIndices[end] * 2 + 1] = -1
                row += 2
        print(f"AMatrix:\n {cls.AMatrix}")

    @classmethod
    def create_lVec(cls):
        cls.lVec = np.empty((0, 1))
        cls.LVec = cls.AMatrix @ cls.xVec
        numRows = cls.LVec.shape[0] // 2
        cls.LVec = cls.LVec.reshape(numRows, 2)

        for i in range(0, cls.LVec.shape[0] - 1):
            numVecX = np.array([math.sqrt(cls.LVec[i, 0] ** 2 + cls.LVec[i, 1] ** 2)])
            cls.lVec = np.vstack((cls.lVec, numVecX))

            numVecY = np.array([math.sqrt(cls.LVec[i+1, 0] ** 2 + cls.LVec[i+1, 1] ** 2)])
            cls.lVec = np.vstack((cls.lVec, numVecY))

        print(f"lVec:\n {cls.lVec}")
        if len(cls.lVecList) >= 2:
            cls.lVecList.pop(0)
        cls.lVecList.append(cls.lVec)

    @classmethod
    def calculate_error(cls):
        if len(cls.lVecList) >= 2:
            cls.eVec = cls.lVecList[1] - cls.lVecList[0]
        else:
            cls.eVec = np.zeros((len(cls.lVec), 1))

        print(f"Differences:\n{cls.eVec}")


edited_data = pd.DataFrame({
        "Punkt": ["B", "C", "E", "center"],
        "x": [0, 10, -25, -30],
        "y": [0, 35, 10, 0],
        "Fest": [True, False, False, True]  # True = fest, False = lose
    })

for node in edited_data.iterrows():
    if node[0] == edited_data.shape[0]-1:
        centerVec = Center(node[1]["Punkt"], node[1]["x"], node[1]["y"], Point.allPoints[-1])
    else:
        Point(node[1]["Punkt"], node[1]["x"], node[1]["y"], node[1]["Fest"])

for i in range(len(Point.allPoints)-1):
    Point.allPoints[i].add_connection(Point.allPoints[i+1])

Calculation.create_xVec(Point.allPoints)
Calculation.create_AMatrix(Point.allPoints)
Calculation.create_lVec()
Calculation.calculate_error()

print("\n Winkel ändern \n")
centerVec.rotate_point(10)

Calculation.create_xVec(Point.allPoints)
Calculation.create_AMatrix(Point.allPoints)
Calculation.create_lVec()
Calculation.calculate_error()

points = [p for p in Point.allPoints]
print(points)
points.append(centerVec)
print(points)

fig, ax = plt.subplots()


#p0Vec = Point("A", 0, 0, False)
#p1Vec = Point("B", 10, 35, False)
#p2Vec = Point("C", -25, 10, False)
#centerVec = Center("center", -30, 0, p2Vec)

#p0Vec.add_connection(p1Vec)
#p1Vec.add_connection(p2Vec)

fig, ax = plt.subplots()
ax.set_title('Punkte und Verbindungen')

def update(num):
    ax.clear()
    # Bewege p1Vec um 1 in X-Richtung
    points[1].update_position(points[1].posX + 1, points[1].posY)
    centerVec.rotate_point(5)
    # Zeichne die Punkte
    for point in points:
        ax.plot(point.posX, point.posY, 'o', markersize=10)
        ax.text(point.posX, point.posY, point.name, fontsize=12, ha='right')

    # Zeichne die Verbindungen
    for point in points[:-1]:  # Der letzte Punkt ist centerVec und hat keine Verbindung
        for connectedPoint in point.connectedPoints:
            ax.plot([point.posX, connectedPoint.posX], [point.posY, connectedPoint.posY], 'r-')

    # Zeichne centerVec und seine Verbindung
    ax.plot(centerVec.posX, centerVec.posY, 'o', markersize=10, color='green')
    ax.text(centerVec.posX, centerVec.posY, centerVec.name, fontsize=12, ha='right')
    ax.plot([centerVec.posX, centerVec.rotatingPoint.posX], [centerVec.posY, centerVec.rotatingPoint.posY], 'g--')

    ax.set_xlim(-50, 50)
    ax.set_ylim(-10, 50)
    ax.set_xlabel('X-Achse')
    ax.set_ylabel('Y-Achse')
    ax.set_title('Punkte und Verbindungen')

ani = animation.FuncAnimation(fig, update, frames=range(800), interval=300, repeat=False)

plt.show()
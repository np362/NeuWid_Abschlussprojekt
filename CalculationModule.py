import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.optimize import least_squares
import math
from typing import Tuple

class Center:
    """Only one instance of Center can exist, insert angle in degrees"""
    _instance = None

    def __new__(cls, name: str, posX: int, posY: int, rotatingPoint):
        if cls._instance is None:
            cls._instance = super(Center, cls).__new__(cls)
            cls._instance.name = name
            cls._instance.posX = posX
            cls._instance.posY = posY
            cls._instance.rotatingPoint = rotatingPoint
            cls._instance.radius = math.sqrt((posX - rotatingPoint.posX) ** 2 + (posY - rotatingPoint.posY) ** 2)
            cls._instance.angle = math.atan2(rotatingPoint.posY - posY, rotatingPoint.posX - posX)
            

        return cls._instance

    def __init__(self, name: str, posX: int, posY: int, rotatingPoint):
        if not hasattr(self, 'initialized'):
            self.name = name
            self.posX = posX
            self.posY = posY
            self.rotatingPoint = rotatingPoint
            self.radius = math.sqrt((posX - rotatingPoint.posX) ** 2 + (posY - rotatingPoint.posY) ** 2)
            self.angle = math.atan2(rotatingPoint.posY - posY, rotatingPoint.posX - posX)           
            self.initialized = True

    def rotate_point(self, degree: float):
        
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

    def __init__(self, name: str, posX: int, posY: int, isFixed: bool):
        self.name = name
        self.posX = posX
        self.posY = posY
        self.isFixed = isFixed
        self.vec = np.array([[self.posX], [self.posY]])
        self.connectedPoints = []
        
    def get_position(self):
        return self.posX, self.posY

    def update_position(self, newPosX, newPosY):
        if not self.isFixed:
            self.posX = newPosX
            self.posY = newPosY
            self.vec = np.array([[self.posX], [self.posY]])  # Update the vec attribute

    def add_connection(self, point):
        """The order of connection doesnt matter - can be point1 -> point2 or other way around"""
        if isinstance(point, Point) and point not in self.connectedPoints:
            self.connectedPoints.append(point)

    def get_connection(self):
        return self.connectedPoints

class Calculation():

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
        #print(f"AMatrix:\n {cls.AMatrix}")

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

        #print(f"lVec:\n {cls.lVec}")
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

    @classmethod
    def output_error(cls, points : list):
        Calculation.create_xVec(points)
        Calculation.create_AMatrix(points)
        Calculation.create_lVec()
        Calculation.calculate_error()

    
    @classmethod
    def distance(cls, p1, p2):
        """Calculates euclidean distance between points"""
        return np.linalg.norm(np.array(p1.get_position()) - np.array(p2.get_position()))
    
    @classmethod
    def residuals(cls, params, fixedPoints, desiredDistances):
        x, y = params
        tempPoint = Point("Temp", x, y, False)
        residuals = []
        for fixedPoint, desiredDistance in zip(fixedPoints, desiredDistances):
            currentDistance = cls.distance(tempPoint, fixedPoint)
            residuals.append(currentDistance - desiredDistance)
        return residuals
    
    @classmethod
    def optimize_all_positions(cls, points, distances : Tuple, tolerance=5, max_iterations=100):
        """points - list of all Points, no Center! distances - tuples with (moving point, fixed point, distance between points)
           If a Tuple contains two moving points the order doesnt matter
        """

        for _ in range(max_iterations):
            max_error = 0
            for i, (p1, p2, desiredDistance) in enumerate(distances): 
                
                if not p1.isFixed:  
    
                    initialPosition = p1.get_position()
                    
                    fixedPoints = [p2] #not fixed in the sense that it doesn´t move, just the point the distance is referenced to 
                    desiredDistances = [desiredDistance]
                    for p, f, d in distances:  
                        if p == p1 and not p.isFixed:
                            fixedPoints.append(f)
                            desiredDistances.append(d)
                    result = least_squares(cls.residuals, initialPosition, jac='2-point', args=(fixedPoints, desiredDistances), max_nfev=1000)
                    p1.update_position(result.x[0], result.x[1])
                    error = abs(cls.distance(p1, p2) - desiredDistance)
                    if error > max_error:
                        max_error = error
            if max_error < tolerance:
                break

        return [p.get_position() for p in points]



p0Vec = Point("A", 0, 0, True) 
p1Vec = Point("B", 10, 35, False)
p2Vec = Point("C", -25, 10, False)
p3Vec = Point("D", 20, 20, False)
p4Vec = Point("E", -10, -30, False)


centerVec = Center("Center", -30, 0, p2Vec)

desDistp0p1 = Calculation.distance(p0Vec, p1Vec)
desDistp1p2 = Calculation.distance(p1Vec, p2Vec)
desDistp0p3 = Calculation.distance(p0Vec, p3Vec)
desDistp1p3 = Calculation.distance(p1Vec, p3Vec)

desDistp2p4 = Calculation.distance(p2Vec, p4Vec) 
desDistp2p3 = Calculation.distance(p2Vec, p3Vec)
desDistp0p4 = Calculation.distance(p0Vec, p4Vec)



p0Vec.add_connection(p1Vec) #p2Vec ist C
p1Vec.add_connection(p2Vec) #p0Vec ist A -fix
p0Vec.add_connection(p3Vec)
p1Vec.add_connection(p3Vec)
p2Vec.add_connection(p4Vec)
p3Vec.add_connection(p2Vec)
p0Vec.add_connection(p4Vec)


#It is important to list like the following: moving Point - fixed/referenced Point - distance
distances = [
    (p1Vec, p2Vec, desDistp1p2), 
    (p1Vec, p0Vec, desDistp0p1),
    (p3Vec, p0Vec, desDistp0p3),
    (p3Vec, p1Vec, desDistp1p3),
    (p3Vec, p2Vec, desDistp2p3),
    (p4Vec, p2Vec, desDistp2p4),
    (p4Vec, p0Vec, desDistp0p4)
    ]

points = [p0Vec, p1Vec, p2Vec, p3Vec, p4Vec]

# Listen zum Speichern der Positionen
p1_positions = []
p2_positions = []
p3_positions = []

print("Start der Animation")

def update(num):
    ax.clear()
    
    # Ändere die Position von p2Vec
    centerVec.rotate_point(10)
    Calculation.output_error(points)
    
    # Optimiere die Position von p1Vec basierend auf p2Vec
    Calculation.optimize_all_positions(points, distances)
    

    
    # Speichere die aktuellen Positionen
    p1_positions.append(p1Vec.get_position())
    p2_positions.append(p2Vec.get_position())

    # Zeichne die Punkte
    for point in points:
        ax.plot(point.posX, point.posY, 'o', markersize=10)
        ax.text(point.posX, point.posY, point.name, fontsize=12, ha='right')

    # Zeichne die Verbindungen
    for point in points[:-1]:  # Der letzte Punkt ist centerVec und hat keine Verbindung
        for connected_point in point.get_connection():
            ax.plot([point.posX, connected_point.posX], [point.posY, connected_point.posY], 'r-')
    
    # Zeichne centerVec und seine Verbindung
    ax.plot(centerVec.posX, centerVec.posY, 'o', markersize=10, color='green')
    ax.text(centerVec.posX, centerVec.posY, centerVec.name, fontsize=12, ha='right')
    ax.plot([centerVec.posX, centerVec.rotatingPoint.posX], [centerVec.posY, centerVec.rotatingPoint.posY], 'g--')

    # Zeichne die Bahnkurven der Punkte
    if len(p1_positions) > 1:
        ax.plot([pos[0] for pos in p1_positions], [pos[1] for pos in p1_positions], label='Punkt B Bahnkurve')
    if len(p2_positions) > 1:
        ax.plot([pos[0] for pos in p2_positions], [pos[1] for pos in p2_positions], label='Punkt C Bahnkurve')
    if len(p3_positions) > 1:
        ax.plot([pos[0] for pos in p3_positions], [pos[1] for pos in p3_positions], label='Punkt D Bahnkurve')

    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_xlabel('X-Achse')
    ax.set_ylabel('Y-Achse')
    ax.set_title('Punkte und Verbindungen')

    ax.legend()


fig, ax = plt.subplots() 
ani = animation.FuncAnimation(fig, update, frames=range(800), interval=100, repeat=False) 
plt.show()

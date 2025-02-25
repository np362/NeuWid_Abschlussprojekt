import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.optimize import least_squares
import math

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

def distance(p1, p2):
    return np.linalg.norm(np.array(p1.get_position()) - np.array(p2.get_position()))

def residuals(params, fixed_point, desired_distance):
    x, y = params
    temp_point = Point("Temp", x, y, False)
    current_distance = distance(temp_point, fixed_point)
    return [current_distance - desired_distance]

def optimize_positions(p1, p2, desiredDistance):
    # Gewünschte Distanz zwischen den Punkten
    print(f"Desire Distance: {desiredDistance}")
    # Anfangsposition für p1
    initial_position = p1.get_position()
    print(f"Inital Position {initial_position}")

    # Least-Squares-Optimierung durchführen, um p1 anzupassen
    result = least_squares(residuals, initial_position, args=(p2, desiredDistance))

    # Aktualisiere die Position von p1 mit den optimierten Werten
    p1.update_position(result.x[0], result.x[1])


p0Vec = Point("A", 0, 0, False)
p1Vec = Point("B", 10, 35, False)
p2Vec = Point("C", -25, 10, False)
p3Vec = Point("C", 25, 10, False)
centerVec = Center("Center", -30, 0, p2Vec)

desiredDistancep1p2 = distance(p1Vec, p2Vec)
desiredDistancep0p1 = distance(p0Vec, p2Vec)
desiredDistancep1p3 = distance(p1Vec, p3Vec)

p0Vec.add_connection(p1Vec)
p1Vec.add_connection(p2Vec)
p3Vec.add_connection(p1Vec)

points = [p0Vec, p1Vec, p2Vec, p3Vec, centerVec]
    
def update(num):
    ax.clear()
    
    # Ändere die Position von p2Vec
    centerVec.rotate_point(10)
    # Optimiere die Position von p1Vec basierend auf p2Vec
    optimize_positions(p1Vec, p2Vec, desiredDistancep1p2)
    optimize_positions(p1Vec, p0Vec, desiredDistancep0p1)
    optimize_positions(p3Vec, p1Vec, desiredDistancep1p3)
    print(f"Distance: {distance(p1Vec, p2Vec)}")
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

    ax.set_xlim(-50, 50)
    ax.set_ylim(-10, 50)
    ax.set_xlabel('X-Achse')
    ax.set_ylabel('Y-Achse')
    ax.set_title('Punkte und Verbindungen')

fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, update, frames=range(800), interval=300, repeat=False)

plt.show()

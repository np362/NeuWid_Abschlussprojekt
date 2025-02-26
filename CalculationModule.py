import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.optimize import least_squares
import math
import pandas as pd

class Center:
    """Since we only want one rotation point, a Singleton is used here"""
    _instance = None

    def __new__(cls, name: str, posX: int, posY: int, rotatingPoint):
        if cls._instance is None:
            cls._instance = super(Center, cls).__new__(cls)
            cls._instance.name = name
            cls._instance.posX = posX
            cls._instance.posY = posY
            cls._instance.rotatingPoint = rotatingPoint
        return cls._instance

    def __init__(self, name: str, posX: int, posY: int, rotatingPoint):
        if not hasattr(self, 'initialized'):
            self.name = name
            self.posX = posX
            self.posY = posY
            self.rotatingPoint = rotatingPoint
            self.radius = math.sqrt((self.posX - self.rotatingPoint.posX) ** 2 + (self.posY - self.rotatingPoint.posY) ** 2)
            self.angle = math.atan2( self.rotatingPoint.posY - self.posY, self.rotatingPoint.posX - self.posX)
            print(f"Radius: {self.radius}")
           
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
        if isinstance(point, Point) and point not in self.connectedPoints:
            self.connectedPoints.append(point)
        
    def remove_connection(self, point):
        if point in self.connectedPoints:
            self.connectedPoints.remove(point)

    def get_connection(self):
        return self.connectedPoints

class Calculation():
    @classmethod
    def distance(cls, p1, p2):
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
    def optimize_all_positions(cls, points, distances, tolerance=0.6, max_iterations=100):
        
        for _ in range(max_iterations):
            max_error = 0.3
            for i, (p1, p2, desiredDistance) in enumerate(distances):  # Hier erwarten wir 3 Werte
                
                if not p1.isFixed:  # Nur Punkte optimieren, die nicht fest sind
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

edited_data = pd.DataFrame({
            "Punkt": ["B", "C", "D", "A", "E", "center"],
            "x": [0, 10, -25, 25, 5, -30],
            "y": [0, 35, 10, 10, 10, 0],
            "Fest": [True, False, False, False, True, True]  # True = fest, False = lose
        })

# points = []
# for node in edited_data.iterrows():
#     if node[0] == edited_data.shape[0]-1:
#         centerVec = Center(node[1]["Punkt"], node[1]["x"], node[1]["y"], points[2])
#     else:
#         points.append(Point(node[1]["Punkt"], node[1]["x"], node[1]["y"], node[1]["Fest"]))



p0Vec = Point("A", 0, 0, True)  # p0Vec auf True gesetzt
p1Vec = Point("B", 10, 35, False)
p2Vec = Point("C", -25, 10, False)
#p3Vec = Point("D", 25, 10, False)
#p4Vec = Point("E", 5, 10, True)
centerVec = Center("Center", -30, 0, p2Vec)

p1Vec.add_connection(p0Vec)
p1Vec.add_connection(p2Vec)
#p3Vec.add_connection(p1Vec)
#p3Vec.add_connection(p4Vec)

# points[1].add_connection(points[0])
# points[2].add_connection(points[1])
# points[3].add_connection(points[1])
# points[3].add_connection(points[4])

print(f"Winkel: {math.degrees(math.atan2(centerVec.rotatingPoint.posY - centerVec.posY, centerVec.rotatingPoint.posX - centerVec.posX))}")

# distances = []
# for point1 in points:
#     for point2 in points:
#         if point1 != point2:
#             print(f"Distance {point1.name} - {point2.name}: {Calculation.distance(point1, point2)}")
#             distances.append((point1, point2, Calculation.distance(point1, point2)))

desiredDistancep1p2 = Calculation.distance(p1Vec, p2Vec)
desiredDistancep0p1 = Calculation.distance(p0Vec, p1Vec)
#desiredDistancep1p3 = Calculation.distance(p1Vec, p3Vec)
#desiredDistancep3p4 = Calculation.distance(p4Vec, p3Vec)

# desiredDistancep1p2 = Calculation.distance(points[1], points[2])
# desiredDistancep0p1 = Calculation.distance(points[0], points[1])
# desiredDistancep1p3 = Calculation.distance(points[1], points[3])
#desiredDistancep3p4 = Calculation.distance(points[4], points[3])


#print(f"Connected Points: {len(points[3].connectedPoints)}")

#distances = [(points[1], points[2], desiredDistancep1p2), (points[1], points[0], desiredDistancep0p1), (points[3], points[1], desiredDistancep1p3), (points[3], points[4], desiredDistancep3p4)]

#It is important to list like the following: moving Point - fixed/referenced Point - distance
distances = [(p1Vec, p2Vec, desiredDistancep1p2), (p1Vec, p0Vec, desiredDistancep0p1)]#, (p3Vec, p1Vec, desiredDistancep1p3), (p3Vec, p4Vec, desiredDistancep3p4)]
points = [p0Vec, p1Vec, p2Vec]#, p3Vec, p4Vec]

# Listen zum Speichern der Positionen
p1_positions = []
p2_positions = []
p3_positions = []

    
def update(num):
    ax.clear()
    
    # Ändere die Position von p2Vec
    centerVec.rotate_point(1)
    # Optimiere die Position von p1Vec basierend auf p2Vec
    Calculation.optimize_all_positions(points, distances)
    
    #Speichere die aktuellen Positionen
    p1_positions.append(p1Vec.get_position())
    p2_positions.append(p2Vec.get_position())
    #p3_positions.append(p3Vec.get_position())


    #Zeichne die Punkte
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

    ax.set_xlim(-50, 50)
    ax.set_ylim(-10, 50)
    ax.set_xlabel('X-Achse')
    ax.set_ylabel('Y-Achse')
    ax.set_title('Punkte und Verbindungen')

    ax.legend()


fig, ax = plt.subplots() 
ani = animation.FuncAnimation(fig, update, frames=range(800), interval=100, repeat=False) 
plt.show()
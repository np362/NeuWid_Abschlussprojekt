import numpy as np
from scipy.optimize import least_squares
import math

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

        if len(cls.lVecList) >= 2:
            cls.lVecList.pop(0)
        cls.lVecList.append(cls.lVec)

    @classmethod
    def calculate_error(cls):
        if len(cls.lVecList) >= 2 and cls.lVecList[1].shape == cls.lVecList[0].shape:
            cls.eVec = cls.lVecList[1] - cls.lVecList[0]
        else:
            cls.eVec = np.zeros((len(cls.lVec), 1))


    @classmethod
    def output_error(cls, points : list):
        Calculation.create_xVec(points)
        Calculation.create_AMatrix(points)
        Calculation.create_lVec()
        Calculation.calculate_error()

    @staticmethod
    def distance(p1, p2):
        return np.linalg.norm(np.array(p1.get_position()) - np.array(p2.get_position()))
    
    @staticmethod
    def residuals(params, fixedPoints, desiredDistances):
        x, y = params
        tempPoint = Point("Temp", x, y, False)
        residuals = []
        for fixedPoint, desiredDistance in zip(fixedPoints, desiredDistances):
            currentDistance = Calculation.distance(tempPoint, fixedPoint)
            residuals.append(currentDistance - desiredDistance)
        return residuals
    
    @staticmethod
    def optimize_all_positions(points, distances, tolerance=0.6, max_iterations=100):
        fixedPoints = [p for p in points if p.isFixed]
        
        if len(fixedPoints) >= 3:
           raise ValueError("Genau zwei Fixpunkte erforderlich: Center + 1 weiterer Punkt!")

        for _ in range(max_iterations):
            max_error = 0
            for p1, p2, desiredDistance in distances:
                if not p1.isFixed:  # Nur bewegliche Punkte anpassen
                    initialPosition = p1.get_position()
                    result = least_squares(Calculation.residuals, initialPosition, args=([p2], [desiredDistance]), max_nfev=1000)
                    p1.update_position(result.x[0], result.x[1])
                    error = abs(Calculation.distance(p1, p2) - desiredDistance)
                    if error > max_error:
                        max_error = error

            if max_error < tolerance:
                break


        return [p.get_position() for p in points]

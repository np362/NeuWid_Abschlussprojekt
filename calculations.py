import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import math
from typing import Tuple


class Point:
    def __init__(self, posX: int, posY: int, isFixed: bool):
        self.posX = posX
        self.posY = posY
        self.isFixed = isFixed
        self.vec = np.array([[self.posX], [self.posY]])
        self.connectedPoints = []

    def draw_point(self):
        print(self.vec)

    def update_position(self, newPosX, newPosY):
        self.posX = newPosX
        self.posY = newPosY

    def add_connected_point(self, point):
        if isinstance(point, Point) and point not in self.connectedPoints: #Adds a Point to the connectedPoint-list, if it´s not itself
            self.connectedPoints.append(point)

    def display_connected_points(self):
        if not self.connectedPoints:
            print("No connected points")
        else:
            for point in self.connectedPoints:
                print(f'Connected point at ({point.posX}, {point.posY})')


class Calculation:
    points = []
    pointPairs = []

    @staticmethod
    def add_points(*points: Point):
        for point in points:
            if isinstance(point, Point):
                Calculation.points.append(point)

    @staticmethod
    def add_point_pair(point_pair: Tuple[Point, Point]):
        point1, point2 = point_pair
        if isinstance(point1, Point) and isinstance(point2, Point):
            Calculation.pointPairs.append(point_pair)

    @staticmethod
    def display_points():
        if not Calculation.points:
            print("Keine Punkte hinzugefügt")
        else:
            for point in Calculation.points:
                print(f'Point at ({point.posX}, {point.posY}), Fixed: {point.isFixed}')

    @staticmethod
    def display_point_pairs():
        if not Calculation.pointPairs:
            print("Keine Punktpaare hinzugefügt")
        else:
            for point1, point2 in Calculation.pointPairs:
                print(f'Pair: Point 1 at ({point1.posX}, {point1.posY}), Point 2 at ({point2.posX}, {point2.posY})')

# Beispielverwendung:
p0Vec = Point(0, 0, True)
p1Vec = Point(10, 35, True)
p2Vec = Point(-25, 10, True)
p3Vec = Point(15, 20, False)

Calculation.add_points(p0Vec, p1Vec, p2Vec)
Calculation.add_point_pair((p0Vec, p1Vec))
Calculation.add_point_pair((p2Vec, p3Vec))


p0Vec = np.array([[0], [0]])
p1Vec = np.array([[10], [30]])
p2Vec = np.array([[-25], [10]])

xVec = np.vstack((p0Vec, p1Vec, p2Vec))
print(xVec)

cVec = np.array([[-30], [0]])
# Matrix 2m x 2n, m-Points, n-rods
                #p0-x,y  p1-x.y
AMatrix = np.array([[1,0,-1,0,0,0], #connection of p0 to p1
                    [0,1,0,-1,0,0],
                    [0,0,1,0,-1,0], #connection of p1 to p2
                    [0,0,0,1,0,-1]])

diffVec = AMatrix @ xVec

# lVec umwandeln in die gewünschte Form
LMatrix = np.array([[diffVec[0,0], diffVec[1,0]],
                    [diffVec[2,0], diffVec[3,0]]])

lVec = np.empty((0,1))



for i in range(0, int(LMatrix.shape[0])-1):
    #calculating the distance between the two points
    print(f"{LMatrix[i,0]} {LMatrix[i+1,0]}")

    numVec = np.array([math.sqrt(LMatrix[i,0]**2 + LMatrix[i+1,0]**2)])
    lVec = np.vstack((lVec, numVec))
    numVec = np.array([math.sqrt(LMatrix[i,1]**2 + LMatrix[i+1,1]**2)])
    lVec = np.vstack((lVec, numVec))
   


print(lVec)

# Ausgabe der neuen Matrix


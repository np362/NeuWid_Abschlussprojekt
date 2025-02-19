"""This module contains all the calculations for the application."""

import numpy as np
import pandas as pd
import scipy as sp

class Mechanism():

    def __init__(self):
        self.matrix = None
        self.connections = None
        self.vector = None

    def calculate_track(self, *args):
        pass

    def Are_points_connected(self, point1, point2):
        pass

    @classmethod
    def calculate_matrix(cls, vec, FT_points):
        A = np.zeros([len(vec)-2, len(vec)])
        for state in range(len(FT_points)):
            if FT_points[state]:
                A[state][state] = 1
                A[state][state+2] = -1
        print(A)
        x = vec.T
        return A*x

if __name__ == "__main__":
    rechnung = Mechanism()
    punkte = np.array([1, 2, 3, 3, 2, 1])
    statements = np.array([True, False, True])
    rechnung.calculate_matrix(punkte, statements)


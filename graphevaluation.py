""" The following code is used to modify the uploaded file so that the mechanism can be generated. """

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx

class GraphEvaluation:
    def __init__(self, uploaded_file=None):
        # Bild laden
        if uploaded_file is not None:
            self.image = cv2.imread(uploaded_file)
            self.output = self.image.copy()
        #self.image = cv2.imread(uploaded_file)
        #self.output = self.image.copy()
    
    def detect_contours(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        edges = cv2.Canny(thresh, 20, 40)
        new_output, nodes = self.detect_nodes(self.output, edges)
        final_img = cv2.cvtColor(new_output, cv2.COLOR_BGR2RGB)
        print(nodes)

        #return final_img
        return final_img, nodes
    
    @classmethod
    def detect_nodes(cls, output, edges):
        
        height, width = output.shape[:2]
        maxRadius = int(min(height, width)*0.044)
        minRadius = int(min(height, width)*0.005)
        # Punkterkennung
        nodes, output2 = cls.circle_detection(edges, minRadius, maxRadius, output, 1, height)
        nodes2, output3 = cls.circle_detection(edges, minRadius*16, maxRadius*3, output2, 2, height)
        
        nodes = [list(node) for node in nodes]
        for node in nodes:
            if nodes2[0][0] -5 <= node[0] <= nodes2[0][0] +5 and nodes2[0][1] -5 <= node[1] <= nodes2[0][1]+5:
                node[2] = True
        nodes = [tuple(node) for node in nodes]
        #nodes.extend(nodes2)
        return output3, nodes
    
    @classmethod
    def circle_detection(cls, edges, minRadius, maxRadius, output, circle_type, height):
        nodes = []
        if circle_type == 2:
            mask = np.zeros(edges.shape, dtype=np.uint8)
            # nur rote Kreise
            mask = cv2.inRange(output, (255, 0, 0), (255, 50, 50))
            edges = cv2.bitwise_not(edges, edges, mask=mask)
        circles = cv2.HoughCircles(image=edges,
                                method=cv2.HOUGH_GRADIENT,
                                dp=1.8,
                                minDist=2*minRadius, 
                                param1=100, 
                                param2=32, 
                                minRadius=minRadius, 
                                maxRadius=maxRadius
                                )
        if circles is not None:
            circles = np.around(circles).astype(np.uint32)
            for (k, (x,y,r)) in enumerate(circles[0, :], start=1):
                cv2.circle(output, (x, y), r, (0, 255, 0), thickness=2) # Kreis
                cv2.circle(output, (x, y), 2, (0, 0, 255), 3) # Mittelpunkt
                print(f"Radius: {r, k}")
                fixed_nodes = False
                nodes.append((x, height-y, fixed_nodes))
                cv2.putText(output, f"{k}", (x+5, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        
        return nodes, output

    def matplot_mechanism(self, data):
        distance_a_b = np.sqrt((data["x"][1] - data["x"][0])**2 + (data["y"][1] - data["y"][0])**2)
        fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
        ax.plot(data["x"], data["y"], "o-")
        for i, txt in enumerate(data["Punkt"]):
            ax.annotate(txt, (data["x"][i], data["y"][i]-2), fontsize=7)
        circle = plt.Circle((data["x"][0], data["y"][0]), distance_a_b, fill=False)
        ax.add_patch(circle)

        return fig

    def plotly_mechanism(self, centerVec, points):
        fig = go.Figure()
        #points[1].update_position(points[1].posX + 1, points[1].posY)
        print(f"Pos points[1]: {points[1].posX}, {points[1].posY}")
        print(f"pos centerVec: {centerVec.posX}, {centerVec.posY}")
        #centerVec.rotate_point(5)

        # Zeichne die Verbindungen
        for point in points[:-1]:  # Der letzte Punkt ist centerVec und hat keine Verbindung
            for connectedPoint in point.connectedPoints:
                fig.add_trace(go.Scatter(x=[point.posX, connectedPoint.posX],
                                        y=[point.posY, connectedPoint.posY],
                                        mode='lines', line=dict(color='blue'),
                                        showlegend=False))
        # Zeichne die Punkte
        for point in points:
            fig.add_trace(go.Scatter(x=[point.posX], y=[point.posY], mode='markers+text',
                                    marker=dict(size=15, color='green'),
                                    text=[point.name],
                                    textposition="top right",
                                    showlegend=False))

        # Zeichne centerVec und seine Verbindung
        fig.add_trace(go.Scatter(x=[centerVec.posX], y=[centerVec.posY], mode='markers+text',
                                marker=dict(size=15, color='green'),
                                text=[centerVec.name],
                                textposition="top right",
                                showlegend=False))

        fig.add_trace(go.Scatter(x=[centerVec.posX, centerVec.rotatingPoint.posX],
                                y=[centerVec.posY, centerVec.rotatingPoint.posY],
                                mode='lines', line=dict(color='blue', dash='dash')))
        
        print(f"centerVec.rotating: {centerVec.rotatingPoint.posX}, {centerVec.rotatingPoint.posY}")
        # Kreiserstellung für centerVec und Nachbar
        distance_center_n = np.sqrt((centerVec.posX - centerVec.rotatingPoint.posX)**2 + (centerVec.posY - centerVec.rotatingPoint.posY)**2)
        angles = np.linspace(0, 2*np.pi, 300)
        circle_x = centerVec.posX + distance_center_n * np.cos(angles)
        circle_y = centerVec.posY + distance_center_n * np.sin(angles)

        circle = go.Scatter(x=circle_x, y=circle_y,
                            mode='lines', line=dict(color='red'), showlegend=False)
        
        fig.add_trace(circle)

        fig.update_layout(title='Punkte und Verbindungen',
                        xaxis_title='X-Achse',
                        yaxis_title='Y-Achse',
                        #xaxis=dict(range=[-50, 50]),
                        #yaxis=dict(range=[-10, 50])
                        yaxis_scaleanchor = "x",
                        yaxis_scaleratio = 1,
                        showlegend=False
                        )

        return fig
    
    def save_image(self):
        plotly_fig = self.plotly_mechanism()
        plotly_fig.write_image("mechanism.png")

    def save_gif(self):
        pass

    def save_video(self):
        pass




#if __name__ == "__main__":
 #   test = FileHandling("test2-bilderkennung.png").detect_contours()
  #  cv2.imshow("test", test)
   # cv2.waitKey(0)
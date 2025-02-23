""" The following code is used to modify the uploaded file so that the mechanism can be generated. """

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
        nodes = []
        height, width = output.shape[:2]
        maxRadius = int(0.02*width)
        minRadius = int(width*0.01)
        # Punkterkennung
        circles = cv2.HoughCircles(image=edges,
                                   method=cv2.HOUGH_GRADIENT,
                                   dp=1.5,
                                   minDist=2*minRadius, 
                                   param1=100, 
                                   param2=30, 
                                   minRadius=minRadius, 
                                   maxRadius=maxRadius
                                   )
        if circles is not None:
            circles = np.around(circles).astype(np.uint32)
            for (k, (x,y,r)) in enumerate(circles[0, :], start=1):
                cv2.circle(output, (x, y), r, (0, 255, 0), thickness=2) # Kreis
                cv2.circle(output, (x, y), 2, (0, 0, 255), 3) # Mittelpunkt
                fixed_nodes = (k == 1 or k == len(circles[0, :]))
                nodes.append((x, y, fixed_nodes))
                cv2.putText(output, f"{k}", (x+5, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        return output, nodes
    
    def matplot_mechanism(self, data):
        distance_a_b = np.sqrt((data["x"][1] - data["x"][0])**2 + (data["y"][1] - data["y"][0])**2)
        fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
        ax.plot(data["x"], data["y"], "o-")
        for i, txt in enumerate(data["Punkt"]):
            ax.annotate(txt, (data["x"][i], data["y"][i]-2), fontsize=7)
        circle = plt.Circle((data["x"][0], data["y"][0]), distance_a_b, fill=False)
        ax.add_patch(circle)

        return fig

    def plotly_mechanism(self, data):
        distance_a_b = np.sqrt((data["x"][1] - data["x"][0])**2 + (data["y"][1] - data["y"][0])**2)

        # Kreis-Koordinaten generieren
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = data["x"][0] + distance_a_b * np.cos(theta)
        circle_y = data["y"][0] + distance_a_b * np.sin(theta)

        # Plotly-Figur erstellen
        fig = go.Figure()

        # Kreis
        fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode="lines"))

        # Linie + Punkte
        fig.add_trace(go.Scatter(x=data["x"],
                                 y=data["y"], 
                                 mode="lines+markers", 
                                 marker=dict(
                                            size=15,
                                            color="red",
                                            symbol="circle"
                                            ), 
                                 line=dict(
                                          color="blue"
                                          )))

        # Layout anpassen
        fig.update_layout(
            title="Plot mit Kreis in Plotly",
            xaxis_title="X-Koordinate",
            yaxis_title="Y-Koordinate",
            xaxis=dict(ticks="inside", showticklabels=True),
            yaxis=dict(ticks="", showticklabels=True), 
            showlegend=False,           
            width=400,
            height=700,
        )

        return fig
    
    def interpret_image(self):
        pass

    def interpret_csv(self):
        pass



#if __name__ == "__main__":
 #   test = FileHandling("test2-bilderkennung.png").detect_contours()
  #  cv2.imshow("test", test)
   # cv2.waitKey(0)
""" The following code is used to modify the uploaded file so that the mechanism can be generated. """

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from CalculationModule import Center, Point, Calculation
import os
import imageio
import kaleido

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
            mask = cv2.inRange(output, (255, 0, 0), (200, 0, 0))
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

    def plotly_mechanism(self, centerVec, points, distances, winkel):
        fig = go.Figure()
        
        print(f"centerVec.rotating: {centerVec.rotatingPoint.posX}, {centerVec.rotatingPoint.posY}")
        # Kreiserstellung für centerVec und Nachbar
        distance_center_n = centerVec.radius
        #np.sqrt((centerVec.posX - centerVec.rotatingPoint.posX)**2 + (centerVec.posY - centerVec.rotatingPoint.posY)**2)
        angles = np.linspace(0, 2*np.pi, 300)
        circle_x = centerVec.posX + distance_center_n * np.cos(angles)
        circle_y = centerVec.posY + distance_center_n * np.sin(angles)

        circle = go.Scatter(x=circle_x, y=circle_y,
                            mode='lines', line=dict(color='red'), showlegend=False)
        
        ani_dict = {point.name: [] for point in points if not point.isFixed}
        trajec_dict = {point.name: [] for point in points if not point.isFixed}
        # Animation
        frames = []
        frames_count = 360

        degree = winkel

        for frame_num in range(frames_count):

            centerVec.rotate_point(1)
            Calculation.output_error(points)
            degree += 1
            print(f"Degree: {degree}")
            Calculation.optimize_all_positions(points, distances)
            
            # Limit der Achsen
            min_x = min(min([point.posX for point in points]), centerVec.posX)-int(distance_center_n*2)
            max_x = max(max([point.posX for point in points]), centerVec.posX)+int(distance_center_n*2)
            min_y = min(min([point.posY for point in points]), centerVec.posY)-int(distance_center_n*2)
            max_y = max(max([point.posY for point in points]), centerVec.posY)+int(distance_center_n*2)

            for point in points:
                if not point.isFixed:
                    ani_dict[point.name].append(point.get_position())
                    trajec_dict[point.name].append((point.get_position(), degree))

            #print(f"Ani_dict: {ani_dict}")
            
            points_trace = go.Scatter(
                x=[p.posX for p in points] + [centerVec.posX],
                y=[p.posY for p in points] + [centerVec.posY],
                mode='markers+text',
                marker=dict(size=15, color='green'),
                text=[p.name for p in points] + [centerVec.name],
                textposition="top right",
                showlegend=False
            )

            connections_traces = []
            for point in points[:-1]:
                for connectedPoint in point.connectedPoints:
                    connections_traces.append(go.Scatter(
                        x=[point.posX, connectedPoint.posX],
                        y=[point.posY, connectedPoint.posY],
                        mode='lines', 
                        line=dict(color='black'),
                        showlegend=False
                        ))
                    
            trajectories = []
            for name, positions in ani_dict.items():
                if len(positions) > 1:
                    trajectories.append(go.Scatter(
                        x=[pos[0] for pos in positions],
                        y=[pos[1] for pos in positions],
                        mode='lines',
                        line=dict(color='blue'),
                        visible='legendonly',
                    ))
                    
            center_line = go.Scatter(
            x=[centerVec.posX, centerVec.rotatingPoint.posX],
            y=[centerVec.posY, centerVec.rotatingPoint.posY],
            mode='lines', 
                line=dict(color='blue', dash='dash'),
                showlegend=False
            )
                
            frames.append(go.Frame(data=[points_trace, center_line] + connections_traces+ trajectories))
        fig = go.Figure(
            data=[points_trace, center_line] + connections_traces + trajectories,
            layout=go.Layout(
                title="Animation",
                xaxis=dict(range=[min_x, max_x]),
                yaxis=dict(range=[min_y, max_y]),
                #showlegend=False,
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    buttons=[dict(label="Play",
                                method="animate",
                                args=[None, dict(frame=dict(duration=30, redraw=True), fromcurrent=True)]),
                            dict(label="Pause",
                                    method="animate",
                                    args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                                ]
                )]
            ),
            frames=frames
        )
        fig.add_trace(circle)        

        return fig, trajec_dict        
        
    
    def save_image(self):
        plotly_fig = self.plotly_mechanism()
        plotly_fig.write_image("mechanism.png")

    def save_gif(self, mechfig, filename="mechanism.gif"):
        os.makedirs("images", exist_ok=True)
        frame_files = []

        for i, frame in enumerate(mechfig.frames):
            frame_name = f"images/frame_{i:03d}.png"
            mechfig.update(frames=[frame])
            mechfig.write_image(frame_name, format="png")
            frame_files.append(frame_name)

        images = [imageio.imread(f) for f in frame_files]
        imageio.mimsave(filename, images, duration=0.03)

        for f in frame_files:
            os.remove(f)
        os.rmdir("images")

        return filename


    def save_video(self):
        pass




#if __name__ == "__main__":
 #   test = FileHandling("test2-bilderkennung.png").detect_contours()
  #  cv2.imshow("test", test)
   # cv2.waitKey(0)
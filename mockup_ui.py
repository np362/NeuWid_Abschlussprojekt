"""This module contains the mockup UI for the application."""

import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import pandas as pd
from graphevaluation import GraphEvaluation
from CalculationModule import Center, Point, Calculation#, update
import matplotlib.animation as animation
import networkx as nx

st.title("Mechanismusanalyse")

if "dataframe" not in st.session_state:
        st.session_state.dataframe = pd.DataFrame({
            "Punkt": ["B", "C", "E", "center"],
            "x": [0, 10, -25, -30],
            "y": [0, 35, 10, 0],
            "Fest": [True, False, False, True]  # True = fest, False = lose
        })

with st.sidebar:
        st.subheader("Mechanismus bearbeiten")

        # Slider für den Kurbelwinkel (0° bis 360°)
        winkel = st.slider("Kurbelwinkel (°)", 0, 360, 0, step=1)
        st.write("Ändere die Skalierung des Graphen:")
        scale = st.number_input("Skalierungsfaktor", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        
        
tab1, tab2, tab3 = st.tabs(["Mechanismus laden", "Mechanismus erstellen", "Mechanismusvorlagen"])

with tab1:
    st.subheader("Mechanismus hochladen")
    uploaded_file = st.file_uploader("Lade hier deinen Mechanismus hoch", type=["csv", "png", "jpg", "jpeg"])
    generate_file = st.button("Mechanismus generieren")
    #if generate_file:
        #with st.progress(text='In progress', value=0):
         #   time.sleep(0.3)
          #  st.success("Der Mechanismus wurde erfolgreich hochgeladen!")
    
    if uploaded_file is not None:
        #st.write("Mechanismus wurde erfolgreich hochgeladen!")
        if uploaded_file.type == "image/jpeg" or uploaded_file.type == "image/png" or uploaded_file.type == "image/jpg":
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name
            image, input_nodes = GraphEvaluation(temp_path).detect_contours()
            #st.write("Knoten:", input_nodes)                
            # Anzeigen des Ergebnisses mit Matplotlib
            plt.figure(figsize=(10, 6))
            plt.imshow(image)
            plt.axis('off')
            

            if generate_file:
                df_rows = []
                center_node = None
                for node in input_nodes:
                    print(f"Node[0]: {node[0]}")
                    if node[2]:
                        center_node = node
                        continue
                    df_rows.append([None, node[0], node[1], node[2]])

                # Anpassung der Nummerierung nach Filterung
                for idx, row in enumerate(df_rows):
                    row[0] = str(idx)

                # Erstelle DataFrame aus gefilterten Daten
                st.session_state.dataframe = pd.DataFrame(df_rows, columns=["Punkt", "x", "y", "Fest"])

                # Stelle sicher, dass der center_node immer an letzter Stelle steht
                if center_node:
                    st.session_state.dataframe = pd.concat([
                        st.session_state.dataframe,
                        pd.DataFrame([["center", center_node[0], center_node[1], center_node[2]]], columns=["Punkt", "x", "y", "Fest"])
                    ], ignore_index=True)
                with st.progress(text='In progress', value=0):
                    time.sleep(0.3)
                    st.success("Der Mechanismus wurde erfolgreich hochgeladen!")

                st.write(st.session_state.dataframe)
            if st.button("Bild anzeigen"):
                st.pyplot(fig=plt)
        
        elif uploaded_file.type == "text/csv":
            st.session_state.dataframe = pd.read_csv(uploaded_file)
            """
                read csv file
            """
            
    elif uploaded_file is None and generate_file:
        st.error("Es wurde keine Datei hochgeladen.")

with tab2:
    st.subheader("Mechanismus erstellen")
    
    if "prev_last_index" not in st.session_state:
        st.session_state.prev_last_index = len(st.session_state.dataframe) - 1

    def update_dataframe():
        edited = st.session_state.data_editor
        if isinstance(edited_data, pd.DataFrame):
            st.session_state.dataframe = edited_data
        else:
            st.session_state.dataframe = pd.DataFrame(edited_data)

        st.session_state.prev_last_index = len(st.session_state.dataframe) - 2

    def swap_rows():
        idx1 = st.session_state.idx1
        idx2 = st.session_state.idx2
        df = st.session_state.dataframe.copy()
        df.iloc[[idx1, idx2]] = df.iloc[[idx2, idx1]].to_numpy() 
        st.session_state.dataframe = df
    
    


    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("Hier kannst du deine Mechanismusdaten bearbeiten:")
        edited_data = st.data_editor(
            st.session_state.dataframe, 
            num_rows="dynamic",
            key="data_editor",
            on_change=update_dataframe
        )

    with col2:
        st.write("Wähle Zeilen zum Tauschen:")
        last_index = st.session_state.prev_last_index
        if last_index < 0:
            last_index = 0

        st.selectbox("Erste Zeile", st.session_state.dataframe.index, key="idx1")
        st.selectbox("Zweite Zeile", st.session_state.dataframe.index, key="idx2")
        st.button("Zwei Zeilen tauschen", on_click=swap_rows)
        
    #edited_data = st.session_state.edited_data

    col3, col4 = st.columns([3, 1])
    
    if st.button("Mechanismus erstellen"):
        #Point.allPoints = []
        #Calculation.AMatrix = np.empty((0,0))
        #Calculation.LVec = np.empty((0,0))
        #Calculation.xVec = np.empty((0,0))
        #Calculation.lVec = np.empty((0,1))
        if Center._instance:
            Center._instance = None

        #with st.spinner(text='In progress'):
            #time.sleep(3)

        points = []
        desiredDistance = []
        distances = []
        for node in edited_data.iterrows():
            if node[0] == edited_data.shape[0]-1:
                centerVec = Center(node[1]["Punkt"], node[1]["x"], node[1]["y"], points[2])
            else:
                points.append(Point(node[1]["Punkt"], node[1]["x"], node[1]["y"], node[1]["Fest"]))
        
        

        for i in range(len(points)-1):
            points[i].add_connection(points[i+1])

        def add_connection():
            idx1 = st.session_state.idx1
            idx2 = st.session_state.idx2
            if idx1 != idx2:
                points[idx1].add_connection(points[idx2])
                desiredDistance.append(Calculation.distance(points[idx1], points[idx2]))
                distances.append((points[idx1], points[idx2], desiredDistance[-1]))
                print(f"Verbindung hinzugefügt: {points[idx1].name} - {points[idx2].name}")
            else:
                st.error("Die Punkte dürfen nicht identisch sein.")

        def remove_connection():
            idx1 = st.session_state.idx1
            idx2 = st.session_state.idx2
            if idx1 != idx2:
                points[idx1].remove_connection(points[idx2])
                desiredDistance.remove(Calculation.distance(points[idx1], points[idx2]))
                distances.remove((points[idx1], points[idx2], desiredDistance[-1]))
            else:
                st.error("Die Punkte dürfen nicht identisch sein.")

        #Calculation.create_xVec(Point.allPoints)
        #Calculation.create_AMatrix(Point.allPoints)
        #Calculation.create_lVec()
        #Calculation.calculate_error()

        print("\n Winkel ändern \n")
        #centerVec.rotate_point(winkel)

        #Calculation.create_xVec(Point.allPoints)
        #Calculation.create_AMatrix(Point.allPoints)
        #Calculation.create_lVec()
        #Calculation.calculate_error()

        #points = [p for p in Point.allPoints]
        #print(points)
        #points.append(centerVec)
        #print(points)

        def degree_of_freedom(points):
            f = 2*(len(points))
            print(f"DOF: {f}")
            for point in points:
                if point.isFixed:
                    f -= 2
                if point == centerVec.rotatingPoint:
                    f -= 2
                if point.connectedPoints:
                    f -= 1
            #f = f - len(Point.connectedPoints)
            print(f"DOF: {f}")
            return f == 0, f
        DOF, f = degree_of_freedom(points)
        with col3:
            if DOF == True:
                st.success('Mechanismus wurde erfolgreich erstellt')
                mechanism_fig = GraphEvaluation().plotly_mechanism(centerVec, points)
                st.plotly_chart(mechanism_fig)
            else:
                st.error('Mechanismus kann nicht erstellt werden')
                if f < 0:
                    st.write("Der Mechanismus ist überbestimmt.")
                    st.write("Entferne mindestens", int(abs(f)), "Verbindungen.")
                elif f > 0:
                    st.write("Der Mechanismus ist unterbestimmt.")
                    st.write("Füge mindestens", int(f), "Verbindungen hinzu.")
        with col4:
            st.write("Erstelle Verbindungen zwischen den Punkten:")
            if st.button("Verbindung hinzufügen", on_click=add_connection):
                pass
            if st.button("Verbindung entfernen", on_click=remove_connection):
                pass
    

with tab3:

    st.subheader("Vorlagen für Mechanismen")
    st.write("Wähle eine Vorlage aus, um einen Mechanismus zu erstellen:")
    colsave1, colsave2, colsave3 = st.columns(3)

    with colsave1:
        # Button zum Speichern der Bewegung als csv-Datei
        if st.button("Bahnkurve als csv-Datei speichern"):
            st.write("Bahnkurve erfolgreich in einer csv-Datei gespeichert!")
    with colsave2:
        # Button zum Speichern der Animation als GIF
        if st.button("GIF Speichern"):
            st.write("Animation als GIF gespeichert!")
    with colsave3:
        # Button zum Speichern der Animation als mp4
        if st.button("Video Speichern"):
            st.write("Animation im MP4-Format gespeichert!")













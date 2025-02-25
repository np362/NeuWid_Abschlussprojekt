"""This module contains the mockup UI for the application."""

import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import pandas as pd
from graphevaluation import GraphEvaluation
from CalculationModule import Center, Point, Calculation, update
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
    if generate_file:
        with st.progress(text='In progress', value=0):
            time.sleep(0.3)
            st.success("Der Mechanismus wurde erfolgreich hochgeladen!")
    
    if uploaded_file is not None:
        st.write("Mechanismus wurde erfolgreich hochgeladen!")
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
            if st.button("Bild anzeigen"):
                st.pyplot(fig=plt)
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

                st.write(st.session_state.dataframe)

with tab2:
    st.subheader("Mechanismus erstellen")
    
    # Beispiel-Daten für die Tabelle
    # Erlaubt dem Nutzer, die Tabelle zu bearbeiten
    st.write("Hier kannst du deine Mechanismusdaten bearbeiten:")
    edited_data = st.data_editor(st.session_state.dataframe, num_rows="dynamic")

    if st.button("Mechanismus erstellen"):
        Point.allPoints = []
        Calculation.AMatrix = np.empty((0,0))
        Calculation.LVec = np.empty((0,0))
        Calculation.xVec = np.empty((0,0))
        Calculation.lVec = np.empty((0,1))

        #with st.spinner(text='In progress'):
            #time.sleep(3)


        for node in edited_data.iterrows():
            if node[0] == edited_data.shape[0]-1:
                centerVec = Center(node[1]["Punkt"], node[1]["x"], node[1]["y"], Point.allPoints[-1])
            else:
                Point(node[1]["Punkt"], node[1]["x"], node[1]["y"], node[1]["Fest"])

        for i in range(len(Point.allPoints)-1):
            Point.allPoints[i].add_connection(Point.allPoints[i+1])

        Calculation.create_xVec(Point.allPoints)
        Calculation.create_AMatrix(Point.allPoints)
        Calculation.create_lVec()
        Calculation.calculate_error()

        print("\n Winkel ändern \n")
        centerVec.rotate_point(winkel)

        Calculation.create_xVec(Point.allPoints)
        Calculation.create_AMatrix(Point.allPoints)
        Calculation.create_lVec()
        Calculation.calculate_error()

        points = [p for p in Point.allPoints]
        print(points)
        points.append(centerVec)
        print(points)


        #fig, ax = plt.subplots()
        #nx.draw(P, pos=nx.get_node_attributes(P, 'pos'), with_labels=True, node_size=300, node_color="skyblue", font_size=10, font_weight="bold")
        #ani = animation.FuncAnimation(fig, update, frames=range(20), fargs=(Point.allPoints[0],), interval=200, repeat=False)
        #st.pyplot(fig)

        mechanism_fig = GraphEvaluation().plotly_mechanism(centerVec, Point.allPoints)
        st.plotly_chart(mechanism_fig)



        if edited_data["Fest"][0] == True:
            st.success('Mechanismus wurde erfolgreich erstellt')
        else:
            st.error('Mechanismus kann nicht erstellt werden')

    

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













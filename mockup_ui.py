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
import ast

st.title("Mechanismusanalyse")

if "dataframe" not in st.session_state:
        st.session_state.dataframe = pd.DataFrame({
            "Punkt": ["B", "C", "D", "A", "E", "center"],
            "x": [0, 10, -25, 25, 5, -30],
            "y": [0, 35, 10, 10, 10, 0],
            "Fest": [True, False, False, False, True, True]  # True = fest, False = lose
        })

if "connections" not in st.session_state:
    st.session_state.connections = [
        (0, 1),
        (1, 2),
        (3, 1),
        (3, 4)
    ]

with st.sidebar:
        st.subheader("Mechanismus bearbeiten")

        # Slider für den Kurbelwinkel (0° bis 360°)
        winkel = st.slider("Inertialwinkel (°)", 0, 360, 0, step=1)

        st.write("Aktuelle Verbindungen:")
        for idx1, idx2 in st.session_state.connections:
            st.write(st.session_state.dataframe.iloc[idx1]["Punkt"], "-", st.session_state.dataframe.iloc[idx2]["Punkt"])
        
        
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

                for (i,node) in enumerate(input_nodes):
                    if i < len(input_nodes)-1:
                        st.session_state.connections.append((i, i+1))

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
            if generate_file:
                st.success("Der Mechanismus wurde erfolgreich hochgeladen!")
                df = pd.read_csv(uploaded_file)
                st.session_state.dataframe = df[["Punkt", "x", "y", "Fest"]].copy()
                df["Verbindungen"] = df["Verbindungen"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else [])
                st.session_state.connections = [(idx, connected_idx) for idx, connected in enumerate(df["Verbindungen"]) for connected_idx in connected]
                st.write(st.session_state.dataframe)

            
    elif uploaded_file is None and generate_file:
        st.error("Es wurde keine Datei hochgeladen.")

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

def add_connection():
    idx1 = st.session_state.idx1
    idx2 = st.session_state.idx2
    if idx1 != idx2:
        points[idx1].add_connection(points[idx2])
        st.session_state.connections.append((idx1, idx2))
        desiredDistance.append(Calculation.distance(points[idx1], points[idx2]))
        distances.append((points[idx1], points[idx2], desiredDistance[-1]))
        print(f"Verbindung hinzugefügt: {points[idx1].name} - {points[idx2].name}")
    else:
        st.error("Die Punkte dürfen nicht identisch sein.")

def remove_connection():
            idx1 = st.session_state.idx1
            idx2 = st.session_state.idx2
            if idx1 != idx2 and points[idx2] in points[idx1].connectedPoints:
                points[idx1].remove_connection(points[idx2])
                st.session_state.connections.remove((idx1, idx2))
                desiredDistance.remove(Calculation.distance(points[idx1], points[idx2]))
                distances.remove((points[idx1], points[idx2], desiredDistance[-1]))
            else:
                st.error("Die Punkte dürfen nicht identisch sein.")

def vorlage(option):
    if option == 1:
        st.session_state.dataframe = vorlage1
        st.session_state.connections = con_vorlage1
    elif option == 2:
        st.session_state.dataframe = vorlage2
        st.session_state.connections = con_vorlage2
    elif option == 3:
        st.session_state.dataframe = vorlage3
        st.session_state.connections = con_vorlage3

def vorlage_2():
    st.session_state.dataframe = vorlage2
    st.session_state.connections = con_vorlage2


with tab2:
    st.subheader("Mechanismus erstellen")
    
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
    save_col1, save_col2 = st.columns([1, 1])
    
    if st.button("Mechanismus erstellen"):
        
        if Center._instance:
            Center._instance = None

        points = []
        desiredDistance = []
        distances = []
        for node in edited_data.iterrows():
            if node[0] == edited_data.shape[0]-1:
                centerVec = Center(node[1]["Punkt"], node[1]["x"], node[1]["y"], points[2])
            else:
                points.append(Point(node[1]["Punkt"], node[1]["x"], node[1]["y"], node[1]["Fest"]))
        
        print(f"connections: {len(st.session_state.connections)}")         
        # for idx1, idx2 in st.session_state.connections:
        #     points[idx1].add_connection(points[idx2])
        #     print(f"Verbindung {points[idx1].name} - {points[idx2].name} hinzugefügt")
        #     desiredDistance.append(Calculation.distance(points[idx1], points[idx2]))
        #     if points[idx1].isFixed:
        #         distances.append((points[idx2], points[idx1], desiredDistance[-1]))
        #     else:
        #         distances.append((points[idx1], points[idx2], desiredDistance[-1]))

        for idx1, idx2 in st.session_state.connections:
            p1, p2 = points[idx1], points[idx2]
            p1.add_connection(p2)
            dist = Calculation.distance(p1, p2)
            desiredDistance.append(dist)
            distances.append((p1, p2, dist))
            distances.append((p2, p1, dist))

        print(f"CenterrotatingPoint: {centerVec.rotatingPoint.name}")
        print(f"Rotating Point: {centerVec.rotatingPoint.name} ({centerVec.rotatingPoint.posX}, {centerVec.rotatingPoint.posY})")
        # Ändert Inertialwinkel
        centerVec.rotate_point(winkel)

        def degree_of_freedom(points):
            f = 2*(len(points))
            print(f"DOF: {f}")
            for point in points:
                if point.isFixed:
                    f -= 2
                if point == centerVec.rotatingPoint:
                    f -= 2
                f -= len(point.connectedPoints)
            #f = f - len(Point.connectedPoints)
            print(f"DOF: {f}")
            return f == 0, f
        DOF, f = degree_of_freedom(points)
        with col3:
            if DOF == True:
                st.success('Mechanismus wurde erfolgreich erstellt')
                mechanism_fig, traject_dict = GraphEvaluation().plotly_mechanism(centerVec, points, distances, winkel)
                st.plotly_chart(mechanism_fig)
                
                data_trajec = []
                for name, values in traject_dict.items():
                    for position, degree in values:
                        data_trajec.append([name, position, degree])
                trajec_df = pd.DataFrame(data_trajec, columns=["Punkt", "x  y", "Winkel"])

                connections_dict = {idx: [] for idx in edited_data.index}
                for idx1, idx2 in st.session_state.connections:
                    connections_dict[idx1].append(idx2)

                with save_col1:
                    # Save the mechanism as a csv file
                    csv_name = "mechanism.csv"
                    edited_and_connectedpoints = edited_data.copy()
                    edited_and_connectedpoints["Verbindungen"] = edited_and_connectedpoints.index.map(lambda idx: connections_dict[idx])

                    st.download_button(label="Mechanismus als CSV speichern",
                                        data=edited_and_connectedpoints.to_csv(index=False),
                                        file_name=csv_name,
                                        mime="text/csv")
                with save_col2:
                    # Save the trajectories as a csv file
                    csv_name = "trajectories.csv"
                    csv_data = trajec_df.to_csv(index=False)
                    st.download_button(label="Trajektorien als CSV speichern",
                                        data=csv_data,
                                        file_name=csv_name,
                                        mime="text/csv")


                # gif_name = "mechanism.gif"
                # GraphEvaluation().save_gif(mechanism_fig, gif_name)

                # with open(gif_name, "rb") as gif_file:
                #     st.download_button(label="Animation als GIF speichern",
                #                         data=gif_file,
                #                         file_name=gif_name,
                #                         mime="image/gif")
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
            st.button("Verbindung hinzufügen", on_click=add_connection)
            st.button("Verbindung entfernen", on_click=remove_connection)
    

with tab3:

    st.subheader("Vorlagen für Mechanismen")
    st.write("Wähle eine Vorlage aus, um einen Mechanismus zu erstellen:")
    colsave1, colsave2, colsave3 = st.columns(3)

    with colsave1:
        vorlage1 = pd.DataFrame({
                "Punkt": ["B", "C", "D", "A", "E", "center"],
                "x": [0, 10, -25, 25, 5, -30],
                "y": [0, 35, 10, 10, 10, 0],
                "Fest": [True, False, False, False, True, True]  # True = fest, False = lose
            })
        con_vorlage1 = [
                (0, 1),
                (1, 2),
                (3, 1),
                (3, 4)
            ]
        # Viergelenkkette Bild einfügen
        st.image("images/Sechsgelenkkette.png", use_container_width=True)

        # Button zum Verwenden der Vorlage 1
        if st.button("Vorlage 1", on_click=lambda: vorlage(1)):
            st.write("Vorlage 1 erfolgreich geladen!")
    with colsave2:
        vorlage2 = pd.DataFrame({
                        "Punkt": ["A", "B", "C", "Center"],
                        "x": [0.0, 10, -25, -30],
                        "y": [0.0, 35, 10, 0],
                        "Fest": [True, False, False, True]  # "Fixiert" = True, andere = False
                    })
        
        con_vorlage2 = [
            (0, 1),  
            (1, 2)
        ]

        st.image("images/Viergelenkkette.png", use_container_width=True)
        # Button zum Verwenden der Vorlage 2
        if st.button("Vorlage 2", on_click=lambda: vorlage(2)):
            st.write("Vorlage 2 erfolgreich geladen!")
    with colsave3:
        vorlage3 = pd.DataFrame({
                        "Punkt": ["A", "B", "C", "D", "E", "F", "G", "Center"],
                        "x": [0.0, 18, 40, -35, -30, -19, 1, 38],
                        "y": [0.0, 50, 25, 20, -19, -84, -39, 8],
                        "Fest": [True, False, False, False, False, False, False, True]  # "Fixiert" = True, andere = False
                    })
        con_vorlage3 = [
            (2, 1),  
            (2, 6),
            (1, 0),
            (3, 1),
            (3, 4),
            (3, 0),
            (4, 5),
            (4, 6),
            (0, 6),
            (5, 6)
        ]
        st.image("images/Strandbeest_vorlage.png", use_container_width=True)
        # Button zum Verwenden der Vorlage 3
        if st.button("Vorlage 3", on_click=lambda:vorlage(3)):
            st.write("Vorlage 3 erfolgreich geladen!")













"""This module contains the mockup UI for the application."""

import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
import pandas as pd
from graphevaluation import GraphEvaluation

st.title("Mechanismusanalyse")

with st.sidebar:
        st.subheader("Mechanismus bearbeiten")

        # Slider für den Kurbelwinkel (0° bis 360°)
        winkel = st.slider("Kurbelwinkel (°)", 0, 360, 0, step=1)
        st.write("Ändere die Skalierung des Graphen:")
        scale = st.number_input("Skalierungsfaktor", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        
        
tab1, tab2, tab3 = st.tabs(["Mechanismus laden", "Mechanismus erstellen", "Mechanismus speichern"])

with tab1:
    st.subheader("Mechanismus hochladen")
    uploaded_file = st.file_uploader("Lade hier deinen Mechanismus hoch", type=["csv", "png", "jpg", "jpeg"])
    generate_file = st.button("Mechanismus generieren")
    if generate_file:
        with st.spinner(text='In progress'):
            time.sleep(3)
            st.success('Done')
    
    if uploaded_file is not None:
        st.write("Mechanismus wurde erfolgreich hochgeladen!")
        if uploaded_file.type == "image/jpeg" or uploaded_file.type == "image/png" or uploaded_file.type == "image/jpg":
            image, input_nodes = GraphEvaluation(uploaded_file.name).detect_contours()
            # Anzeigen des Ergebnisses mit Matplotlib
            plt.figure(figsize=(10, 6))
            plt.imshow(image)
            #plt.axis('off')
            if st.button("Bild anzeigen"):
                st.pyplot(fig=plt)

with tab2:
    st.subheader("Mechanismus erstellen")
    # Beispiel-Daten für die Tabelle
    data = pd.DataFrame({
        "Punkt": ["A", "B", "C", "E"],
        "x": [-30, -25, 10, 0],
        "y": [0, 10, 35, 0],
        "Fest": [True, False, False, True]  # True = fest, False = lose
    })

    # Erlaubt dem Nutzer, die Tabelle zu bearbeiten
    edited_data = st.data_editor(data, num_rows="dynamic")

    if st.button("Mechanismus erstellen"):
        #with st.spinner(text='In progress'):
            #time.sleep(3)
            

        if edited_data["Fest"][0] == True:
            st.success('Mechanismus wurde erfolgreich erstellt')
            
            figure = GraphEvaluation().matplot_mechanism(data)
            st.pyplot(figure)

            plotfigure = GraphEvaluation().plotly_mechanism(data)
            st.plotly_chart(plotfigure)

            if st.button("Animation starten"):
                st.write("Animation wird gestartet!")

        else:
            st.error('Mechanismus kann nicht erstellt werden')

    

with tab3:

    st.subheader("Ergebnisse speichern")
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













"""This module contains the mockup UI for the application."""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd


st.title("Mockup UI")

if st.button("Click me"):
    st.write("You clicked me!")

st.subheader("Graph analysis")
with st.form("Graph_calculation"):
    # Feste Punkte (nicht änderbar)
    fixed_points = ["Punkt A", "Punkt B"]

    # Lose Punkte (mit Checkbox wählbar)
    optional_points = ["Punkt C", "Punkt D", "Punkt E"]
    selected_optional = []

    # Anzeige der festen Punkte
    st.write("Feste Punkte (können nicht abgewählt werden):")
    for point in fixed_points:
        st.write(f"- {point}")  # Fest, keine Checkbox

    # Auswahl für lose Punkte
    st.write("Lose Punkte (zum An- und Abwählen):")
    for point in optional_points:
        if st.checkbox(point, key=point):
            selected_optional.append(point)

    submit_graph = st.form_submit_button("Graph berechnen")
    if submit_graph:
        st.write("Graph wird berechnet...")
        st.write("### Deine Auswahl:")
        st.write("✅ Feste Punkte:", fixed_points)
        st.write("✅ Lose Punkte:", selected_optional)

    # Beispiel-Daten für die Tabelle
    data = pd.DataFrame({
        "x": [0, 1, 2],
        "y": [0, 1, 4],
        "Fest": [False, True, False]  # True = fest, False = lose
    })

    # Erlaubt dem Nutzer, die Tabelle zu bearbeiten
    edited_data = st.data_editor(data, num_rows="dynamic")

    if data["Fest"][0] == True:
        st.write("Feste Punkte:")

# Slider für den Kurbelwinkel (0° bis 360°)
winkel = st.slider("Kurbelwinkel (°)", 0, 360, 0, step=5)

# Längen der Bauteile
l_kurbel = 2    # Länge der Kurbel
l_pleuel = 3    # Länge der Pleuelstange

# Ursprung der Kurbel (feststehender Punkt)
kurbel_fix = np.array([1, 2])

# Position der Kurbelspitze (rotiert um das feste Gelenk)
theta = np.radians(winkel)  # Winkel in Radiant umrechnen
kurbel_spitze = kurbel_fix + l_kurbel * np.array([np.cos(theta), np.sin(theta)])

# Berechnung der Pleuel- und Schieberposition (einfaches Koppelgetriebe-Modell)
pleuel_x = kurbel_spitze[0] + np.sqrt(l_pleuel**2 - (kurbel_spitze[1] - 2)**2)
pleuel = np.array([pleuel_x, 2])  # Y-Koordinate bleibt konstant (vereinfachtes Modell)
schieber = np.array([pleuel[0] + 1, 2])  # Schieber bewegt sich entlang der X-Achse








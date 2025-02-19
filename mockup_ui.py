"""This module contains the mockup UI for the application."""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd


col1, col2 = st.columns([2, 3])
with col1:
    st.title("Mechanismus")
with col2:
    st.title("Meine App")

    st.title("Mockup UI")

st.subheader("Graph analysis")

# Beispiel-Daten für die Tabelle
data = pd.DataFrame({
    "Punkt": ["A", "B", "C", "E"],
    "x": [-30, -25, 10, 0],
    "y": [0, 10, 35, 0],
    "Fest": [True, False, False, True]  # True = fest, False = lose
})

# Erlaubt dem Nutzer, die Tabelle zu bearbeiten
edited_data = st.data_editor(data, num_rows="dynamic")

if data["Fest"][0] == True:
    st.write("Feste Punkte:")

distance_a_b = np.sqrt((data["x"][1] - data["x"][0])**2 + (data["y"][1] - data["y"][0])**2)

# Matplotlib plot in upper right corner
fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
ax.plot(data["x"], data["y"], "o-")
circle = plt.Circle((data["x"][0], data["y"][0]), distance_a_b, fill=False)
ax.add_patch(circle)
for i, txt in enumerate(data["Punkt"]):
    ax.annotate(txt, (data["x"][i], data["y"][i]))

st.pyplot(fig)

# Slider für den Kurbelwinkel (0° bis 360°)
winkel = st.slider("Kurbelwinkel (°)", 0, 360, 0, step=5)

colsave1, colsave2, colsave3 = st.columns(3)

with colsave1:
    # Button zum Speichern der Bewegung als csv-Datei
    if st.button("Als csv-Datei Speichern"):
        st.write("Bewegung in einer csv-Datei gespeichert!")
with colsave2:
    # Button zum Speichern der Animation als GIF
    if st.button("GIF Speichern"):
        st.write("Animation als GIF gespeichert!")
with colsave3:
    # Button zum Speichern der Animation als mp4
    if st.button("Video Speichern"):
        st.write("Animation im MP4-Format gespeichert!")









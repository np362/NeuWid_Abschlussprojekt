"""This module contains the mockup UI for the application."""

import streamlit as st


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
        st.rerun()
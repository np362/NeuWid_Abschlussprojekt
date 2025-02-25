import networkx as nx
import plotly.graph_objects as go
import streamlit as st

# Angenommen, Dein Graph P und die Positionen der Knoten ('pos') sind bereits definiert:
# Beispiel:
P = nx.Graph()
P.add_node(1, pos=(1, 2))
P.add_node(2, pos=(2, 3))
P.add_edge(1, 2)
pos = nx.get_node_attributes(P, 'pos')

# Erstelle einen Trace für die Kanten
edge_x = []
edge_y = []
for edge in P.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=1, color='#888'),
    hoverinfo='none',
    mode='lines'
)

# Erstelle einen Trace für die Knoten
node_x = []
node_y = []
node_text = []
for node in P.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(str(node))

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers+text',
    text=node_text,
    textposition="top center",
    marker=dict(
        color='skyblue',
        size=10,
        line_width=2
    )
)

# Erstelle die Plotly-Figur
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title="Netzwerk-Graph mit Plotly",
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                ))

# Zeige die Figur in Streamlit an
st.plotly_chart(fig)
import numpy as np

# Beispiel: Animation eines beweglichen Punktes
frames = []
for i in range(60):
    # Aktualisiere z.B. die Position des Punktes
    new_x = np.array(node_x) + i * 0.1
    new_y = np.array(node_y) + i * 0.1
    frame = go.Frame(data=[
        go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#888')),
        go.Scatter(x=new_x, y=new_y, mode='markers+text', text=node_text, textposition="top center",
                   marker=dict(color='skyblue', size=10, line_width=2))
    ], name=str(i))
    frames.append(frame)

fig.frames = frames

# Füge ein Play-Button hinzu
fig.update_layout(
    updatemenus=[dict(
        type="buttons",
        buttons=[dict(label="Play",
                      method="animate",
                      args=[None, {"frame": {"duration": 200, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 0}}]
                      )]
    )]
)

st.plotly_chart(fig)

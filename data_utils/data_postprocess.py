import plotly.graph_objects as go
import pandas as pd
import numpy as np
# Read data from a csv

def plot_surface(z_data, title):
    z = z_data
    # print(z.shape)
    sh_0, sh_1 = z.shape
    x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(title=title, autosize=False,
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90))
    fig.show()
# visualization.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go

def plot_contour(X, Y, T, title="Temperature Distribution"):
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X, Y, T, levels=50, cmap=cm.jet)
    plt.colorbar(contour, label="Temperature (°C)")
    plt.quiver(X[::2, ::2], Y[::2, ::2], -np.gradient(T, axis=1)[::2, ::2], 
               -np.gradient(T, axis=0)[::2, ::2], color='white', scale=50)
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.show()

import matplotlib.pyplot as plt

def plot_temperature(T, X, Y, title="Temperature Distribution"):
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X, Y, T, 20, cmap='hot')
    plt.colorbar(cp)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
def plot_3d_surface(X, Y, T):
    fig = go.Figure(data=[go.Surface(z=T, x=X, y=Y)])
    fig.update_layout(title="3D Temperature Distribution", scene=dict(
        xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="Temperature (°C)"))
    fig.show()
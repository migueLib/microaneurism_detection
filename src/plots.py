import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import ListedColormap, rgb2hex, BoundaryNorm 

def plot_color_bar(colors):
    # Transform 0-255 RGB to 0-1 RGB
    colors = [np.asarray(c)/255 for c in colors]
    
    # Create a color map form the provided colors
    cmap = ListedColormap(colors)
       
    # Convert the colors to hex
    hex = np.array([rgb2hex(x) for x in colors])
    
    # Set canvas
    fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    fig.subplots_adjust(bottom=0.25)

    # Calculate bounds
    bounds = range(cmap.N + 1)
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Plot colorbar
    bar = ColorbarBase(ax=ax,
                      cmap=cmap,
                      norm=norm,
                      boundaries=bounds,
                      extend="neither",
                      ticks=None,
                      ticklocation="top",
                      drawedges=False,
                      spacing="uniform",
                      filled=True,
                      orientation="horizontal")
    
    bar.set_ticklabels(hex)
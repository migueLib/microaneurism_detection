# Built-In libraries
import os
import copy
import itertools

# External libraries
import PIL
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, rgb2hex, BoundaryNorm 
from matplotlib.colorbar import ColorbarBase
from tqdm import tqdm

import pandas as pd
import scipy.cluster.hierarchy as shc
import sklearn
from sklearn.cluster import AgglomerativeClustering


class Fundus():
    def __init__(self, source=False, **kwargs):
        # Constructors
        if isinstance(source, str):
            self.im = self._image_from_file(source)

        if isinstance(source, np.ndarray):
            self.im = self._image_from_pixels(source, **kwargs)
            
        if isinstance(source, PIL.Image.Image):
            # TODO: need to re-name the class
            self.im = source

        # Attributes
        self._pixels = np.array(self.get_channels_flattened()).T
        
        self._palette, self._counts = np.unique(self._pixels, axis=0, return_counts=True)
        
        self.cmap = self.get_cmap()

    # Constructors
    @staticmethod
    def _image_from_file(path):
        return Image.open(path, mode="r")

    @staticmethod
    def _image_from_pixels(pixels, **kwargs):
        arr = np.resize(pixels, (kwargs["w"], kwargs["h"], 3)).astype(np.uint8)
        im = Image.fromarray(arr)
        im = im.rotate(90, expand=True)
        im = ImageOps.flip(im)
        return im

    # Get attributes
    @property
    def palette(self):
        return self._palette
    
    @property
    def counts(self):
        return self._counts
    
    @property
    def pixels(self):
        return self._pixels
    
    def get_cmap(self):
        # Transform 0-255 RGB to 0-1 RGB        
        # Create a color map form the provided colors
        return ListedColormap(self._palette/255, N=len(self._palette))
    
    def get_palette_sorted(self):
        return self.palette[np.argsort(self._counts)][::-1]

    def get_channels_flattened(self):
        """
        Returns per color channel a 1-D array of size  (w * h) with all
        the pixels.

        :return: np.array for R, G, B channels respectively.
        """
        r, g, b = np.asarray(self.im).T
        return r.flatten(), g.flatten(), b.flatten()


    # VISUALIZATION
    @staticmethod
    def plot_color_bar(colors):
        # Transform 0-255 RGB to 0-1 RGB
        colors = colors/255
        
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
    
    def plot_palette(self):
        self.plot_color_bar(np.sort(self._palette, axis=0))

    # MODIFICATION FILTERING
    def mask(self, colors, replacement, inplace=False, inverse=False):
        """
        Replaces a list of pixels for a given value
        :param colors: 2-D array of the RGB pixel values for the image.
        :param replacement: 1-D [0-255] RGB array of the color to replace with
        :return: modified 2-D array
        """
        # Empty black canvas if inverse else image
        pixels = np.zeros(self._pixels.shape, dtype=np.uint8) if inverse else self._pixels
        
        # Mask pixels
        for c in colors:
            #pixels[(self._pixels == c).all(axis=1)] = c if inverse else replacement
            pixels[(self._pixels == c).all(axis=1)] = replacement

        # Output 
        #if inplace:
        #    self.im = self._image_from_pixels(pixels, w=self.im.size[0], h=self.im.size[1])
        return pixels
 

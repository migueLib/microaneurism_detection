# Built-In libraries
import os
import copy

# External libraries
import PIL
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, rgb2hex, BoundaryNorm
from matplotlib.colorbar import ColorbarBase

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

        # Attributes
        self.palette, self.counts = self.get_palette()

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
    def get_palette(self):
        pixels = self.get_pixels()
        return np.unique(pixels, axis=0, return_counts=True)

    # Data Transformations
    def as_array(self):
        return np.asarray(self.im)

    def get_channels(self):
        """
        :return np.array [c * x * y] [3 * w * h]
        """
        r, g, b = self.as_array().T
        return r, g, b

    def get_channels_flattened(self):
        """
        Returns per color channel a 1-D array of size  (w * h) with all
        the pixels.

        :return: np.array for R, G, B channels respectively.
        """
        r, g, b = self.get_channels()
        return r.flatten(), g.flatten(), b.flatten()

    def get_pixels(self):
        """
        Returns 2-D array of the RGB pixel values for the whole image.
        size = 3 * (w * h)
        :return: np.array
        """
        return np.array(self.get_channels_flattened()).T

    # VISUALIZATION
    @staticmethod
    def plot_color_bar(colors):
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

    # MODIFICATION FILTERING
    def replace_pixels(self, colors, replacement=(0, 0, 0)):
        """
        Replaces a list of pixels for a given value
        :param colors: 2-D array of the RGB pixel values for the image.
        :param replacement: 1-D [0-255] RGB array of the color to replace with
        :return: modified 2-D array
        """
        pixels = self.get_pixels()
        for c in colors:
            pixels[(self.get_pixels() == c).all(axis=1)] = replacement
        return pixels

    def clear_pixels(self, colors, inplace=False):
        """
        Clears a list of colors from the image the rest are set to black
        :param colors: 2-D array of the RGB pixel values to clear
        :param inplace: Boolean
        :return: modified 2-D array
        """
        pixels = self.get_pixels()
        for c in colors:
            pixels[(self.get_pixels() != c).all(axis=1)] = [0, 0, 0]
        return pixels

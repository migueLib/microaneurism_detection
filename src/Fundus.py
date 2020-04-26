# Built-In libraries
import os
import copy

# External libraries
import PIL
from PIL import Image, ImageFilter
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as shc
import sklearn
from sklearn.cluster import AgglomerativeClustering


class Fundus():
    def __init__(self, source=False, **kwargs):
        # Constructors
        if isinstance(source, str):
            self.I = self._image_from_file(source)

        if isinstance(source, np.ndarray):
            self.I = self._image_from_pixels(source, **kwargs)

        # Attributes
        self.palette, self.counts = self.get_palette()

    # Constructors
    @staticmethod
    def _image_from_file(path):
        return Image.open(path, mode="r")

    @staticmethod
    def _image_from_pixels(pixels, **kwargs):
        arr = np.resize(pixels, kwargs.get("w", None), kwargs.get("w", None))
        return Image.fromarray(arr)

    # Get attributes
    def get_palette(self):
        pixels = self.get_pixels()
        return np.unique(pixels, axis=0, return_counts=True)

    # Data Transformations
    def as_array(self):
        return np.asarray(self.I)

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

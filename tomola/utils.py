"""
Utilities used in the package.
"""
import numpy as np

class Phantom():
    """
    Create phantom distributions for testing or masking purposes.
    """
    def __init__(self, size, center=None):
        """
        TODO: add docstring
        """
        self.size = size

        if center is None:
            self.x0 = size / 2
            self.y0 = size / 2
        else:
            self.x0 = center[0]
            self.y0 = center[1]

        self.x = np.arange(0, self.size, 1, float)
        self.y = self.x[:, np.newaxis]
        self.img = None

    def gaussian(self, fwhm=3):
        """
        TODO: add docstring
        """
        self.img = np.exp(-4*np.log(2)*((self.x-self.x0)**2 + (self.y-self.y0)**2)/fwhm**2)

        return self.img

    def circle(self, cval=0, d=None):
        """
        TODO: add docstring
        """
        if d is None:
            d = self.size

        circle = ((self.x-self.x0)**2 + (self.y-self.y0)**2)/(d/2)**2
        circle[circle == 0] = 1
        circle[circle > 1] = 0
        circle[circle != 0] = 1
        circle[circle == 0] = cval
        self.img = circle
        return self.img

    def data(self, data):
        """
        TODO: add docstring
        """
        self.size = data.shape[0]
        self.img = data

        return self.img

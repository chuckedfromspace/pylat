"""
Utilities used in the package.
"""
import numpy as np
import pickle

# quick save and load
def pkl_dump(path_write, data):
    """
    Dump a data into a pikle file

    Parameters
    ----------
    path_write : dir
        A valid directory in the system
    name_data : str
        The full name of the file to be saved
    data : python object
    """
    with open(path_write, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)

def pkl_load(path_load):
    """
    Load a data into from pikle file

    Parameters
    ----------
    path_load : dir
        A valid directory in the system
    name_data : str
        The full name of the file to be loaded
    """
    with open(path_load, 'rb') as pkl_file:
        data = pickle.load(pkl_file)

    return data


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

    def gaussian(self, fwhm=3, cutoff=0.):
        """
        TODO: add docstring
        """
        self.img = np.exp(-4*np.log(2)*((self.x-self.x0)**2
                                        + (self.y-self.y0)**2)/fwhm**2)
        if cutoff:
            self.img[self.img > cutoff] = cutoff
            self.img /= self.img.max()

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

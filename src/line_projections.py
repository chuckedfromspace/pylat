"""
Create projections for tomographic laser absorption.
"""
import numpy as np

class SinoGram():
    '''
    Create sinograms based on configurations of laser beams.
    '''
    def __init__(self, size):
        """
        Parameters
        ----------
        size : int, optional
            The size of the (square) measurement volume.
        """
        self.sinogram = None
        self.size = size

    def sinogram_parallel(self, num_laser, spacing_adjust=1, ang_laser=None):
        """
        Calculate the sinogram for uniformly-spaced parallel laser beams with specified angles and
        spancings.

        Parameters
        ----------
        num_laser : int
            Number of laser lines used.
        spacing_adjust : float, optional
            Adjust the spacing between the lasers. By default the laser lines are uniformly
            distributed across the size of the measurement volume
        ang_laser : list of floats, optional
            List of angles used to create the parallel laser lines. If None, four angles 0, 45, 90,
            and 135 degrees are used.
        """
        if ang_laser is None:
            ang_laser = np.arange(4)*45

        # Distances between the lasers
        dis_laser = np.linspace(-(self.size//2)*spacing_adjust, (self.size//2)*spacing_adjust,
                                num_laser)

        # Create sinogram
        self.sinogram = np.empty(len(dis_laser)*len(ang_laser), dtype=dict)
        k = 0
        for _dis in dis_laser:
            for _ang in ang_laser:
                self.sinogram[k] = {'d':_dis, 'theta': _ang, 'center': None, 'axis': 'y'}
                k += 1

        return self.sinogram

    def sinogram_custom(self, sinolist, norm_fac=1):
        """
        Calculate the sinogram with custom (random) lines.

        Parameters
        ----------
        sinolist : 2d array
            A 2d array with the shape of Nx2. The two columns are [distance (non-dimensionalized),
            angle]
        norm_fac : float, optional
            The physical dimension of the measurement volume matching the same unit of the provided
            distances. Default is 1 assuming the distances are already normalized.
        """
        sinolist = np.array(sinolist)
        if np.shape(sinolist)[1] != 2:
            raise ValueError('Sinolist has to be a 2xN array!')

        self.sinogram = np.empty(len(sinolist), dtype=dict)
        for i in range(np.shape(sinolist)[0]):
            self.sinogram[i] = {'d':sinolist[i, 0]/norm_fac*self.size, 'theta': sinolist[i, 1],
                                'center': None, 'axis': 'y'}
        return self.sinogram


class LineProj():
    """
    Calculate projections from parallel or arbitrary configurations of laser beams.
    """
    def __init__(self, size=401, img=None):
        """
        Parameters
        ----------
        size : int, optional
            The size of the (square) measurement volume or mesh size. Default is 400 [pixels].
        img : 2d array, optional
            Image data used to calculate absorbance or projection path. It is also used to
            visualize laser lines.
        """
        self.size = size

        if img is None:
            self.img = np.zeros([size, size])
        else:
            self.img = img
        self.proj_abs = None
        self.proj_line = None
        self.mask = None
        self.Lij = None
        self.size_reconst = 40

    def proj(self, d, theta, center=None, axis='y', proj_cal=False, im_alt=True, value=1):
        """
        Calculate projections or absorbance along an arbitrary line through the measurement volume.

        Parameters
        ----------
        d: int or float
            The perpendicular distance from the `center` to the laser line.
        theta: int or float [degrees]
            The angle of laser line relative to the `axis` of a Cartesian coordinate with its origin
            set at the `center` of the image.
        center: None or array_like [i, j], optional
            The point of rotation. By default is the center of the measurement volume (i.e.,
            [N/2, N/2]). To specify a center, the array order is [row(i), column(j)], i and j can be
            larger than N.
        axis: str, 'x' or 'y', optional
            The starting direction of the line for rotation, either vertical ('y') or horizontal
            ('x').
        im_alt: bool, optional
            If true, switch the pixels the arbitrary line crosses to the value specified by the
            parameter 'value' and output the modified image. It is normally set to false for
            calculating absorbance.
        value: float, optional
            The value used to replace the values in the pixels crossed by the laser line.
        proj_cal: bool, optional
            If true, calculate the integral of pixel intensity over pixels and ouput the integral.

        Note
        ----
        The basic idea behind the routine is as follows:
        (1) convert all the pixel locations [i,j] into coordinates with the origin at the `center`
            of the image.
        (2) create a straight line along the `axis`, it has to have more pixels than the image size
            N, but it doesn't have to be more than sqrt(2)*N, as the diagonal is the longest line in
            a square. The line has a offset of `d` from the `axis`.
        (3) clockwise rotation by the amount of `theta` of the straight line
        (4) search through the rotated coordinates of the line and match them with the pixel
            coordinates and then convert them back to pixel locations [i,j].

        Because this is a numerical solution, the finer the original image array, the more accurate
        the results. In certain cases, there might be two matching pixels after Step (4). The first
        one is taken as an approximation.
        """

        n = self.img.shape[0]
        theta = theta / 180 * np.pi

        # create a coordinate for rotation (center at the matrix)
        if center is None:
            center = [(n-1)/2, (n-1)/2] # at the middle of the image

        x = np.linspace(0 - center[1], (n-1) - center[1], n)
        y = np.linspace(center[0] - 0, center[0] - (n-1), n)
        cor_x = np.zeros([n, n])
        cor_y = np.zeros([n, n])
        for i in range(n):
            cor_x[i, :] = x
            cor_y[:, i] = y

        # create a straight line across the center of the coordinate as a start
        step = np.abs(x[1] - x[0])
        l = np.arange(y[0] + 1.5*n*step, y[-1] - 1.5*n*step, -step)
        line = np.zeros([len(l), 2])
        line[:, 0] = d # by default is a vertical line along 'y'
        line[:, 1] = l

        if axis == 'x':
            line[:, [1, 0]] = line[:, [0, 1]] # swap the coordinate to 'x'

        line_r = np.zeros_like(line)
        proj_line = np.empty([0])

        # clockwise rotation of line to line_r
        line_r[:, 0] = np.cos(theta)*line[:, 0] + np.sin(theta)*line[:, 1]
        line_r[:, 1] = -np.sin(theta)*line[:, 0] + np.cos(theta)*line[:, 1]

        # truncate the line to within the image
        line_t = line_r[np.logical_and(line_r[:, 0] >= x[0], line_r[:, 0] <= x[-1])]
        line_t = line_t[np.logical_and(line_t[:, 1] >= y[-1], line_t[:, 1] <= y[0])]

        # assign matrix index to rotated coordinates
        for j in range(len(line_t[:, 0])):
            check = (cor_x - line_t[j, 0])**2 + (cor_y - line_t[j, 1])**2
            index = np.argwhere(check == check.min())

            if proj_cal:
                proj_line = np.append(proj_line, self.img[index[0][0], index[0][1]])
                self.proj_abs = np.trapz(proj_line)
                self.proj_line = proj_line

            if im_alt:
                self.img[index[0][0], index[0][1]] = value

    def conf_vis(self, sinogram, supply_img=False):
        """
        TODO: add docstring
        """
        if supply_img is True:
            self.img = np.zeros([self.size, self.size])
        k = 0
        _tot = len(sinogram)
        _value = 1
        for _sino in sinogram:
            self.proj(d=_sino['d'], theta=_sino['theta'], center=_sino['center'],
                      axis=_sino['axis'], im_alt=True, value=_value, proj_cal=False)

            print('\rProcessing Laser-line #%d of %d lines' % (k+1, _tot), end="")
            k += 1
            # _value += 1

        return self.img

    def absorbance(self, sinogram):
        """
        TODO: add docstring
        """
        _absorb = np.zeros(len(sinogram))
        k = 0
        _tot = len(sinogram)
        for i, _sino in enumerate(sinogram):
            self.proj(d=_sino['d'], theta=_sino['theta'], center=_sino['center'],
                      axis=_sino['axis'], im_alt=False, value=1, proj_cal=1)

            _absorb[i] = self.proj_abs
            print('\rProcessing Laser-line #%d of %d lines' % (k+1, _tot), end="")
            k += 1

        return _absorb

    def path(self, sinogram, size_reconst=40, hamming=False, mask=None):
        """
        Calculate the path length of the laser line in each reconstructed element (cell).

        Parameters
        ----------
        sinogram : list of dict
            Sinogram in the form of a dictionary list, containing 4 keys: 'd', 'theta', 'center' and
            'axis'.
        size_reconst : int, optional
            The size of the reconstructed image. 40 or 50 is often used in the literatur. Default is
            40.
        hamming : bool, optional
            Use a hamming window to emphasize the path lengths in the middle of the image,
            potentially reducing edge effect.
        mask : 2d array, optional
            A mask used to exclude areas for calculation of path lengths.

        Output
        ------
        Lij: 2D array
            Path lengths associated with all the cells (pixels) in the reconstructed image (or
            array).

        """
        # create a zero matrix and assign 1 to all pixels the beam passes
        self.mask = mask
        self.size_reconst = size_reconst

        resize = self.size//size_reconst
        self.Lij = np.zeros([len(sinogram), size_reconst, size_reconst])
        _tot = len(sinogram)
        for k, _sino in enumerate(sinogram):
            self.img = np.zeros([self.size, self.size])
            self.proj(d=_sino['d'], theta=_sino['theta'], center=_sino['center'],
                      axis=_sino['axis'], im_alt=True, value=1, proj_cal=False)
            matProj = self.img

            if mask is not None:
                matProj = matProj * mask

            if hamming is True:
                # find the locations of the pixels==1
                local = np.argwhere(matProj == 1)
                coeff_hamming = np.hamming(len(local))
                for _k, _local in enumerate(local):
                    matProj[_local[0], _local[1]] = matProj[_local[0], _local[1]]*coeff_hamming[_k]

            # count the total number of nonzero pixels in reconstructed matrix
            mat_reconst = np.zeros([size_reconst, size_reconst])
            for j in range(size_reconst):
                for i in range(size_reconst):
                    cell = matProj[i*resize:i*resize + resize, j*resize:j*resize + resize]
                    # path length of ith laser in jth cell
                    mat_reconst[i, j] = np.count_nonzero(cell)

            self.Lij[k, :, :] = mat_reconst

            print('\rProcessing Laser-line #%d of %d lines' % (k+1, _tot), end="")

        return self.Lij

"""
Tomographic reconstruction based on line-of-sight laser absorption measurements
"""
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import minimize, Bounds
from scipy.sparse import spdiags, kron
from numpy.linalg import norm, inv

def dis_lap(nx, ny, alpha=1):
    """
    TODO: add docstring
    """
    ex = np.ones(nx)
    ey = np.ones(ny)
    Lx = spdiags([ex, -2*ex, ex], [-1, 0, 1], nx, nx)
    Ly = spdiags([ey, -2*ey, ey], [-1, 0, 1], ny, ny)
    Ix = spdiags(ex, 0, nx, nx)
    Iy = spdiags(ey, 0, ny, ny)
    return -(kron(Iy, Lx) + kron(Ly, Ix))/4*alpha

def reg_single(N):
    """
    Regulation functions for reconstruction of a single laser line

    Parameters
    ----------
    N: ndarray
        Numpy array containing the reconstruction from the previous iteration

    Returns
    -------
    N: ndarray
        Numpy array containing the regulated reconstruction based on specified
        criteria
    """

    # smooth the field
    N = gaussian_filter(N, sigma=0.5, mode='reflect')
    N[N < 0.] = 0.0
    N[N > 1.] = 1.0

    return N

class TomoReconst():
    """
    Perform tomographic reconstruction based on laser-line configuration
    """
    def __init__(self, proj_obj, absorb):
        """
        TODO: add docstring
        """
        self.size_reconst = proj_obj.size_reconst
        self.Lij = proj_obj.Lij
        self.absorb = absorb

    def SART(self, reg_fcn, relax=0.15, resi=0.01, maxiter=200):
        """
        TODO: add docstring
        """
        # Simultaneous algebraic Reconstruction Technique (ART)
        e = 1 # residual
        k = 0 # count of iteration
        N = np.zeros([self.size_reconst, self.size_reconst]) # initial guess 0
        N_new = np.zeros_like(N)
        while e > resi:
            # Randomly shuffle iteration orders through different projections
            iter_order = np.arange(len(self.absorb))
            np.random.shuffle(iter_order)
            # iterate through the number of laser lines
            for i in iter_order:
                delta_N = ((self.absorb[i]-np.sum(N*self.Lij[i, :, :]))*self.Lij[i, :, :]
                           /np.sum(self.Lij[i, :, :]**2))

                N = N + delta_N * relax

            # calculate residuals
            N_new = N * 1
            N_new = reg_fcn(N_new)
            e = np.sqrt(np.sum((N-N_new)**2))/np.sqrt(np.sum(N**2))
            N = N_new * 1

            # control iterations
            k += 1
            if k > maxiter:
                print('Not converged. Try a larger maxiter or resi.')
                break

        return N

    def SIRT(self, reg_fcn, relax=0.15, resi=0.01, maxiter=200):
        """
        TODO: add docstring
        """
        # Simultaneous Iterative Reconstructive Technique (SIRT)
        e = 1 # residual
        k = 0 # count of iteration
        N = np.zeros([self.size_reconst, self.size_reconst]) # initial guess 0
        delta = np.zeros_like(N)
        N_new = np.zeros_like(N)
        while e > resi:
            for i in range(len(self.absorb)):
                delta_N = ((self.absorb[i]- np.sum(N*self.Lij[i, :, :]))*self.Lij[i, :, :]
                           /np.sum(self.Lij[i, :, :]**2))
                delta += delta_N * relax

            N_new = N + delta
            N_new = reg_fcn(N_new)

            # evaluate residual
            if np.sqrt(np.sum(N**2)) > 0:
                e = np.sqrt(np.sum((N-N_new)**2))/np.sqrt(np.sum(N**2))
            N = N_new * 1

            # control iterations
            k += 1
            if k > maxiter:
                break

        return N

    def Landweber(self, reg_fcn, relax=0.15, resi=0.001, maxiter=200):
        """
        TODO: add docstring
        """
        # Landweber reconstruction
        e = 1 # residual
        k = 0 # count of iteration
        L = self.Lij.reshape([np.shape(self.Lij)[0], self.size_reconst**2])
        # transpose of L as an approximation for L^-1
        L_t = np.transpose(L)
        # estimate relaxation factor
        relax_est = 4 / np.sqrt(np.sum(np.dot(L_t, L)*
                                       np.dot(L_t, L)))
        relax = min(relax, relax_est)
        # first guess
        N = np.dot(L_t, self.absorb)
        N_new = N*1
        while e > resi:
            N_new = N + relax * np.dot(L_t, (self.absorb - np.dot(L, N)))
            M = N_new.reshape([self.size_reconst, self.size_reconst])
            # regulate M
            M = reg_fcn(M)
            N_new = M.reshape([self.size_reconst*self.size_reconst, ])

            e = np.sqrt(np.sum((N-N_new)**2))/(np.sqrt(np.sum(N**2))+1e-6)
            N = N_new * 1

            k += 1
            if k > maxiter:
                print('Not converged. Try a larger maxiter or resi.')
                break

        N = N.reshape([self.size_reconst, self.size_reconst])
        print(e, k)
        return N

    def reg_tikhonov(self, alpha, x0=None, mask=None):
        """
        TODO: add docstring
        """
        # construct Ax=b
        n = self.size_reconst
        mat_A = self.Lij
        mat_A = np.reshape(mat_A, [np.shape(mat_A)[0], n*n])

        mat_Gamma = dis_lap(n, n, alpha=alpha)
        if x0 is None:
            x0 = np.zeros(n*n)

        # define function for minimization
        def func(x):
            return norm(mat_A @ x - self.absorb)**2 + norm(mat_Gamma @ x)**2

        # solve minimization problem
        bnds = Bounds(0, 1)
        # SLSQP or trust-constr
        res = minimize(func, x0, method='trust-constr', bounds=bnds, tol=1e-6,
                       options={'maxiter': 1000, 'disp': True})
        return res

    def reg_tikhonov_direct(self, alpha):
        """
        TODO: add docstring
        """
        # construct Ax=b
        n = self.size_reconst
        mat_A = self.Lij
        mat_A = np.reshape(mat_A, [np.shape(mat_A)[0], n*n])

        mat_Gamma = dis_lap(n, n, alpha=alpha)

        x = inv(mat_A.T @ mat_A + mat_Gamma.T @ mat_Gamma) @ mat_A.T @ self.absorb

        return x

class TomoReconst_2C():
    """
    Perform tomographic reconstruction based on laser-line configuration for 2-color laser
    absorption.
    """
    def __init__(self, proj_obj, absorb1, absorb2, mask=None):
        """
        TODO: add docstring
        """
        self.size_reconst = proj_obj.size_reconst
        self.Lij = proj_obj.Lij
        self.absorb1 = absorb1
        self.absorb2 = absorb2
        self.mask = mask

    def Landweber(self, reg_fcn, relax=0.01, resi=0.01, maxiter=200):
        """
        TODO: add docstring
        """
        # Landweber reconstruction
        e1 = 1 # residual
        e2 = 1
        k = 0 # count of iteration
        L = self.Lij.reshape([np.shape(self.Lij)[0], self.size_reconst**2])
        # transpose of L as an approximation for L^-1
        L_t = np.transpose(L)
        # estimate relaxation factor
        relax_est = 4 / np.sqrt(np.sum(np.dot(L_t, L)*
                                       np.dot(L_t, L)))
        relax = min(relax, relax_est)
        # first guess
        N1 = np.dot(L_t, self.absorb1)
        N2 = np.dot(L_t, self.absorb2)

        if self.mask is not None:
            N1[self.mask.reshape([self.size_reconst*self.size_reconst, ]) == 0] = 0
            N2[self.mask.reshape([self.size_reconst*self.size_reconst, ]) == 0] = 0

        N1_new = N1 * 1
        N2_new = N2 * 1

        while e1 > resi or e2 > resi:
            N1_new = N1 + relax * np.dot(L_t, (self.absorb1 - np.dot(L, N1)))
            M1 = N1_new.reshape([self.size_reconst, self.size_reconst])
            N2_new = N2 + relax * np.dot(L_t, (self.absorb2 - np.dot(L, N2)))
            M2 = N2_new.reshape([self.size_reconst, self.size_reconst])

            # regulate M
            M1, M2, T, mole_frac = reg_fcn(M1, M2, mask=self.mask)
            N1_new = M1.reshape([self.size_reconst*self.size_reconst,])
            N2_new = M2.reshape([self.size_reconst*self.size_reconst,])

            e1 = np.sqrt(np.sum((N1-N1_new)**2))/np.sqrt(np.sum(N1**2))
            e2 = np.sqrt(np.sum((N2-N2_new)**2))/np.sqrt(np.sum(N2**2))
            N1 = N1_new * 1
            N2 = N2_new * 1

            k += 1
            if k > maxiter:
                break

        N1 = N1.reshape([self.size_reconst, self.size_reconst])
        N2 = N2.reshape([self.size_reconst, self.size_reconst])

        return N1, N2, T, mole_frac

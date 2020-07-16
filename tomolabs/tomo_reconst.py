"""
Tomographic reconstruction based on line-of-sight laser absorption measurements
"""
import numpy as np

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

    def Landweber(self, reg_fcn, relax=0.15, resi=0.01, maxiter=200):
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


            e = np.sqrt(np.sum((N-N_new)**2))/np.sqrt(np.sum(N**2))
            N = N_new * 1

            k += 1
            if k > maxiter:
                print('Not converged. Try a larger maxiter or resi.')
                break

        N = N.reshape([self.size_reconst, self.size_reconst])
#         print(e, k)
        return N

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

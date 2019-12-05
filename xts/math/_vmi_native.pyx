
# cython: boundscheck=False, wraparound=False, cdivision=True

cimport numpy
from libc.math cimport fabs, sqrt, atan2, sin, cos, pi
from cython.parallel import prange


def cart2d_to_pol2d(self,
                    numpy.ndarray[double, ndim=2, mode="c"] M,
                    numpy.ndarray[double, ndim=1, mode="c"] Q1,
                    numpy.ndarray[double, ndim=2, mode="c"] Q2):
    
    cdef double[:] R = self.R, A = self.A
    
    cdef int row_len = M.shape[0], col_len = M.shape[1], \
             row_center = self.row_center, col_center = self.col_center, \
             R_len = R.shape[0], A_len = A.shape[0]
    cdef double dr = getattr(self, 'Δr'), da = getattr(self, 'Δα')
    
    cdef int idx_r, idx_a, row_lo, row_up, col_lo, col_up
    cdef double row, col, t, u
    
    for idx_r in range(R_len):
        for idx_a in range(A_len):
            row = -R[idx_r] * cos(A[idx_a]) + row_center
            col = R[idx_r] * sin(A[idx_a]) + col_center

            row_lo = int(row)
            row_up = row_lo + 1
            col_lo = int(col)
            col_up = col_lo + 1

            if row_lo >= 0 and row_up < row_len and col_lo >= 0 and col_up < col_len:
                t = row - row_lo
                u = col - col_lo
                
                Q2[idx_r, idx_a] = (1 - t) * (1 - u) * M[row_lo, col_lo] + \
                                      t    * (1 - u) * M[row_up, col_lo] + \
                                   (1 - t) *    u    * M[row_lo, col_up] + \
                                      t    *    u    * M[row_up, col_up]

                Q1[idx_r] += (Q2[idx_r, idx_a] * da)

        if Q1[idx_r] == 0:
            for idx_a in range(A_len):
                Q2[idx_r, idx_a] = 0
        else:
            for idx_a in range(A_len):
                Q2[idx_r, idx_a] /= Q1[idx_r]

    cdef double norm = 0.0
                
    for idx_r in range(R_len):
        for idx_a in range(A_len):
            norm += Q2[idx_r, idx_a] * Q1[idx_r] * R[idx_r]
            
    for idx_r in range(R_len):
        Q1[idx_r] /= norm * dr * da
                
def pol3d_to_cart2d(self,
                    numpy.ndarray[double, ndim=1, mode="c"] P1,
                    numpy.ndarray[double, ndim=2, mode="c"] P2,
                    numpy.ndarray[double, ndim=2, mode="c"] M,
                    double dz = 1.0, int num_threads=0):
    
    cdef int row_len = M.shape[0], col_len = M.shape[1], \
             row_center = self.row_center, col_center = self.col_center, \
             r_len = self.r_len, a_len = getattr(self, 'α_len')
    cdef double dr = getattr(self, 'Δr'), da = getattr(self, 'Δα')
    
    cdef double r_max_idx = float(r_len)**2 * dr**2
    
    cdef int row, col, r_lo, r_up, a_lo, a_up
    cdef double x, y, z, z_max, r_idx, a_idx, t, u
    
    for row in prange(row_len, nogil=True, num_threads=num_threads):
        y = float(row_center - row)
        
        for col in range(col_len):
            x = float(col - col_center)
            z_max = sqrt(r_max_idx - x**2 - y**2)

            for z from 0.0 <= z < z_max by dz:
                r_idx = sqrt(x**2 + y**2 + z**2) / dr - 1
                
                # TODO: This and the checks below may not actually be
                # needed, but their performance hit is less than 5%.
                if r_idx < 0 or r_idx > r_len-1:
                    continue

                if (x == 0) and (y == 0) and (z == 0):
                    a_idx = 0
                elif x >= 0:
                    a_idx = atan2(sqrt(x**2 + z**2), y) / da
                else:
                    a_idx = (2*pi - atan2(sqrt(x**2 + z**2), y)) / da
                    
                if a_idx < 0 or a_idx > a_len-1:
                    continue

                r_lo = int(r_idx)
                r_up = r_lo + 1
                a_lo = int(a_idx)
                a_up = a_lo + 1

                t = r_idx - r_lo
                u = a_idx - a_lo
                    
                if t == 0:
                    r_up = r_lo
                    
                if u == 0:
                    a_up = a_lo
                    
                M[row, col] += 2 * (
                    (1 - t) * (1 - u) * P1[r_lo] * P2[r_lo, a_lo] + \
                       t    * (1 - u) * P1[r_up] * P2[r_up, a_lo] + \
                    (1 - t) *    u    * P1[r_lo] * P2[r_lo, a_up] + \
                       t    *    u    * P1[r_up] * P2[r_up, a_up]
                )

def pol3d_to_section2d(self,
                       numpy.ndarray[double, ndim=1, mode="c"] P1,
                       numpy.ndarray[double, ndim=2, mode="c"] P2,
                       numpy.ndarray[double, ndim=2, mode="c"] M):
    cdef int row_len = M.shape[0], col_len = M.shape[1], \
             row_center = self.row_center, col_center = self.col_center, \
             r_len = self.r_len, a_len = getattr(self, 'α_len')
    cdef double dr = getattr(self, 'Δr'), da = getattr(self, 'Δα')
    
    cdef int row, col, r_lo, r_up, a_lo, a_up
    cdef double x, y, r_idx, a_idx, t, u
    
    for row in prange(row_len, nogil=True):
        y = float(row_center - row)
        
        for col in range(col_len):
            x = float(col - col_center)

            r_idx = sqrt(x**2 + y**2) / dr - 1

            if x >= 0:
                a_idx = atan2(sqrt(x**2), y) / da
            else:
                a_idx = (2*pi - atan2(fabs(x), y)) / da

            r_lo = int(r_idx)
            r_up = r_lo + 1
            a_lo = int(a_idx)
            a_up = a_lo + 1

            if r_up < r_len and a_up < a_len:
                t = r_idx - r_lo
                u = a_idx - a_lo
                
                M[row, col] += 2 * (
                    (1 - t) * (1 - u) * P1[r_lo] * P2[r_lo, a_lo] + \
                       t    * (1 - u) * P1[r_up] * P2[r_up, a_lo] + \
                    (1 - t) *    u    * P1[r_lo] * P2[r_lo, a_up] + \
                       t    *    u    * P1[r_up] * P2[r_up, a_up]
                )

# cython: boundscheck=False, wraparound=False, cdivision=True

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import numpy as np
cimport numpy
from libc.math cimport fabs, sqrt, atan2, sin, cos, pi
from cython.parallel import prange

from .vmi import IterativeAbelInversion


cdef class data_format:
    cdef int row_center
    cdef int row_len
    cdef int col_center
    cdef int col_len
    cdef int r_len
    cdef int a_len
    cdef double dr
    cdef double da
    cdef int num_threads

    def __cinit__(self, fmt, num_threads):
        self.row_center = fmt.row_center
        self.row_len = fmt.row_len
        self.col_center = fmt.col_center
        self.col_len = fmt.col_len
        self.r_len = fmt.r_len
        self.a_len = getattr(fmt, 'α_len')
        self.dr = getattr(fmt, 'Δr')
        self.da = getattr(fmt, 'Δα')
        self.num_threads = num_threads


class IterativeAbelInversion_native(IterativeAbelInversion):
    def __init__(self, *args, native_num_threads=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.fmt = data_format(self, native_num_threads)

    def cart2d_to_pol2d(self, M):
        Q1 = np.zeros((len(self.R),), dtype=np.float64)
        Q2 = np.zeros((len(self.R), len(self.A)), dtype=np.float64)

        cart2d_to_pol2d(self.fmt, M, Q1, Q2)

        return Q1, Q2

    def pol3d_to_cart2d(self, P1, P2, dz=1.0):
        M = np.zeros((self.row_len, self.col_len), dtype=np.float64)

        pol3d_to_cart2d(self.fmt, P1, P2, M, dz)

        return M

    def pol3d_to_slice2d(self, P1, P2, num_threads=0):
        S = np.zeros((self.row_len, self.col_len), dtype=np.float64)

        pol3d_to_slice2d(self.fmt, P1, P2, S, num_threads)

        return S


cdef cart2d_to_pol2d(data_format fmt,
                     numpy.ndarray[double, ndim=2, mode="c"] M,
                     numpy.ndarray[double, ndim=1, mode="c"] Q1,
                     numpy.ndarray[double, ndim=2, mode="c"] Q2):

    cdef int r_idx, a_idx, row_lo, row_up, col_lo, col_up
    cdef double r, a, row, col, t, u

    for r_idx in range(fmt.r_len):
        for a_idx in range(fmt.a_len):
            r = (r_idx + 1) * fmt.dr
            a = a_idx * fmt.da;

            row = -r * cos(a) + fmt.row_center
            col = r * sin(a) + fmt.col_center

            row_lo = int(row)
            row_up = row_lo + 1
            col_lo = int(col)
            col_up = col_lo + 1

            if row_lo >= 0 and row_up < fmt.row_len and col_lo >= 0 \
                    and col_up < fmt.col_len:
                t = row - row_lo
                u = col - col_lo

                Q2[r_idx, a_idx] = (1 - t) * (1 - u) * M[row_lo, col_lo] + \
                                      t    * (1 - u) * M[row_up, col_lo] + \
                                   (1 - t) *    u    * M[row_lo, col_up] + \
                                      t    *    u    * M[row_up, col_up]

                Q1[r_idx] += (Q2[r_idx, a_idx] * fmt.da)

        if Q1[r_idx] == 0:
            for a_idx in range(fmt.a_len):
                Q2[r_idx, a_idx] = 0
        else:
            for a_idx in range(fmt.a_len):
                Q2[r_idx, a_idx] /= Q1[r_idx]

    cdef double norm = 0.0

    for r_idx in range(fmt.r_len):
        for a_idx in range(fmt.a_len):
            norm += Q2[r_idx, a_idx] * Q1[r_idx] * (r_idx+1) * fmt.dr

    for r_idx in range(fmt.r_len):
        Q1[r_idx] /= norm * fmt.dr * fmt.da


cdef pol3d_to_cart2d(data_format fmt,
                     numpy.ndarray[double, ndim=1, mode="c"] P1,
                     numpy.ndarray[double, ndim=2, mode="c"] P2,
                     numpy.ndarray[double, ndim=2, mode="c"] M,
                     double dz):

    cdef double r_max_idx = float(fmt.r_len)**2 * fmt.dr**2

    cdef int row, col, r_lo, r_up, a_lo, a_up
    cdef double x, y, z, z_max, r_idx, a_idx, t, u

    for row in prange(fmt.row_len, nogil=True, num_threads=fmt.num_threads):
        y = float(fmt.row_center - row)

        for col in range(fmt.col_len):
            x = float(col - fmt.col_center)
            z_max = sqrt(r_max_idx - x**2 - y**2)

            for z from 0.0 <= z < z_max by dz:
                r_idx = sqrt(x**2 + y**2 + z**2) / fmt.dr - 1

                # TODO: This and the checks below may not actually be
                # needed, but their performance hit is less than 5%.
                if r_idx < 0 or r_idx > fmt.r_len-1:
                    continue

                if (x == 0) and (y == 0) and (z == 0):
                    a_idx = 0
                elif x >= 0:
                    a_idx = atan2(sqrt(x**2 + z**2), y) / fmt.da
                else:
                    a_idx = (2*pi - atan2(sqrt(x**2 + z**2), y)) / fmt.da

                if a_idx < 0 or a_idx > fmt.a_len-1:
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


cdef pol3d_to_slice2d(data_format fmt,
                      numpy.ndarray[double, ndim=1, mode="c"] P1,
                      numpy.ndarray[double, ndim=2, mode="c"] P2,
                      numpy.ndarray[double, ndim=2, mode="c"] S,
                      int num_threads):

    cdef int row, col, r_lo, r_up, a_lo, a_up
    cdef double x, y, r_idx, a_idx, t, u

    for row in prange(fmt.row_len, nogil=True, num_threads=num_threads):
        y = float(fmt.row_center - row)

        for col in range(fmt.col_len):
            x = float(col - fmt.col_center)

            r_idx = sqrt(x**2 + y**2) / fmt.dr - 1

            if x >= 0:
                a_idx = atan2(x, y) / fmt.da
            else:
                a_idx = (2*pi - atan2(-x, y)) / fmt.da

            r_lo = int(r_idx)
            r_up = r_lo + 1
            a_lo = int(a_idx)
            a_up = a_lo + 1

            if r_up < fmt.r_len and a_up < fmt.a_len:
                t = r_idx - r_lo
                u = a_idx - a_lo

                S[row, col] += 2 * (
                    (1 - t) * (1 - u) * P1[r_lo] * P2[r_lo, a_lo] + \
                       t    * (1 - u) * P1[r_up] * P2[r_up, a_lo] + \
                    (1 - t) *    u    * P1[r_lo] * P2[r_lo, a_up] + \
                       t    *    u    * P1[r_up] * P2[r_up, a_up]
                )

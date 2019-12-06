# cython: boundscheck=False, wraparound=False, cdivision=True

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


cimport numpy
from libc.math cimport round, floor, ceil

ctypedef fused y_t:
    char
    unsigned char
    short
    unsigned short
    int
    unsigned int
    long
    unsigned long
    long long
    unsigned long long
    float
    double


def stack(numpy.ndarray[y_t, ndim=1, mode="c"] inp,
          numpy.ndarray[y_t, ndim=1, mode="c"] outp,
          double stacking, int stack_begin = 0, int stack_end = -1):

    cdef int inp_len = inp.shape[1], outp_len = min(inp_len, <int>stacking)
    cdef int i, j, inp_idx1, inp_idx2
    cdef double inp_pos
    cdef y_t datum

    if stack_end == -1:
        stack_end = <int>(inp_len / stacking)

    for i in range(outp_len):
        datum = <y_t>0

        for j in range(stack_begin, stack_end):
            inp_pos = i + j * stacking

            inp_idx1 = <int>floor(inp_pos)
            inp_idx2 = <int>ceil(inp_pos)

            datum += <y_t>(inp[inp_idx1] + (inp[inp_idx2] - inp[inp_idx1]) * (inp_pos - <double>inp_idx1))

        outp[i] = datum


def separate(numpy.ndarray[y_t, ndim=1, mode="c"] inp,
             numpy.ndarray[y_t, ndim=2, mode="c"] outp,
             double stacking, int stack_begin = 0, int stack_end = -1):

    cdef int inp_len = inp.shape[0], outp_len = min(inp_len, <int>stacking)
    cdef int i, j, inp_idx1, inp_idx2
    cdef double inp_pos

    if stack_end == -1:
        stack_end = <int>(inp_len / stacking)

    for i in range(outp_len):
        for j in range(stack_begin, stack_end):
            inp_pos = i + j * stacking

            inp_idx1 = <int>floor(inp_pos)
            inp_idx2 = <int>ceil(inp_pos)

            outp[j, i] = <y_t>(inp[inp_idx1] + (inp[inp_idx2] - inp[inp_idx1]) * (inp_pos - <double>inp_idx1))


def cfd_full(numpy.ndarray[y_t, ndim=1, mode='c'] signal,
             numpy.ndarray[double, ndim=1, mode='c'] cfd_signal,
             numpy.ndarray[double, ndim=1, mode='c'] edge_indices,
             numpy.ndarray[y_t, ndim=1, mode='c'] edge_heights,
             int threshold, double fraction, int delay, int width,
             double zero):

    cdef int i, j, k, signal_len = cfd_signal.shape[0], edge_idx = 0, \
             next_edge = -1
    cdef y_t height

    for i in range(0, delay + 1):
        cfd_signal[i] = signal[i] - fraction * signal[0]

    if threshold > 0:
        for i in range(delay, signal_len - 1):
            j = i + 1
            cfd_signal[j] = signal[j] - fraction * signal[j - delay]

            if signal[i] > threshold and (cfd_signal[i] > zero) and \
                    (cfd_signal[j] < zero) and i > next_edge:
                height = signal[i - delay]

                for k in range(j - delay, i + delay):
                    if signal[k] > height:
                        height = signal[k]

                edge_indices[edge_idx] = i + (cfd_signal[i] - zero) \
                    / (cfd_signal[i] - cfd_signal[j])
                edge_heights[edge_idx] = height

                next_edge = i + width
                edge_idx += 1
    else:
        for i in range(delay, signal_len - 1):
            j = i + 1
            cfd_signal[j] = signal[j] - fraction * signal[j - delay]

            if signal[i] < threshold and (cfd_signal[i] < zero) and \
                    (cfd_signal[j] > zero) and i > next_edge:
                height = signal[i - delay]

                for k in range(j - delay, i + delay):
                    if signal[k] < height:
                        height = signal[k]

                edge_indices[edge_idx] = i + (zero - cfd_signal[i]) \
                    / (cfd_signal[j] - cfd_signal[i])
                edge_heights[edge_idx] = height

                next_edge = i + width
                edge_idx += 1

    return edge_idx


def cfd_fast(numpy.ndarray[y_t, ndim=1, mode='c'] signal,
             numpy.ndarray[double, ndim=1, mode='c'] edge_indices,
             numpy.ndarray[y_t, ndim=1, mode='c'] edge_heights,
             int threshold, double fraction, int delay, int width,
             double zero):
    if threshold > 0:
        return cfd_fast_pos(signal, edge_indices, edge_heights, threshold,
                            fraction, delay, width, zero)
    else:
        return cfd_fast_neg(signal, edge_indices, edge_heights, threshold,
                            fraction, delay, width, zero)


def cfd_fast_pos(numpy.ndarray[y_t, ndim=1, mode='c'] signal,
                 numpy.ndarray[double, ndim=1, mode='c'] edge_indices,
                 numpy.ndarray[y_t, ndim=1, mode='c'] edge_heights,
                 int threshold, double fraction, int delay, int width,
                 double zero):

    cdef int i, j, k, edge_idx = 0, next_edge = -1
    cdef double cfd_i, cfd_j
    cdef y_t height

    for i in range(delay, signal.shape[0] - 1):
        if signal[i] <= threshold:
            continue

        j = i + 1

        cfd_i = signal[i] - fraction * signal[i - delay]
        cfd_j = signal[j] - fraction * signal[j - delay]

        if cfd_i > zero and cfd_j < zero and i > next_edge:
            height = signal[i - delay]

            for k in range(j - delay, i + delay):
                if signal[k] > height:
                    height = signal[k]

            edge_indices[edge_idx] = i + (cfd_i - zero) / (cfd_i - cfd_j)
            edge_heights[edge_idx] = height

            next_edge = i + width
            edge_idx += 1

    return edge_idx


def cfd_fast_neg(numpy.ndarray[y_t, ndim=1, mode='c'] signal,
                 numpy.ndarray[double, ndim=1, mode='c'] edge_indices,
                 numpy.ndarray[y_t, ndim=1, mode='c'] edge_heights,
                 int threshold, double fraction, int delay, int width,
                 double zero):

    cdef int i, j, k, edge_idx = 0, next_edge = -1
    cdef double cfd_i, cfd_j
    cdef y_t height

    for i in range(delay, signal.shape[0] - 1):
        if signal[i] >= threshold:
            continue

        j = i + 1

        cfd_i = signal[i] - fraction * signal[i - delay]
        cfd_j = signal[j] - fraction * signal[j - delay]

        if cfd_i < zero and cfd_j > zero and i > next_edge:
            height = signal[i - delay]

            for k in range(j - delay, i + delay):
                if signal[k] < height:
                    height = signal[k]

            edge_indices[edge_idx] = i + (zero - cfd_i) / (cfd_j - cfd_i)
            edge_heights[edge_idx] = height

            next_edge = i + width
            edge_idx += 1

    return edge_idx

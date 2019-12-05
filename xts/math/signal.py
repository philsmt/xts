
import numpy as np
from scipy.ndimage.interpolation import shift

try:
    from ._signal_native import stack, separate, cfd_full, cfd_fast, cfd_fast_neg, cfd_fast_pos
except ImportError:
    pass


def shift_center(data):
    """Shifts 1D data to center of array assuming a periodic boundary.


    Args:
        data (array_like): 1D array to be centered.

    Returns:
        ndarray: centered 1D array.

    """

    N = len(data)
    w = np.exp((-2j * np.pi * np.arange(N)) / N)
    ang = np.angle(np.dot(data, w).conj())

    if ang < 0:
        ang += 2*np.pi

    centerpos = int((N * ang) / (2 * np.pi))

    return np.roll(data, int(N / 2) - centerpos)


def get_bin_func(bin_factor):
    """Generate a binning lambda.

    Args:
        bin_factor (int): Bin factor > 1.

    Returns:
        callabel (ndarray) -> ndarray: binning function.

    """

    return eval('lambda x: (x + {0})[::{1}]'.format(
        ' + '.join(['np.roll(x, {0})'.format(i)
                    for i in range(-1, -bin_factor, -1)]), bin_factor
    ))


def cfd(signal, threshold=150, fraction=1.0, delay=25, width=0, zero=0.0):
    """Constant fraction discrimator.

    Preliminary implementation in pure python for debugging purposes.

    Args:
        signal (array_like): Input signal.
        threshold (float): Trigger threshold.
        fraction (float, optional): Fraction value.
        delay (int or float, optional): Delay value.
        width (int, optional): Output pulse width, any further edges
            within the width of an edge are suppressed.

    Returns:
        edge_indice (array_like): Peak position for all edges.
        edge_heights (array_like): Peak height for all edges.
        cfd_signal (array_like): superimposed CFD signal.
    """

    # Delay and invert
    if isinstance(delay, int) or delay.is_integer():
        # Optimization for integer, MUCH faster than the scipy call

        cfd_signal = np.array(signal, dtype=np.float64)
        cfd_signal[delay:] -= fraction * signal[:-delay]
        cfd_signal[:delay] -= fraction * signal[1]
    else:
        cfd_signal = signal - shift(signal * fraction, delay, mode="nearest")

    # Edge detection
    try:
        if threshold > 0:
            edges_bool = cfd_signal < zero
            above_threshold = signal[:-1] > threshold

        else:
            edges_bool = cfd_signal > zero
            above_threshold = signal[:-1] < threshold

        zero_crossings = edges_bool[1:] != edges_bool[:-1]

        edge_indices = (zero_crossings & above_threshold).nonzero()[0]

    except IndexError:
        edge_indices = np.empty((0,), dtype=np.float32 if interpolate
                                            else np.int32)

    if len(edge_indices) > 0:
        # Width suppression
        if width > 0:
            edge_diffs = edge_indices[1:] - edge_indices[:-1]
            edge_mask = np.ones((len(edge_diffs) + 1,), dtype=bool)

            try:
                for edge_idx in (edge_diffs < width).nonzero()[0]:
                    # Re-check the condition, as we modify the array
                    if edge_diffs[edge_idx] < width:
                        edge_mask[edge_idx+1] = False
                        edge_diffs[edge_idx+1] += edge_diffs[edge_idx]

            except IndexError:
                # Might be thrown on the last found edge
                pass

            edge_indices = edge_indices[edge_mask]

        new_indices = np.empty_like(edge_indices, dtype=np.float32)

        # Interpolation
        if threshold > 0:
            for i in range(len(edge_indices)):
                x = edge_indices[i]
                new_indices[i] = x + cfd_signal[x]/(cfd_signal[x]
                    - cfd_signal[x+1])
        else:
            for i in range(len(edge_indices)):
                x = edge_indices[i]
                new_indices[i] = x - cfd_signal[x]/(cfd_signal[x+1]
                    - cfd_signal[x])

        edge_indices = new_indices

        # Peak heights
        edge_heights = np.empty_like(edge_indices, dtype=signal.dtype)

        if threshold > 0:
            for i in range(len(edge_indices)):
                edge_idx = int(edge_indices[i])
                try:
                    edge_heights[i] = \
                        signal[edge_idx-delay:edge_idx+delay].max()
                except ValueError:
                    edge_heights[i] = 0
        else:
            for i in range(len(edge_indices)):
                edge_idx = int(edge_indices[i])
                try:
                    edge_heights[i] = \
                        signal[edge_idx-delay:edge_idx+delay].min()
                except ValueError:
                    edge_heights[i] = 0

    else:
        edge_heights = np.empty((0,), dtype=signal.dtype)

    return edge_indices, edge_heights, cfd_signal


def _is_unique_peak(peak_indices, peak_xs, peak_ys, n_peaks, cur_x, cur_y, search_size):
    for i, peak_idx in zip(range(n_peaks), peak_indices):
        if abs(cur_x - peak_xs[peak_idx]) <= search_size \
                and abs(cur_y - peak_ys[peak_idx]) <= search_size:
            return False

    return True


def _sort_peaks(data, peak_xs, peak_ys, all_indices, search_size, peak_indices, peak_coms_x, peak_coms_y):
    n_peaks = 0

    xy_shape = (6, 6)
    xy_vals = np.arange(2*search_size)

    for cur_idx in all_indices:
        cur_x = peak_xs[cur_idx]
        cur_y = peak_ys[cur_idx]

        if is_unique_peak_py(peak_indices, peak_xs, peak_ys, n_peaks, cur_x, cur_y, search_size):
            peak_indices[n_peaks] = cur_idx

            peak_region = data[cur_y-search_size:cur_y+search_size,
                               cur_x-search_size:cur_x+search_size]

            if peak_region.shape != xy_shape:
                x_vals = np.arange(peak_region.shape[1])
                y_vals = np.arange(peak_region.shape[0])
            else:
                x_vals, y_vals = xy_vals, xy_vals

            area = peak_region.sum()

            if area == 0:
                peak_coms_x[n_peaks] = cur_x
                peak_coms_y[n_peaks] = cur_y
            else:
                peak_coms_x[n_peaks] = (cur_x - search_size + (peak_region.sum(0) * x_vals).sum() / area)
                peak_coms_y[n_peaks] = (cur_y - search_size + (peak_region.sum(1) * y_vals).sum() / area)

            n_peaks += 1

    return n_peaks


def find_2d_hits(data, threshold, search_size, sort_peaks=None):
    # Get all points above the threshold
    peak_ys, peak_xs = np.where(data > threshold)

    # Peak indices sorted from the largest number down
    peak_values = data[peak_ys, peak_xs]
    all_indices = sorted(range(len(peak_xs)), key=lambda x: peak_values[x], reverse=True)

    if sort_peaks is None:
        try:
            sort_peaks = sort_peaks_ntv
        except NameError:
            sort_peaks = _sort_peaks

    if sort_peaks is sort_peaks_ntv:
        # Ensure they are contiguous for C
        data = np.ascontiguousarray(data)
        all_indices = np.ascontiguousarray(all_indices)
        peak_xs = np.ascontiguousarray(peak_xs)
        peak_ys = np.ascontiguousarray(peak_ys)

    # Holds the indices of all unique peaks
    peak_indices = np.empty_like(all_indices)

    # Holds the proper center of mass of all unique peaks
    peak_coms_x = np.empty((peak_xs.shape[0],), dtype=np.float64)
    peak_coms_y = np.empty((peak_ys.shape[0],), dtype=np.float64)

    n_peaks = sort_peaks(data, peak_xs, peak_ys, all_indices, search_size, peak_indices, peak_coms_x, peak_coms_y)

    if n_peaks == 0:
        return None

    peak_indices = peak_indices[:n_peaks]
    peak_coms_x = peak_coms_x[:n_peaks]
    peak_coms_y = peak_coms_y[:n_peaks]

    return peak_xs[peak_indices], peak_ys[peak_indices], peak_coms_x, peak_coms_y


def covariance(X, Y=None, r_corr=1.0, r_uncorr=1.0):
    """Covariance matrix of one or two variables.

    Args:
        X (array_like): First random variable of shape NxM.
        Y (array_like, optional): Second random variable of shape NxM'
            or None for the autocovariance matrix of X.
        r_corr (float, optional): Ratio for the correlated component,
            i.e. <XY>.
        r_uncorr (float, optional): Ratio for the uncorrelated
            component, i.e. <X><Y>.

    Returns:
        ndarray: Covariance matrix of shape MxM'.

    """

    if Y is None:
        Y = X

    nx = X.shape[0]
    ny = Y.shape[0]

    Xsum = (1/nx) * X.sum(axis=0).reshape(1, -1)
    Ysum = (1/ny) * Y.sum(axis=0).reshape(1, -1)

    res_corr   = (r_corr/nx) * np.dot(X.T,    Y)
    res_uncorr =  r_uncorr   * np.dot(Xsum.T, Ysum)

    return res_corr - res_uncorr


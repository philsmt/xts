
import inspect

import numpy as np
import scipy.optimize
import scipy.sparse


def find_nearest_index(array, value):
    return (np.abs(array - value)).argmin()


def find_nearest_value(array, value):
    return array[find_nearest_index(array, value)]


def identity(x):
    """Identity function.

    Args:
        x (Any): Any argument.

    Returns:
        Any: The passed argument.

    """

    return x


def project_2d_hits(xy, shape, dtype=None):
    """Project an array of xy positions onto a matrix, i.e. 2d binning.

    Args:
        xy (array_like): Input array of shape NxM, M ≥ 2.
        shape (tuple): Output matrix shape.
        dtype (data-type): Output matrix data type.

    Returns:
        ndarray: The output matrix

    """

    if dtype is None:
        dtype = xy.dtype

    return scipy.sparse.coo_matrix(
        (np.ones_like(xy[:, 0]), (xy[:, 1], xy[:, 0])),
        shape=shape, dtype=dtype
    ).toarray()


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


def radial_mask(shape, center=None, min_radius=0, max_radius=float('inf')):
    """Generate a radial mask.

    Args:
        shape (tuple): Mask shape as (rows, columns).
        center (tuple, optional): Center point, (rows//2, columns//2) is
            used if omitted.
        min_radius (float, optional): Lower radial bound, 0 if omitted.
        max_radius (float, otpional): Upper radial bound, inf if omitted.

    Returns:
        (ndarray): Boolean mask array.

    """

    if center is None:
        center = (shape[0] // 2, shape[1] // 2)

    grid_y, grid_x = np.ogrid[:shape[0], :shape[1]]
    radial_grid = (grid_x - center[1])**2 + (grid_y - center[0])**2
    mask = (min_radius**2 <= radial_grid) * (radial_grid <= max_radius**2)

    return mask


def apply_radial_mask(data, value=None, **kwargs):
    """Apply a radial mask.

    A shorthand function for radial_mask(), a radial mask is generated and
    immediately applied to either clear all points outside the mask or
    set those inside the mask to a specific value.

    Any additiona keyword arguments are passed to radial_mask().

    Args:
        data (ndarray): Data array to mask
        value (float or None, optional): Set all points inside the mask to
            this value or clear all remaining points if None, defaults to
            None.

    Returns:
        (ndarray) Masked data array

    """

    mask = radial_mask(data.shape, **kwargs)

    new_data = data.copy()

    if value is None:
        new_data[~mask] = 0
    else:
        new_data[mask] = value

    return new_data


def theta_mask(shape, center=None, min_angle=None, max_angle=None,
               left=False, right=False, top=False, bottom=False,
               topleft=False, topright=False,
               bottomleft=False, bottomright=False):
    """Generate an angular mask.

    The mask may either be specified by a minimum and/or maximum angl
    or any combination of image quadrants (left, right, top, bottom,
    topleft, topright, bottomleft, bottomright). If either explicit
    angle value is specified, the quadrants are ignored.

    The zero angle is located on the upper end of the x axis, i.e. at
    higher columns and then moves CW in positive direction and
    correspondingly CCW in negative direction.

    Args:
        shape (tuple): Mask shape as (rows, columns).
        center (tuple, optional): Center point, (rows//2, columns//2)
            is used if omitted.
        min_angle (float, optional): Lower angle bound, -π if omitted
        max_angle (float, optional): Upper angle bound, π if omitted
        left (bool, optional): left quadrant, ignored if either
            min_angle and/or max_angle are specified
        right (bool, optional): top quadrant, ignored if either
            min_angle and/or max_angle are specified
        top (bool, optional): top quadrant, ignored if either min_angle
            and/or max_angle are specified
        bottom (bool, optional): bottom quadrant, ignored if either
            min_angle and/or max_angle are specified
        topleft (bool, optional): topleft quadrant, ignored if either
            min_angle and/or max_angle are specified
        topright (bool, optional): topright quadrant, ignored if either
            min_angle and/or max_angle are specified
        bottomleft (bool, optional): bottomleft quadrant, ignored if
            either min_angle and/or max_angle are specified
        bottomright (bool, optional): bottomright quadrant, ignored if
            either min_angle and/or max_angle are specified

    Returns:
        (ndarray): Boolean mask array

    """

    if center is None:
        center = (shape[0] // 2, shape[1] // 2)

    grid_y, grid_x = np.ogrid[:shape[0], :shape[1]]
    theta_grid     = np.arctan2(grid_y - center[0], grid_x - center[1])

    if min_angle is not None or max_angle is not None:
        min_angle = min_angle if min_angle is not None else -np.pi
        max_angle = max_angle if max_angle is not None else np.pi

        mask = (min_angle <= theta_grid) * (theta_grid <= max_angle)

    else:
        mask = np.zeros_like(theta_grid, dtype=bool)

        if left:
            mask |= ~((-3*np.pi/4 <= theta_grid) * (theta_grid <= 3*np.pi/4))

        if right:
            mask |=   (-np.pi/4   <= theta_grid) * (theta_grid <= np.pi/4)

        if top:
            mask |=   (-3*np.pi/4 <= theta_grid) * (theta_grid <= -np.pi/4)

        if bottom:
            mask |=   (np.pi/4    <= theta_grid) * (theta_grid <= 3*np.pi/4)

        if topleft:
            mask |=   (-np.pi     <= theta_grid) * (theta_grid <= -np.pi/2)

        if topright:
            mask |=   (-np.pi/2   <= theta_grid) * (theta_grid <= 0)

        if bottomright:
            mask |=   (0          <= theta_grid) * (theta_grid <= np.pi/2)

        if bottomleft:
            mask |=   (np.pi/2    <= theta_grid) * (theta_grid <= np.pi)

    return mask


def apply_theta_mask(data, value=None, **kwargs):
    """Apply an angular mask.

    A shorthand function for theta_mask(), a theta mask is generated
    and immediately applied to either clear all points outside the mask
    or set those inside the mask to a specific value.

    Any additiona keyword arguments are passed to theta_mask().

    Args:
        data (ndarray): Data array to mask
        value (float or None, optional): Set all points inside the mask
            to this value or clear all remaining points if None,
            defaults to None.

    Returns:
        (ndarray) Masked data array

    """

    mask = theta_mask(data.shape, **kwargs)

    new_data = data.copy()

    if value is None:
        new_data[~mask] = 0
    else:
        new_data[mask] = value

    return new_data


def gaussian(x, y0, A, μ, σ):
    """Normalized gaussian distribution, i.e. normal distribution.

    Args:
        x (array_like, real): Function argument
        y0 (real): Vertical offset
        A (real): Amplitude
        μ (real): Expected value
        σ (real): Standard deviation

    Returns:
        Function value(s)

    """

    return y0 + (A / (σ * np.sqrt(2*np.pi))) \
        * np.exp( -(x - μ)**2 / (2 * σ**2) )


def gaussian2d(x, y, z0, A, μ_x, μ_y, σ_x, σ_y):
    """Normalized 2d gaussian distribution.

    Args:
        x, y (array_like, real): Function arguments
        z0 (real): Vertical offset
        μ_x, μ_y (real): Expected values
        σ_x, σ_y (real): Standard deviations

    Returns:
        Function value(s)

    """

    return z0 + (A/(σ_x * σ_y * 2*np.pi)) \
        * np.exp( - (x[None, :] - μ_x)**2 / (2 * σ_x**2)
                  - (y[:, None] - μ_y)**2 / (2 * σ_y**2) )


def lorentzian(x, y0, A, x0, γ):
    """Normalized lorentzian distribution, i.e. cauchy distribution.

    Args:
        x (array_like, real): Function argument
        y0 (real): Vertical offset
        A (real): Amplitude
        x0 (real): Location parameter
        γ (real): Scale parameter

   Returns:
       Function value

    """

    return y0 + (A/(np.pi*γ)) * (γ**2/((x - x0)**2 + γ**2))


def angular_dist(ϑ, σ, β, P1, λ):
    """Electron angular distribution with Stokes parameter.

    Args:
        ϑ (array_like, real): Emission angle
        σ (real): Integrated cross section
        β (real): Beta parameter -1 < β < 2
        P1 (real): First Stoke's parameter
        λ (real): Tilt angle

    Returns:
        Differential cross section dσ/dϑ

    """

    return σ/(4*np.pi) * (1 + (β/4) * (1 + 3 * P1 * np.cos(2*(ϑ - λ))))


def curve_fit(f, x, y, p0=None, bounds=None, label=None, verbose=True, *args,
              **kwargs):
    """Perform a least-square fit based on scipy.optimize.curve_fit

    This function extends scipy.optimize.curve_fit by the option to
    keep any initial parameter in p0 constant if casted to a str, which
    in turn is assumed to be a float. The resulting coefficients and
    covariances still contain these parameters and all entries in the
    latter vanish.

    In addition, the standard deviation and total R² based on the
    residuals and covariance is returned. Furthermore, the verbose
    option enables a summary printed to stdout.

    Args:
        f, x, y, bounds: Same as scipy.optimize.curve_fit
        p0 (array_like): Initial guess for the parameters. If
            specified, it must contain exactly one entry for each
            parameter. A string value is treated as a constant float
            value and not optimized in the least-square fit, but still
            present in the result (with zero error).
        verbose (bool, optional): If True, the fit result and the
            uncertainties are printed to stdout for each parameter.

        Any further arguments are passed on to scipy.optimize.curve_fit

    Returns:
        (ndarray) coefficient results
        (ndarray) coefficient uncertainties (standard deviation)
        (float) R²
        (ndarray) covariance matrix

    """

    param_names = inspect.getargspec(f).args
    n_p = len(param_names) - 1

    if n_p < 2:
        raise ValueError('f receives one or less arguments')

    if p0 is None:
        p0 = [1.0] * n_p

    # Ensure that bounds is specified for each parameter
    if bounds is not None:
        try:
            len(bounds[0])
        except TypeError:
            bounds = ([bounds[0]] * n_p, [bounds[1]] * n_p)

    else:
        bounds = ([-np.inf] * n_p, [np.inf] * n_p)

    # Find the indices of fixed parameters
    fixed_idx = []
    for i in range(len(p0)):
        if isinstance(p0[i], str):
            fixed_idx.append(i)

    n_fixed = len(fixed_idx)
    n_unfixed = n_p - n_fixed

    if n_fixed > 0:
        # New list with only the unfixed parameters
        new_p0 = []
        new_bounds = ([], [])

        # Code strings for each argument when f is called inside our
        # lambda, which will either be 'a'+str(i) or the fixed value.
        argument_strs = []

        i_unfixed = 0
        for i in range(len(p0)):
            if i in fixed_idx:
                argument_strs.append(str(float(p0[i])))
            else:
                new_p0.append(p0[i])
                new_bounds[0].append(bounds[0][i])
                new_bounds[1].append(bounds[1][i])
                argument_strs.append('a' + str(i_unfixed))
                i_unfixed += 1

        # Create our new lambda that substitutes the fixed parameters
        new_f = eval('lambda x, {0}: f(x, {1})'.format(
            ','.join(['a' + str(i) for i in range(n_unfixed)]),
            ','.join(argument_strs)
        ), dict(f=f))
    else:
        new_f = f
        new_p0 = p0
        new_bounds = bounds

    if verbose:
        print('{0}Fitting \'{1}\' to {2} values...'.format(
            label + ': ' if label is not None else '', f.__name__, len(x)
        ))

    # Perform the fit!
    coeffs, cov = scipy.optimize.curve_fit(new_f, x, y, p0=new_p0,
                                           bounds=new_bounds, *args, **kwargs)

    # Fix up the returned arrays
    for fixed_idx in fixed_idx:
        # Reinsert the fixed parameter into the result
        coeffs = np.insert(coeffs, fixed_idx, float(p0[fixed_idx]))

        # And a row/column of zeros into the covariance matrix
        cov = np.insert(cov, fixed_idx, np.zeros((cov.shape[0],)), axis=0)
        cov = np.insert(cov, fixed_idx, np.zeros((cov.shape[0],)), axis=1)

    # Calculate the R² and standard deviations
    r_squared = 1 - (np.sum((y - f(x, *coeffs))**2)
                     / np.sum((y - y.mean())**2))
    errors = np.sqrt(np.diag(cov))

    if verbose:
        print('R²: {0:.5f}'.format(r_squared))

        for i in range(n_p):
            print('{0}: {1:.3f} +/- {2:.3f}'.format(param_names[i+1],
                                                    coeffs[i], errors[i]))

    return coeffs, errors, r_squared, cov


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

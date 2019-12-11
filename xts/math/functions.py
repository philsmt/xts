
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import inspect

import numpy as np
import scipy.optimize


def identity(x):
    """Identity function.

    Args:
        x (Any): Any argument.

    Returns:
        Any: The passed argument.

    """

    return x


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

    return y0 + (A / (σ * np.sqrt(2*np.pi))) * np.exp(-(x - μ)**2 / (2 * σ**2))


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
        * np.exp(- (x[None, :] - μ_x)**2 / (2 * σ_x**2)
                 - (y[:, None] - μ_y)**2 / (2 * σ_y**2))


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

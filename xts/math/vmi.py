
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


from importlib import import_module

import numpy as np

from .image import apply_radial_mask


class DirectAbelInversion:
    pass


class IterativeAbelInversion:
    _impl = {}

    @classmethod
    def load_impl(cls, name=None):
        if name == 'numpy':
            return IterativeAbelInversion

        elif name is None:
            for name in ['opencl', 'native', 'numpy']:
                impl = cls.load_impl(name)

                if impl is not None:
                    return impl

            return IterativeAbelInversion

        try:
            impl = cls._impl[name]
        except KeyError:
            try:
                mod = import_module(f'{__package__}._vmi_{name}')
                impl = getattr(mod, f'IterativeAbelInversion_{name}')
            except ImportError:
                return IterativeAbelInversion
            except AttributeError:
                return IterativeAbelInversion
            else:
                cls._impl[name] = impl

        return impl

    def __init__(self, M_shape, row_center, col_center, r_max, r_len=1024,
                 α_len=1024, backend=None, **kwargs):
        if isinstance(M_shape, np.ndarray):
            M_shape = M_shape.shape

        self.row_len = M_shape[0]
        self.col_len = M_shape[1]
        self.row_center = row_center
        self.col_center = col_center
        self.r_max = r_max
        self.r_len = r_len
        self.α_max = 2*np.pi
        self.α_len = α_len

        self.R = (np.arange(1, r_len+1) * r_max) / r_len
        self.A = np.linspace(0, 2*np.pi, α_len)
        self.Δr = self.R[1] - self.R[0]
        self.Δα = self.A[1] - self.A[0]

    def integrate_pol2d(self, Q1, Q2):
        return (Q1 * Q2.sum(axis=1) * self.R).sum() * self.Δr * self.Δα

    def integrate_pol3d(self, P1, P2):
        return 2 * np.pi * self.Δr * self.Δα * (
            self.R**2 * P1 *
            (P2 * np.sin(self.A)[None, :]).sum(axis=1)
        ).sum()

    def polar_grid(self):
        return np.meshgrid(self.A, self.R)

    def cart2d_to_pol2d(self, M):
        Q2 = np.zeros((len(self.R), len(self.A)), dtype=np.float64)

        row = -self.R[:, None] * np.cos(self.A)[None, :] + self.row_center
        col = self.R[:, None] * np.sin(self.A)[None, :] + self.col_center

        row_lo = row.astype(int)
        row_hi = row_lo + 1
        col_lo = col.astype(int)
        col_hi = col_lo + 1

        in_bounds = (row_lo >= 0) & (row_hi < M.shape[0]) & (col_lo >= 0) & \
            (col_hi < M.shape[1])

        row_lo = row_lo[in_bounds]
        row_hi = row_hi[in_bounds]
        col_lo = col_lo[in_bounds]
        col_hi = col_hi[in_bounds]

        t = row[in_bounds] - row_lo
        u = col[in_bounds] - col_lo

        tc = 1 - t
        uc = 1 - u

        Q2[in_bounds] = (
            M[row_lo, col_lo] * tc * uc +
            M[row_hi, col_lo] * t * uc +
            M[row_lo, col_hi] * tc * u +
            M[row_hi, col_hi] * t * u
        )

        Q1 = Q2.sum(axis=1) * self.Δα

        Q2 /= Q1[:, None]
        Q1 /= (Q2.sum(axis=1) * Q1 * self.R).sum() * self.Δr * self.Δα

        return Q1, Q2

    def pol2d_to_pol3d(self, Q1, Q2):
        P1 = Q1 / (2 * np.pi * self.R)
        P2 = Q2.copy()

        return P1, P2

    def pol3d_to_pol2d(self, P1, P2):
        Q1 = P1 * 2 * np.pi * self.R
        Q2 = P2.copy()

        return Q1, Q2

    def norm_pol3d(self, P1, P2):
        sin_vals = np.empty_like(self.A)

        mask = self.A <= np.pi
        sin_vals[mask] = np.sin(self.A[mask])
        sin_vals[~mask] = np.sin(2*np.pi - self.A[~mask])

        norm_values = (P2 * sin_vals[None, :]).sum(axis=1) \
            * np.pi * self.Δα
        nonzero_norms = norm_values != 0.0

        P1 /= (norm_values * P1 * self.R**2).sum() * self.Δr
        P2[~nonzero_norms, :] = 0
        P2[nonzero_norms, :] /= norm_values[nonzero_norms, None]

    def norm_cart2d(self, M):
        M /= M.sum()

    def pol3d_to_cart2d(self, P1, P2, Z=None):
        if Z is None:
            Z = np.arange(0.0, self.R[-1], 1.0)

        M = np.zeros((self.row_len, self.col_len), dtype=np.float64)

        X, Y = np.meshgrid(
            (np.arange(self.col_len) - self.col_center).astype(np.float64),
            (self.row_center - np.arange(self.row_len)).astype(np.float64)
        )

        XY_squared = X**2 + Y**2

        for z in Z:
            in_bounds = (self.r_max**2 - z**2 - 1) >= XY_squared

            Xi = X[in_bounds]
            Yi = Y[in_bounds]

            r = np.sqrt(Xi**2 + Yi**2 + z**2)
            r_idx = r / self.Δr - 1

            α = np.arctan2(np.sqrt(Xi**2 + z**2), Yi)
            α[Xi < 0] = 2*np.pi - α[Xi < 0]
            α_idx = α / self.Δα

            r_lo = r_idx.astype(int)
            r_up = r_lo + 1
            α_lo = α_idx.astype(int)
            α_up = α_lo + 1

            t = r_idx - r_lo
            u = α_idx - α_lo

            tc = 1 - t
            uc = 1 - u

            M[in_bounds] += 2 * (
                P1[r_lo] * P2[r_lo, α_lo] * tc * uc +
                P1[r_up] * P2[r_up, α_lo] * t * uc +
                P1[r_lo] * P2[r_lo, α_up] * tc * u +
                P1[r_up] * P2[r_up, α_up] * t * u
            )

        return M

    def pol3d_to_slice2d(self, P1, P2):
        return self.pol3d_to_cart2d(P1, P2, Z=[0.0])

    def M_leastsq_err(M_exp, M_cal):
        # Actually a static method!

        return np.sqrt(((M_cal - M_exp)**2).sum()
                       / (M_cal.shape[0] * M_cal.shape[1]))

    def __call__(self, *args, **kwargs):
        n_steps, err = self.iterate(*args, **kwargs)

        return self.P1[:, None] * self.P2

    def iterate(self, M_exp, zero_negative_values=True,
                min_steps=1, max_steps=50, rel_err_tol=1e-3, adaptive=False,
                c1=2.0, c2=1.0, cost_func=M_leastsq_err,
                c1_min=0.5, c1_max=2.0, c2_min=0.25, c2_max=1.0,
                adaptive_inc=1.08, adaptive_dec=0.7, **kwargs):
        '''
        Perform the iterative optimization.

        The optimization process may be stopped either by a maximum
        number of iterations or a minimum tolerance to obtain for the
        relative decrease of the optimization error.

        Arguments:
            M_exp (array_like): Experimental data.
            zero_negative_values (bool): Whether to set any negative
                values to zero (True) or offset by the minimal value,
                so that min(M_exp) == 0.
            min_steps (int): Minimum number of iterations to perform.
            max_steps (int or None): Maximum number of iterations to
                perform or None for no limit.
            rel_err_tol (float or None): Convergence tolerance for the
                relative decrease of the optimization error or None for
                no limit.
            adaptive (bool): Whether to enable adaptive feedback that
                changes the c1, c2 feedback parameters from its initial
                values during the iteration process.
            c1 (float): Radial feedback parameter.
            c2 (float): Angular feedback parameter.
            cost_func (Callable): Cost function to determine the
                convergence quality of the optimization, which takes
                M_exp, M_cal as its arguments and must return a
                non-negative float.
            c1_min, c1_max, c2_min, c2_max (float): Boundaries for the
                feedback parameters with enabled adaptive feedback.
            adaptive_inc, adaptive_dec (float): Relative change of
                feedback parameters with enabled adaptive feedback.

        Returns:
            (int) The number of performed iterations.
            (ndarray) Optimization errors for each iteration step.

        '''

        if max_steps is not None and min_steps >= max_steps:
            raise ValueError('min_steps must be smaller than max_steps')

        if max_steps is None and rel_err_tol is None:
            raise ValueError('no iteration limit')

        assert M_exp.shape == (self.row_len, self.col_len), \
               'Incompatible shape of M_exp'  # noqa: E127

        self.M_exp = M_exp.copy()

        if zero_negative_values:
            self.M_exp[self.M_exp < 0] = 0
        elif self.M_exp.min() < 0:
            self.M_exp -= self.M_exp.min()

        self.norm_cart2d(self.M_exp)

        self.Q1_exp, self.Q2_exp = self.cart2d_to_pol2d(M_exp)

        try:
            del self.Q1_cal
            del self.Q2_cal
        except AttributeError:
            pass

        i = 0
        err_vals = []

        while True:
            if i > 1:
                best_Q1_cal = self.Q1_cal.copy()
                best_Q2_cal = self.Q2_cal.copy()
                best_P1 = self.P1.copy()
                best_P2 = self.P2.copy()

            self.step(c1, c2)

            i += 1
            err_vals.append(cost_func(self.M_exp, self.M_cal))

            if max_steps is not None and i >= max_steps:
                break

            try:
                rel_err_change = (err_vals[-2] - err_vals[-1]) / err_vals[-2]
            except IndexError:
                pass
            else:
                if rel_err_tol is not None and rel_err_change > 0 \
                        and rel_err_change < rel_err_tol and i >= min_steps:
                    break

            # In adaptive mode, always return the result with the lowest error

            if adaptive and i > 2:
                if rel_err_change > 0:
                    # If the error decreased, increase the feedback
                    # coefficients (up to their max)
                    c1 = min(c1_max, c1 * adaptive_inc)
                    c2 = min(c2_max, c2 * adaptive_inc)

                else:
                    # If the error increased, decrease the feedback
                    # coefficients (down to their min)
                    c1 = max(c1_min, c1 * adaptive_dec)
                    c2 = max(c2_min, c2 * adaptive_dec)

                    # Reset the results
                    self.Q1_cal = best_Q1_cal
                    self.Q2_cal = best_Q2_cal
                    self.P1 = best_P1
                    self.P2 = best_P2

        return i, np.array(err_vals)

    def step(self, c1, c2):
        try:
            D1, D2 = self.pol2d_to_pol3d(self.Q1_cal - self.Q1_exp,
                                         self.Q2_cal - self.Q2_exp)
            self.P1 = self.P1 - c1 * D1
            self.P2 = self.P2 - c2 * D2
        except AttributeError:
            # First iteration will raise an AttributeError for Q1_cal, Q2_cal
            self.P1, self.P2 = self.pol2d_to_pol3d(self.Q1_exp, self.Q2_exp)

        nonpositive_r = self.P1 <= 0
        self.P1[nonpositive_r] = 0
        self.P2[nonpositive_r, :] = 0

        negative_α = self.P2 < 0
        self.P2[negative_α] = 0

        self.norm_pol3d(self.P1, self.P2)

        self.M_cal = self.pol3d_to_cart2d(self.P1, self.P2)
        self.norm_cart2d(self.M_cal)

        assert np.isfinite(self.M_cal).all(), 'M_cal is not finite'

        self.Q1_cal, self.Q2_cal = self.cart2d_to_pol2d(self.M_cal)


def invert_abel(data, row_center, col_center, r_max=None,
                method=None, impl=None,
                radial_clip=False, return_full=False, **kwargs):
    if method is None:
        method = IterativeAbelInversion
    elif isinstance(method, str):
        try:
            method = {
                'iterative': IterativeAbelInversion
            }[method]
        except KeyError:
            raise ValueError(f'invalid inversion method \'{method}\'') \
                from None

    if r_max is None:
        r_max = min([row_center, col_center, data.shape[0] - row_center,
                     data.shape[1] - col_center])

    if radial_clip:
        data = apply_radial_mask(data, center=(row_center, col_center),
                                 max_radius=r_max)

    cls = method.load_impl(impl)
    inv = cls(data.shape, row_center, col_center, r_max, **kwargs)
    dist3d = inv(data, **kwargs)

    return (dist3d, inv) if return_full else dist3d


def project_polar():
    pass


def integrate_radius():
    pass


def integrate_angle():
    pass

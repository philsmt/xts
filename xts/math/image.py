
import numpy as np
import scipy.sparse


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

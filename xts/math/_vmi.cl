
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */


$typedef_data_format


kernel void cart2d_to_pol2d_project(constant read_only data_format* fmt,
                                    global read_only double* M,
                                    global write_only double* Q1,
                                    global write_only double* Q2,
                                    global write_only double* norm) {

    const int r_idx = get_global_id(0);
    const double r = (r_idx+1) * fmt->dr;

    if(r_idx >= fmt->r_len) {
        return;
    }

    int a_idx, row_lo, row_up, col_lo, col_up;
    double a, row, col, t, u;
    double Q1_result = 0.0, Q2_result;

    for(a_idx = 0; a_idx < fmt->a_len; a_idx++) {
        a = a_idx * fmt->da;

        row = -r * cos(a) + fmt->row_center;
        col = r * sin(a) + fmt->col_center;

        row_lo = (int)row;
        row_up = row_lo + 1;
        col_lo = (int)col;
        col_up = col_lo + 1;

        if(row_lo < 0 || row_up >= fmt->row_len || col_lo < 0 || col_up >= fmt->col_len) {
            return;
        }

        t = row - row_lo;
        u = col - col_lo;

        // Moving M to texture memory may speed this up even more!
        Q2_result = (
            (1 - t) * (1 - u) * M[row_lo * fmt->col_len + col_lo] +
               t    * (1 - u) * M[row_up * fmt->col_len + col_lo] +
            (1 - t) *    u    * M[row_lo * fmt->col_len + col_up] +
               t    *    u    * M[row_up * fmt->col_len + col_up]
        );

        Q2[r_idx * fmt->a_len + a_idx] = Q2_result;
        Q1_result += Q2_result * fmt->da;
    }

    Q1[r_idx] = Q1_result;

    if(Q1_result != 0.0) {
        for(a_idx = 0; a_idx < fmt->a_len; a_idx++) {
            Q2[r_idx * fmt->a_len + a_idx] /= Q1_result;
        }
    }
    else {
        // This branch actually needed?
        for(a_idx = 0; a_idx < fmt->a_len; a_idx++) {
            Q2[r_idx * fmt->a_len + a_idx] = 0;
        }
    }

    norm[r_idx] = Q1_result * r * fmt->dr;
}


kernel void pol2d_to_pol3d_init(constant read_only data_format* fmt,
                                global read_only double* Q1_exp,
                                global read_only double* Q2_exp,
                                global write_only double* P1,
                                global write_only double* P2) {

    const int r_idx = get_global_id(0);
    int v_idx;

    const double P1_val = Q1_exp[r_idx] / (2 * M_PI * (r_idx+1) * fmt->dr);

    if(P1_val >= 0.0) {
        P1[r_idx] = P1_val;

        for(v_idx = r_idx * fmt->a_len; v_idx < (r_idx+1) * fmt->a_len; v_idx++) {
            P2[v_idx] = max(Q2_exp[v_idx], 0.0);
        }
    }
    else {
        P1[r_idx] = 0.0;

        for(v_idx = r_idx * fmt->a_len; v_idx < (r_idx+1) * fmt->a_len; v_idx++) {
            P2[v_idx] = 0.0;
        }
    }
}


kernel void pol2d_to_pol3d_step(constant read_only data_format* fmt,
                                global read_only double* Q1_exp,
                                global read_only double* Q2_exp,
                                global read_only double* Q1_cal,
                                global read_only double* Q2_cal,
                                global read_write double* P1,
                                global read_write double* P2,
                                double c1, double c2) {

    const int r_idx = get_global_id(0);
    int v_idx;

    const double P1_val = P1[r_idx] - c1 * ((Q1_cal[r_idx] - Q1_exp[r_idx]) / (2 * M_PI * (r_idx+1) * fmt->dr));

    if(P1_val >= 0.0) {
        P1[r_idx] = P1_val;

        for(v_idx = r_idx * fmt->a_len; v_idx < (r_idx+1) * fmt->a_len; v_idx++) {
            P2[v_idx] = max(P2[v_idx] - c2 * (Q2_cal[v_idx] - Q2_exp[v_idx]), 0.0);
        }
    }
    else {
        P1[r_idx] = 0.0;

        for(v_idx = r_idx * fmt->a_len; v_idx < (r_idx+1) * fmt->a_len; v_idx++) {
            P2[v_idx] = 0.0;
        }
    }
}


kernel void norm_pol3d_angular(constant read_only data_format* fmt,
                               global read_only double* P1,
                               global read_write double* P2,
                               global write_only double* radial_norm) {
    const int r_idx = get_global_id(0);
    int a_idx, v_idx = r_idx * fmt->a_len;

    const double r = (r_idx+1) * fmt->dr;
    double a;

    double angular_norm = 0.0;

    for(a_idx = 0; a_idx < fmt->a_len; a_idx++, v_idx++) {
        a = a_idx * fmt->da;

        angular_norm += P2[v_idx]
            * sin( (a <= M_PI) ? a : 2 * M_PI - a )
            * M_PI * fmt->da;
    }

    if(angular_norm != 0.0) {
        for(v_idx = r_idx * fmt->a_len; v_idx < (r_idx+1) * fmt->a_len; v_idx++) {
            P2[v_idx] /= angular_norm;
        }
    }
    else {
        for(v_idx = r_idx * fmt->a_len; v_idx < (r_idx+1) * fmt->a_len; v_idx++) {
            P2[v_idx] = 0.0;
        }
    }

    radial_norm[r_idx] = angular_norm * P1[r_idx] * r * r * fmt->dr;
}


double pol3d_to_cart2d_point(constant read_only data_format* fmt,
                             global read_only double* P1,
                             global read_only double* P2,
                             double x, double y, double z) {

    const double r_idx = sqrt(x*x + y*y + z*z) / fmt->dr - 1;
    double a_idx;

    if(x == 0 && y == 0 && z == 0) {
        a_idx = 0;
    }
    else if(x >= 0) {
        a_idx = atan2(sqrt(x*x + z*z), y) / fmt->da;
    }
    else {
        a_idx = (2 * M_PI - atan2(sqrt(x*x + z*z), y)) / fmt->da;
    }

    if(r_idx < 0 || r_idx > fmt->r_len - 1) {
        return 0.0;
    }

    const int r_lo = convert_int(r_idx),
              a_lo = convert_int(a_idx);

    int r_up = r_lo + 1,
        a_up = a_lo + 1;

    const double t = r_idx - r_lo,
                 u = a_idx - a_lo;

    if(t == 0.0) {
        r_up = r_lo;
    }

    if(u == 0.0) {
        a_up = a_lo;
    }

    // Moving P1, P2 to texture memory may speed this up even more!
    return 2 * (
        (1 - t) * (1 - u) * P1[r_lo] * P2[r_lo * fmt->a_len + a_lo] +
           t    * (1 - u) * P1[r_up] * P2[r_up * fmt->a_len + a_lo] +
        (1 - t) *    u    * P1[r_lo] * P2[r_lo * fmt->a_len + a_up] +
           t    *    u    * P1[r_up] * P2[r_up * fmt->a_len + a_up]
    );
}


kernel void pol3d_to_cart2d(constant read_only data_format* fmt,
                            global read_only double* P1,
                            global read_only double* P2,
                            global write_only double* M) {
    const int row = get_global_id(0),
              col = get_global_id(1);

    const double x = convert_double(col - fmt->col_center),
                 y = convert_double(fmt->row_center - row),
                 z_max = sqrt(convert_double(fmt->r_len * fmt->r_len) * fmt->dr * fmt->dr - x*x - y*y);

    double z, result = 0.0;

    for(z = 0.0; z < z_max; z += 1.0) {
        result += pol3d_to_cart2d_point(fmt, P1, P2, x, y, z);
    }

    M[row * fmt->col_len + col] = result;
}


kernel void pol3d_to_slice2d(constant read_only data_format* fmt,
                             global read_only double* P1,
                             global read_only double* P2,
                             global write_only double* S) {
    const int row = get_global_id(0),
              col = get_global_id(1);

    const double x = convert_double(col - fmt->col_center),
                 y = convert_double(fmt->row_center - row);

    S[row * fmt->col_len + col] = pol3d_to_cart2d_point(fmt, P1, P2, x, y, 0.0);
}

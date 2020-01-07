
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


from pathlib import Path
from string import Template

import numpy as np
import pyopencl as cl
import pyopencl.tools

from .vmi import IterativeAbelInversion


with open(Path(__file__).parent / '_vmi.cl', 'r') as fp:
    cl_code = Template(fp.read())


class IterativeAbelInversion_opencl(IterativeAbelInversion):
    def __init__(self, *args, opencl_devices=None, opencl_interactive=False,
                 opencl_fast_step=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.opencl_fast_step = opencl_fast_step

        if opencl_devices is None or opencl_interactive:
            self.ctx = cl.create_some_context(interactive=opencl_interactive)
        else:
            self.ctx = cl.Context(devices=opencl_devices)

        self.queue = cl.CommandQueue(self.ctx)

        data_format_values = {
            'row_center': np.int32(self.row_center),
            'row_len': np.int32(self.row_len),
            'col_center': np.int32(self.col_center),
            'col_len': np.int32(self.col_len),
            'r_len': np.int32(self.r_len),
            'a_len': np.int32(self.α_len),
            'dr': np.float64(self.Δr),
            'da': np.float64(self.Δα)
        }

        data_format, _data_format_cl = cl.tools.match_dtype_to_c_struct(
            self.ctx.devices[0], 'data_format',
            np.dtype([(k, v.dtype) for k, v in data_format_values.items()]),
            context=self.ctx
        )
        data_format = cl.tools.get_or_register_dtype('data_format',
                                                     data_format)

        _fmt = np.ndarray(1, dtype=data_format)

        for key, value in data_format_values.items():
            _fmt[key] = value

        final_code = cl_code.substitute(typedef_data_format=_data_format_cl)
        self.prg = cl.Program(self.ctx, final_code).build()

        self.buf_fmt = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY |
                                 cl.mem_flags.COPY_HOST_PTR, hostbuf=_fmt)
        self.buf_M = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                               size=8 * self.row_len * self.col_len)
        self.buf_Q1e = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                 size=8 * self.r_len)
        self.buf_Q1c = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                 size=8 * self.r_len)
        self.buf_P1 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                size=8 * self.r_len)
        self.buf_Q2e = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                 size=8 * self.r_len * self.α_len)
        self.buf_Q2c = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                 size=8 * self.r_len * self.α_len)
        self.buf_P2 = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE,
                                size=8 * self.r_len * self.α_len)
        self.buf_norm = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY,
                                  size=8 * self.r_len)
        self.buf_S = self.buf_M  # Just use the same buffer

    def cart2d_to_pol2d(self, M):
        Q1 = np.zeros((len(self.R),), dtype=np.float64)
        Q2 = np.zeros((len(self.R), len(self.A)), dtype=np.float64)
        norm = np.zeros((len(self.R),), dtype=np.float64)

        cl.enqueue_copy(self.queue, self.buf_M, M)
        cl.enqueue_fill_buffer(self.queue, self.buf_Q1e, np.float64(0.0), 0,
                               Q1.nbytes)
        cl.enqueue_fill_buffer(self.queue, self.buf_Q2e, np.float64(0.0), 0,
                               Q2.nbytes)
        cl.enqueue_fill_buffer(self.queue, self.buf_norm, np.float64(0.0), 0,
                               norm.nbytes)

        self.prg.cart2d_to_pol2d_project(self.queue, (self.r_len, 1), None,
                                         self.buf_fmt, self.buf_M,
                                         self.buf_Q1e, self.buf_Q2e,
                                         self.buf_norm)

        cl.enqueue_copy(self.queue, Q1, self.buf_Q1e)
        cl.enqueue_copy(self.queue, Q2, self.buf_Q2e)
        cl.enqueue_copy(self.queue, norm, self.buf_norm)

        cl.enqueue_barrier(self.queue).wait()

        Q1 /= norm.sum()

        return Q1, Q2

    def pol3d_to_cart2d(self, P1, P2):
        M = np.zeros((self.row_len, self.col_len), dtype=np.float64)

        cl.enqueue_copy(self.queue, self.buf_P1, P1)
        cl.enqueue_copy(self.queue, self.buf_P2, P2)
        cl.enqueue_fill_buffer(self.queue, self.buf_M, np.float64(0.0), 0,
                               M.nbytes)

        self.prg.pol3d_to_cart2d(self.queue, (self.row_len, self.col_len),
                                 None, self.buf_fmt, self.buf_P1, self.buf_P2,
                                 self.buf_M)

        cl.enqueue_copy(self.queue, M, self.buf_M)

        cl.enqueue_barrier(self.queue).wait()

        return M

    def pol3d_to_slice2d(self, P1, P2):
        S = np.zeros((self.row_len, self.col_len), dtype=np.float64)

        cl.enqueue_copy(self.queue, self.buf_P1, P1)
        cl.enqueue_copy(self.queue, self.buf_P2, P2)
        cl.enqueue_fill_buffer(self.queue, self.buf_S, np.float64(0.0), 0,
                               S.nbytes)

        self.prg.pol3d_to_slice2d(self.queue, (self.row_len, self.col_len),
                                  None, self.buf_fmt, self.buf_P1, self.buf_P2,
                                  self.buf_S)

        cl.enqueue_copy(self.queue, S, self.buf_S)

        cl.enqueue_barrier(self.queue).wait()

        return S

    def step(self, c1, c2):
        if not self.opencl_fast_step:
            return super().step(c1, c2)

        cl.enqueue_copy(self.queue, self.buf_Q1e, self.Q1_exp)
        cl.enqueue_copy(self.queue, self.buf_Q2e, self.Q2_exp)

        try:
            self.Q1_cal
        except AttributeError:
            # i = 1
            self.prg.pol2d_to_pol3d_init(self.queue, (self.r_len, 1), None,
                                         self.buf_fmt, self.buf_Q1e,
                                         self.buf_Q2e, self.buf_P1,
                                         self.buf_P2)

            self.Q1_cal = np.empty_like(self.Q1_exp)
            self.Q2_cal = np.empty_like(self.Q2_exp)
            self.P1 = np.empty_like(self.Q1_exp)
            self.P2 = np.empty_like(self.Q2_exp)

        else:
            # i > 1
            cl.enqueue_copy(self.queue, self.buf_Q1c, self.Q1_cal)
            cl.enqueue_copy(self.queue, self.buf_Q2c, self.Q2_cal)
            cl.enqueue_copy(self.queue, self.buf_P1, self.P1)
            cl.enqueue_copy(self.queue, self.buf_P2, self.P2)

            self.prg.pol2d_to_pol3d_step(
                self.queue, (self.r_len, 1), None, self.buf_fmt, self.buf_Q1e,
                self.buf_Q2e, self.buf_Q1c, self.buf_Q2c, self.buf_P1,
                self.buf_P2, np.float64(c1), np.float64(c2)
            )

        self.prg.norm_pol3d_angular(self.queue, (self.r_len, 1), None,
                                    self.buf_fmt, self.buf_P1, self.buf_P2,
                                    self.buf_norm)

        radial_norm = np.zeros((len(self.R),), dtype=np.float64)

        cl.enqueue_copy(self.queue, self.Q1_cal, self.buf_Q1c)
        cl.enqueue_copy(self.queue, self.Q2_cal, self.buf_Q2c)
        cl.enqueue_copy(self.queue, self.P1, self.buf_P1)
        cl.enqueue_copy(self.queue, self.P2, self.buf_P2)
        cl.enqueue_copy(self.queue, radial_norm, self.buf_norm)

        cl.enqueue_barrier(self.queue).wait()

        self.P1 /= radial_norm.sum()

        self.M_cal = self.pol3d_to_cart2d(self.P1, self.P2)
        self.norm_cart2d(self.M_cal)

        assert np.isfinite(self.M_cal).all(), 'M_cal is not finite'

        self.Q1_cal, self.Q2_cal = self.cart2d_to_pol2d(self.M_cal)

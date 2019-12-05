
import numpy as np


def cart2d_to_pol2d(self, M, Q1, Q2):
    for idx_r, r in enumerate(self.R):
        for idx_α, α in enumerate(self.A):
            row = -r * np.cos(α) + self.row_center
            col = r * np.sin(α) + self.col_center

            row_lo = int(row)
            row_up = row_lo + 1
            col_lo = int(col)
            col_up = col_lo + 1

            t = row - row_lo
            u = col - col_lo

            try:
                Q2[idx_r, idx_α] = (1 - t) * (1 - u) * M[row_lo, col_lo] + \
                                      t    * (1 - u) * M[row_up, col_lo] + \
                                   (1 - t) *    u    * M[row_lo, col_up] + \
                                      t    *    u    * M[row_up, col_up]
            except IndexError:
                Q2[idx_r, idx_α] = 0

            Q1[idx_r] += (Q2[idx_r, idx_α] * self.Δα)

        if Q1[idx_r] == 0:
            Q2[idx_r, :] = 0
        else:
            Q2[idx_r, :] /= Q1[idx_r]

    Q1 /= (Q2.sum(axis=1) * Q1 * self.R).sum() * self.Δr * self.Δα


def pol3d_to_cart2d(self, P1, P2, M):
    z = np.arange(self.R_len)

    for row in range(self.row_len):
        for col in range(self.col_len):
            x = float(col - self.col_center)
            y = float(self.row_center - row)

            idx_r = np.sqrt(x**2 + y**2 + z**2) / self.Δr
            idx_α = np.arctan2(np.sqrt(x**2 + z**2), y) / self.Δα

            r_lo = idx_r.astype(int)
            r_up = r_lo + 1
            α_lo = idx_α.astype(int)
            α_up = α_lo + 1

            t = idx_r - r_lo
            u = idx_α - α_lo
            
            # vectorized bounds check missing!

            M[row, col] += 2 * (
                (1 - t) * (1 - u) * P1[r_lo] * P2[r_lo, α_lo] + \
                   t    * (1 - u) * P1[r_up] * P2[r_up, α_lo] + \
                (1 - t) *    u    * P1[r_lo] * P2[r_lo, α_up] + \
                   t    *    u    * P1[r_up] * P2[r_up, α_up]
            )


def pol3d_to_section2d(self, P1, P2, M=None):
    if M is not None:
        assert M.shape == (self.row_len, self.col_len)
    else:
        M = np.zeros((self.row_len, self.col_len), dtype=np.float64)

    for row in range(self.row_len):
        for col in range(self.col_len):
            x = float(col - self.col_center)
            y = float(self.row_center - row)

            idx_r = np.sqrt(x**2 + y**2) / self.Δr
            idx_α = np.arctan2(abs(x), y) / self.Δα

            r_lo = int(idx_r)
            r_up = r_lo + 1
            α_lo = int(idx_α)
            α_up = α_lo + 1

            t = idx_r - r_lo
            u = idx_α - α_lo

            try:
                M[row, col] += 2 * (
                    (1 - t) * (1 - u) * P1[r_lo] * P2[r_lo, α_lo] + \
                       t    * (1 - u) * P1[r_up] * P2[r_up, α_lo] + \
                    (1 - t) *    u    * P1[r_lo] * P2[r_lo, α_up] + \
                       t    *    u    * P1[r_up] * P2[r_up, α_up]
                )
            except IndexError:
                M[row, col] = 0

    return M

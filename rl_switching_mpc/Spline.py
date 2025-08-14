"""
cubic spline planner

Author: Atsushi Sakai

"""
import math
import numpy as np
import bisect
from numba import njit

@njit
def spline_eval(a, b, c, d, x, t):
    for i in range(len(x) - 1):
        if x[i] <= t <= x[i + 1]:
            dx = t - x[i]
            return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
    return 0.0  # 혹은 NaN

@njit
def spline_eval_d(b, c, d, x, t):
    for i in range(len(x) - 1):
        if x[i] <= t <= x[i + 1]:
            dx = t - x[i]
            return b[i] + 2.0 * c[i] * dx + 3.0 * d[i] * dx**2
    return 0.0

@njit
def spline_eval_dd(c, d, x, t):
    for i in range(len(x) - 1):
        if x[i] <= t <= x[i + 1]:
            dx = t - x[i]
            return 2.0 * c[i] + 6.0 * d[i] * dx
    return 0.0

@njit
def fast_find_s(s, ax, bx, cx, dx, ay, by, cy, dy, x, y, s_step=0.01):
    s_closest = s[0]
    closest = 1e10
    si = s[0]

    while si < s[-1]:
        px = spline_eval(ax, bx, cx, dx, s, si)
        py = spline_eval(ay, by, cy, dy, s, si)
        dist = math.hypot(x - px, y - py)
        if dist < closest:
            closest = dist
            s_closest = si
        if dist < 0.001:
            break
        si += s_step
    return s_closest

class Spline:
    """
    Cubic Spline class
    """

    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient c
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)
        #  print(self.c1)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    # def calc(self, t):
    #     u"""
    #     Calc position

    #     if t is outside of the input x, return None

    #     """

    #     if t < self.x[0]:
    #         return None
    #     elif t > self.x[-1]:
    #         return None

    #     i = self.__search_index(t)
    #     dx = t - self.x[i]
    #     result = self.a[i] + self.b[i] * dx + \
    #         self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

    #     return result

    # def calcd(self, t):
    #     u"""
    #     Calc first derivative

    #     if t is outside of the input x, return None
    #     """

    #     if t < self.x[0]:
    #         return None
    #     elif t > self.x[-1]:
    #         return None

    #     i = self.__search_index(t)
    #     dx = t - self.x[i]
    #     result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
    #     return result

    # def calcdd(self, t):
    #     u"""
    #     Calc second derivative
    #     """

    #     if t < self.x[0]:
    #         return None
    #     elif t > self.x[-1]:
    #         return None

    #     i = self.__search_index(t)
    #     dx = t - self.x[i]
    #     result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
    #     return result
    def calc(self, t):
        return spline_eval(self.a, self.b, self.c, self.d, self.x, t)

    def calcd(self, t):
        return spline_eval_d(self.b, self.c, self.d, self.x, t)

    def calcdd(self, t):
        return spline_eval_dd(self.c, self.d, self.x, t)

    def __search_index(self, x):
        u"""
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        u"""
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        #  print(A)
        return A

    def __calc_B(self, h):
        u"""
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        #  print(B)
        return B


class Spline2D:
    u"""
    2D Cubic Spline class

    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = [math.sqrt(idx ** 2 + idy ** 2)
                   for (idx, idy) in zip(dx, dy)]
        s = [0.0]
        s.extend(np.cumsum(self.ds))
        return s

    def search_index(self, s):
        u"""
        calc index for given s
        """
        return bisect.bisect(self.s, s) - 1

    # def find_s(self, x, y):
    #     u"""
    #     calc spline s value from x, y
    #     """
    #     s_closest = self.s[0]
    #     closest = float("inf")
    #     si = self.s[0]

    #     while si < self.s[-1]:
    #         if si > self.s[-1]:
    #             si -= self.s[-1]
    #         px = self.sx.calc(si)
    #         py = self.sy.calc(si)
    #         dist = math.hypot(x - px, y - py)
    #         if dist < closest:
    #             closest = dist
    #             s_closest = si
    #         if dist < 0.001:
    #             return s_closest
    #         si += 0.01
    #     return s_closest
    def find_s(self, x, y):
        return fast_find_s(
            self.s,
            self.sx.a, self.sx.b, self.sx.c, self.sx.d,
            self.sy.a, self.sy.b, self.sy.c, self.sy.d,
            x, y
        )

    def local_find_s(self,x, y, s0, search_range):
        s_closest = s0
        closest = float("inf")
        si = s0
        range_covered = 0
        while range_covered < search_range:
            if si > self.s[-1]:
                si -= self.s[-1]
            if si < self.s[0]:
                si = self.s[-1] + si

            px = self.sx.calc(si)
            py = self.sy.calc(si)

            dist = math.hypot(x - px, y - py)

            if dist < closest:
                closest = dist
                s_closest = si

            if dist < 0.4:
                return s_closest

            range_covered += 0.4
            si = s0 + range_covered
        return s_closest

    def calc_position(self, s):
        u"""
        calc position
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)

        return x, y

    def calc_curvature(self, s):
        u"""
        calc curvature
        """
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2 + 1e-10)
        return k

    def calc_yaw(self, s):
        u"""
        calc yaw
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = math.atan2(dy, dx) # range : -pi to pi
        return yaw

    def calc_d(self, x, y, s0):
        s_closest = s0
        px = self.sx.calc(s0)
        py = self.sy.calc(s0)
        dx_dtheta = self.sx.calcd(s0)
        dy_dtheta = self.sy.calcd(s0)
        lateral_error = (x-px)*dy_dtheta - (y-py)*dx_dtheta
        return lateral_error


def calc_spline_course(x, y, ds=0.1):
    sp = Spline2D(x, y)
    s = list(np.arange(0, sp.s[-1], ds))

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s


if __name__ == '__main__':
    print("Spline 2D test")
    import matplotlib.pyplot as plt
    x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
    y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]

    sp = Spline2D(x, y)
    s = np.arange(0, sp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    plt.subplots(1)
    plt.plot(x, y, "xb", label="input")
    plt.plot(rx, ry, "-r", label="spline")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()

    plt.subplots(1)
    plt.plot(s, [np.rad2deg(iyaw) for iyaw in ryaw], "-r", label="yaw")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("yaw angle[deg]")

    plt.subplots(1)
    plt.plot(s, rk, "-r", label="curvature")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("curvature [1/m]")

    plt.show()

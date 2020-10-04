import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft

f_0 = 77.7*1e9 # 77.7GHz
phi_t = 0
A_t = 1
f_ramp = 425*1e6 #Hz
T_ramp = 0.010
m_w = f_ramp/T_ramp

T_ramp1 = T_ramp*0.7
T_ramp2 = T_ramp*0.3
m_w1 = f_ramp/T_ramp1
m_w2 = -f_ramp/T_ramp2

c = 299792458


def f_t(t):
    r1 = f_0 + m_w1 * (t % (T_ramp1 + T_ramp2))
    r1[(t % (T_ramp1 + T_ramp2)) > T_ramp1] = 0

    r2 = f_0 + m_w1 * T_ramp1 + m_w2 * ((t - T_ramp1) % (T_ramp1 + T_ramp2))
    r2[(t % (T_ramp1 + T_ramp2)) <= T_ramp1] = 0

    return r1 + r2


def f_r(t, v, r):
    r1 = f_0 + m_w1 * (t % (T_ramp1 + T_ramp2) - 2 * r / c) - 2 * v * f_0 / c
    r1[(t % (T_ramp1 + T_ramp2)) > T_ramp1] = 0

    r2 = f_0 + m_w1 * T_ramp1 + m_w2 * ((t - T_ramp1) % (T_ramp1 + T_ramp2) - 2 * r / c) - 2 * v * f_0 / c
    r2[(t % (T_ramp1 + T_ramp2)) <= T_ramp1] = 0
    return r1 + r2


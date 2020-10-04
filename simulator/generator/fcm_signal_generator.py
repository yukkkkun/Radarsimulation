import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft


c = 299792458

f_0 = 77.7*1e9 # 77.7GHz

# chirp sequence frequency
f_chirp = 50*1e3 #Hz

# ramp frequency
f_r = 200*1e6 #Hz
T_r = 1/f_chirp # duration of one cycle
m_w = f_r/T_r

n_r = 150 # number of chirps
T_M = T_r*n_r

# sample settings
f_s = 50e6 #50 MHz
n_s = int(T_r*f_s)

# some helpful
w_0 = 2*np.pi*f_0
lambda_0 = c/f_0

# frequencies found by FFT, will be used later
frequencies = np.arange(0, n_s // 2) * f_s / n_s
# the same as freq_to_range(frequencies)
ranges = frequencies*c/(2*m_w)

omega_second = 2*np.pi*np.concatenate((np.arange(0, n_r//2), np.arange(-n_r//2, 0)[::-1]))*f_chirp/n_r

velocities = omega_second*c/(4*np.pi*f_0)
###################################
# This is just for test.
# I'll make a module for verious objects later.

r_0 = 50 # initial distance
v_veh = 36/3.6 # velocity


def get_range(t):
    return r_0+v_veh*t

####################################

def f_transmitted(t):
    return f_0 + m_w*(t%T_r)

def chirp(t):
    return np.cos(2*np.pi*(f_transmitted(t))*t)


def itr(t):
    r = get_range(t)
    w_itr = 2*f_0*v_veh/c + 2*m_w*r/c
    # we do t%T_r because the eq. above only valid within the ramp
    v = np.cos(2*np.pi*w_itr*(t%T_r) +2*r*2*np.pi*f_0/c)
    return v

def freq_to_range(f):
    return f*c/(2*m_w)

def angle_freq_to_velocity(w):
    return w*c/(4*np.pi*f_0)

def generate_tsec():

    t_sample = np.linspace(0, T_M, n_r*n_s)

    v_sample = itr(t_sample)

    table = np.zeros((n_r, n_s))

    for chirp_nr in range(n_r):
        table[chirp_nr, :] = v_sample[(chirp_nr*n_s):(n_s*(chirp_nr+1))]

    table_df = pd.DataFrame(data=table,
                            columns=["sample_%03d"%i for i in range(n_s)],
                            index=["chirp_%03d"%i for i in range(n_r)])

    return table_df

def distance(table_df):

    range_table = np.zeros((n_r, n_s // 2), dtype=np.csingle)

    for chirp_nr in range(n_r):
        chirp_ad_values = table_df.iloc[chirp_nr].values
        chirp_fft = fft(chirp_ad_values)  # FFT
        range_table[chirp_nr, :] = 2.0 / n_s * chirp_fft[:n_s // 2]

    return range_table


def velocity(range_table):

    velocity_table = np.zeros((n_r, range_table.shape[1]), dtype=np.csingle)

    for r in range(range_table.shape[1]):
        range_bin_magn = range_table[:, r]
        range_bin_fft = fft(range_bin_magn)
        velocity_table[:, r] = 2.0 / n_r * range_bin_fft

    return velocity_table


def disp_range(range_table):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), sharex=True, sharey=True)
    abs_axes = ax[0, 0]
    phi_axes = ax[0, 1]
    real_axes = ax[1, 0]
    imag_axes = ax[1, 1]

    im_asb = abs_axes.imshow(np.abs(range_table), cmap=plt.get_cmap('RdYlBu'))
    abs_axes.set_xticks(range(ranges.size)[::50])
    abs_axes.set_xticklabels(ranges[::50], rotation=90)
    fig.colorbar(im_asb, ax=abs_axes)
    abs_axes.set_xlabel("range [m]")
    abs_axes.set_ylabel("chirp number")
    abs_axes.set_title("$|A(j\omega)|$")

    im_phi = phi_axes.imshow(np.angle(range_table) * 360 / (2 * np.pi), cmap=plt.get_cmap('RdYlBu'))
    fig.colorbar(im_phi, ax=phi_axes)
    phi_axes.set_xlabel("range [m]")
    phi_axes.set_ylabel("chirp number")
    phi_axes.set_title("$∠ A(j\omega)$")
    phi_axes.set_xticks(range(ranges.size)[::50])
    phi_axes.set_xticklabels(ranges[::50], rotation=90)

    im_real = real_axes.imshow(np.real(range_table), cmap=plt.get_cmap('RdYlBu'))
    fig.colorbar(im_real, ax=real_axes)
    real_axes.set_xlabel("range [m]")
    real_axes.set_ylabel("chirp number")
    real_axes.set_title("Real{$A(j\omega)$}")
    real_axes.set_xticks(range(ranges.size)[::50])
    real_axes.set_xticklabels(ranges[::50], rotation=90)

    im_imag = imag_axes.imshow(np.imag(range_table), cmap=plt.get_cmap('RdYlBu'))
    fig.colorbar(im_imag, ax=imag_axes)
    imag_axes.set_xlabel("range [m]")
    imag_axes.set_ylabel("chirp number")
    imag_axes.set_title("Imag{$A(j\omega)$}");
    imag_axes.set_xticks(range(ranges.size)[::50])
    imag_axes.set_xticklabels(ranges[::50], rotation=90);

    fig.suptitle("Range FFT table visualized.");

    plt.show()

def disp_range_velocity(velocity_table):
    plt.figure(figsize=(15, 10))
    plt.imshow(np.abs(velocity_table), cmap=plt.get_cmap('RdYlBu'))
    plt.xticks(range(ranges.size)[::20], ranges[::20], rotation=90);
    plt.yticks(range(velocities.size)[::10], velocities[::10]);
    plt.xlim([0, 200])
    plt.xlabel("range $r$ [m]")
    plt.ylabel("velocity $\\dot r = v$ [m/s]");
    plt.title("Chirp Sequence Modulation Result - $r, \\dot r$ map")
    plt.colorbar();
    plt.show()

def main():

    table_df = generate_tsec()
    range_table = distance(table_df)
    velocity_table = velocity(range_table)
    disp_range(range_table)
    disp_range_velocity(velocity_table)





if __name__ == "__main__":


    main()

    pass

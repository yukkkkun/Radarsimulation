import numpy as np

def range_resolution(num_adc_samples, dig_out_sample_rate=2500, freq_slope_const=60.012):
    """ Calculate the range resolution for the given radar configuration

    Args:
        num_adc_samples (int): The number of given ADC samples in a chirp
        dig_out_sample_rate (int): The ADC sample rate
        freq_slope_const (float): The slope of the freq increase in each chirp

    Returns:
        tuple [float, float]:
            range_resolution (float): The range resolution for this bin
            band_width (float): The bandwidth of the radar chirp config
    """
    light_speed_meter_per_sec = 299792458
    freq_slope_m_hz_per_usec = freq_slope_const
    adc_sample_period_usec = 1000.0 / dig_out_sample_rate * num_adc_samples
    band_width = freq_slope_m_hz_per_usec * adc_sample_period_usec * 1e6
    range_resolution = light_speed_meter_per_sec / (2.0 * band_width)

    return range_resolution, band_width


def range_processing(adc_data, window_type_1d=None, axis=-1):
    """Perform 1D FFT on complex-format ADC data.

    Perform optional windowing and 1D FFT on the ADC data.

    Args:
        adc_data (ndarray): (num_chirps_per_frame, num_rx_antennas, num_adc_samples). Performed on each frame. adc_data
                            is in complex by default. Complex is float32/float32 by default.
        window_type_1d (mmwave.dsp.utils.Window): Optional window type on 1D FFT input. Default is None. Can be selected
                                                from Bartlett, Blackman, Hanning and Hamming.

    Returns:
        radar_cube (ndarray): (num_chirps_per_frame, num_rx_antennas, num_range_bins). Also called fft_1d_out
    """
    # windowing numA x numB suggests the coefficients is numA-bits while the
    # input and output are numB-bits. Same rule applies to the FFT.
    fft1d_window_type = window_type_1d
    if fft1d_window_type:
        fft1d_in = utils.windowing(adc_data, fft1d_window_type, axis=axis)
    else:
        fft1d_in = adc_data

    # Note: np.fft.fft is a 1D operation, using higher dimension input defaults to slicing last axis for transformation
    radar_cube = np.fft.fft(fft1d_in, axis=axis)

    return radar_cube
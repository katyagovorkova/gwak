from lalinference import BurstSineGaussian, BurstSineGaussianF
import numpy as np

"""
Collection of waveforms used for injections, or PE recovery via bilby

COPIED EXACTLY from ETHAN MARX OLIB BY RYAN RAIKMAN
https://git.ligo.org/olib/olib/-/blob/main/libs/injection/olib/injection/waveforms.py
"""


def olib_time_domain_sine_gaussian(
    time_array,
    hrss,
    q,
    frequency,
    phase,
    eccentricity,
    geocent_time,
    **kwargs,
):
    """
    A wrapper to the oLIB source model in the time domain,
    convenient for use with bilby

    Args:
        time_array:
            times at which to evaluate the model
        hrss:
            hrss of source
        q:
            quality factor of the burst, determines the width in frequency
        frequency:
            the central frequency of the burst
        phase:
            a reference phase parameter
        eccentricity:
            signal eccentricity, determines the relative fraction of each
            polarisation mode
        **kwargs:
            A 'hack' to make this compatible with bilby_pipe.
            Currently bilby_pipe passes a lot of
            bbh specific parameters to the waveform model.
            This should allow those to be handled gracefully.
    Returns:
        dict containing the two polarization modes
    """

    # get dt from time array
    dt = time_array[1] - time_array[0]

    # cast arguments as proper types for lalinference
    hrss = float(hrss)
    eccentricity = float(hrss)
    phase = float(phase)
    frequency = float(frequency)
    q = float(q)

    # produces wf's with t0 at 0
    hplus, hcross = BurstSineGaussian(
        q, frequency, hrss, eccentricity, phase, dt
    )

    plus = np.zeros(len(time_array))
    cross = np.zeros(len(time_array))

    plus[: len(hplus.data.data)] = hplus.data.data
    cross[: len(hcross.data.data)] = hcross.data.data

    return dict(plus=plus, cross=cross)


def olib_freq_domain_sine_gaussian(
    freq_array, hrss, q, frequency, phase, eccentricity, geocent_time, **kwargs
):

    deltaF = freq_array[1] - freq_array[0]
    deltaT = 0.5 / (freq_array[-1])
    hplus, hcross = BurstSineGaussianF(
        q, frequency, hrss, eccentricity, phase, deltaF, deltaT
    )

    plus = np.zeros(len(freq_array), dtype=complex)
    cross = np.zeros(len(freq_array), dtype=complex)

    plus[: len(hplus.data.data)] = hplus.data.data
    cross[: len(hcross.data.data)] = hcross.data.data

    return dict(plus=plus, cross=cross)

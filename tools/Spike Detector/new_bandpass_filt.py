
def new_bandpass_filt(data, lowcut, highcut, fs, order = 4):
    if fs == 1024:
        b = [0.0342089439540094, 0, -0.0684178879080189, 0, 0.0342089439540094]
        a = [1, -3.40837438183729, 4.36773330454651, -2.50912852732152, 0.549774904492375]
        signal_bp = filtfilt(b, a, data, axis=0)

    elif fs == 512:
        b = [0.110341876092030, 0, -0.220683752184059, 0, 0.110341876092030]
        a = [1, -2.84999684509203, 3.01525532511459, -1.47261949410732, 0.307429302286222]
        signal_bp = filtfilt(b, a, data, axis=0)

    elif fs == 256:
        b = [0.329773746685091, 0, -0.659547493370182, 0, 0.329773746685091]
        a = [0.175233821173555, -0.200810082151730, 0.830452461803929, -1.80406441093276, 1]
        signal_bp = filtfilt(b, a, data, axis=0)

    elif fs == 500:
        b = [0.114657122916782, 0, -0.229314245833564, 0, 0.114657122916782]
        a = [1, -2.82405915548544, 2.95589050957282, -1.43119381095927, 0.299436856070785]
        signal_bp = filtfilt(b, a, data, axis=0)

    elif fs == 1000:
        b = [0.0356639677619672, 0, -0.0713279355239343, 0, 0.0356639677619672]
        a = [1, -3.39453926569963, 4.33233599727022, -2.47975692956295, 0.541965991567219]
        signal_bp = filtfilt(b, a, data, axis=0)

    elif fs == 250:
        b = [0.342416923871813, 0, -0.684833847743626, 0, 0.342416923871813]
        a = [1, -1.75428158626788, 0.736712673740108, -0.159617332038591, 0.178069768015428]
        signal_bp = filtfilt(b, a, data, axis=0)

    else:
        print("Sampling Frequency is not covered by this function, this is the best approximation")
        print("Spike rates/counts will not be affected by more than 5 percent of the original value")
        bandpass_b, bandpass_a = butter(order, [lowcut, highcut], btype='bandpass', fs=fs)
        signal_bp = filtfilt(bandpass_b, bandpass_a, data, axis=0)

    return signal_bp
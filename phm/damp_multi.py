import numpy as np
from scipy.signal import find_peaks, correlate
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

def DAMP_Multidim(T, dimension_num):
    if np.any(np.array(T.shape) == dimension_num):
        T = np.transpose(T)

    autocor = correlate(T[:, 0], T[:, 0], mode='full', method='auto')
    lags = np.arange(-len(T) + 1, len(T))
    peaks, _ = find_peaks(autocor[3009:4000], distance=3000)
    SubsequenceLength = lags[peaks + 3009]
    if not SubsequenceLength:
        SubsequenceLength = 1000
    SubsequenceLength = int(np.floor(SubsequenceLength[0]))

    CurrentIndex = SubsequenceLength * 5 + 1
    Left_MP = np.zeros(T[:, 0].shape)

    best_so_far = -np.inf
    bool_vec = np.ones(T[:, 0].shape)
    lookahead = 2**np.ceil(np.log2(16 * SubsequenceLength))

    for i in range(CurrentIndex, CurrentIndex + 16 * SubsequenceLength):
        if not bool_vec[i]:
            Left_MP[i] = Left_MP[i - 1] - 0.00001
            continue

        if i + SubsequenceLength - 1 > len(T[:, 0]):
            break

        multidim_distance_profile = np.zeros((i - SubsequenceLength + 1, 1))
        for d in range(dimension_num):
            T_tmp = T[:, d]
            query = T_tmp[i:i + SubsequenceLength]
            distance_profile = MASS_V2(T_tmp[:i], query)
            if not distance_profile.size:
                distance_profile = np.zeros((i - SubsequenceLength + 1, 1))
            multidim_distance_profile += distance_profile
        Left_MP[i] = np.min(multidim_distance_profile)

        best_so_far = np.max(Left_MP)

        if lookahead != 0:
            start_of_mass = i + SubsequenceLength
            if start_of_mass > len(T[:, 0]):
                start_of_mass = len(T[:, 0])
            end_of_mass = start_of_mass + lookahead - 1
            if end_of_mass > len(T[:, 0]):
                end_of_mass = len(T[:, 0])
            if end_of_mass - start_of_mass + 1 > SubsequenceLength:
                multidim_distance_profile = np.zeros((len(T_tmp[start_of_mass:end_of_mass]) - SubsequenceLength + 1, 1))
                for d in range(dimension_num):
                    T_tmp = T[:, d]
                    query = T_tmp[i:i + SubsequenceLength]
                    distance_profile = MASS_V2(T_tmp[start_of_mass:end_of_mass], query)
                    if not distance_profile.size:
                        distance_profile = np.zeros((len(T_tmp[start_of_mass:end_of_mass]) - SubsequenceLength + 1, 1))
                    multidim_distance_profile += distance_profile

                dp_index_less_than_BSF = np.where(multidim_distance_profile < best_so_far)[0]
                ts_index_less_than_BSF = dp_index_less_than_BSF + start_of_mass - 1
                bool_vec[ts_index_less_than_BSF] = 0

    for i in range(CurrentIndex + 16 * SubsequenceLength + 1, len(T[:, 0]) - SubsequenceLength + 1):
        if not bool_vec[i]:
            Left_MP[i] = Left_MP[i - 1] - 0.00001
            continue

        approximate_distance = np.inf
        X = 2**np.ceil(np.log2(8 * SubsequenceLength))
        flag = 1
        expansion_num = 0
        if i + SubsequenceLength - 1 > len(T[:, 0]):
            break

        while approximate_distance >= best_so_far:
            if i - X + 1 + (expansion_num * SubsequenceLength) < 1:
                multidim_distance_profile = np.zeros((i - SubsequenceLength + 1, 1))
                for d in range(dimension_num):
                    T_tmp = T[:, d]
                    query = T_tmp[i:i + SubsequenceLength]
                    distance_profile = MASS_V2(T_tmp[:i], query)
                    if not distance_profile.size:
                        distance_profile = np.zeros((i - SubsequenceLength + 1, 1))
                    multidim_distance_profile += distance_profile
                approximate_distance = np.min(multidim_distance_profile)
                Left_MP[i] = approximate_distance
                if approximate_distance > best_so_far:
                    best_so_far = approximate_distance
                break
            else:
                if flag == 1:
                    flag = 0
                    multidim_distance_profile = np.zeros((len(T_tmp[i - X + 1:i]) - SubsequenceLength + 1, 1))
                    for d in range(dimension_num):
                        T_tmp = T[:, d]
                        query = T_tmp[i:i + SubsequenceLength]
                        distance_profile = MASS_V2(T_tmp[i - X + 1:i], query)
                        if not distance_profile.size:
                            distance_profile = np.zeros((len(T_tmp[i - X + 1:i]) - SubsequenceLength + 1, 1))
                        multidim_distance_profile += distance_profile
                    approximate_distance = np.min(multidim_distance_profile)
                else:
                    X_start = i - X + 1 + (expansion_num * SubsequenceLength)
                    X_end = i - (X / 2) + (expansion_num * SubsequenceLength)
                    multidim_distance_profile = np.zeros((len(T_tmp[X_start:X_end]) - SubsequenceLength + 1, 1))
                    for d in range(dimension_num):
                        T_tmp = T[:, d]
                        query = T_tmp[i:i + SubsequenceLength]
                        distance_profile = MASS_V2(T_tmp[X_start:X_end], query)
                        if not distance_profile.size:
                            distance_profile = np.zeros((len(T_tmp[X_start:X_end]) - SubsequenceLength + 1, 1))
                        multidim_distance_profile += distance_profile
                    approximate_distance = np.min(multidim_distance_profile)

                if approximate_distance < best_so_far:
                    Left_MP[i] = approximate_distance
                    break
                else:
                    X = 2 * X
                    expansion_num += 1

        if lookahead != 0:
            start_of_mass = i + SubsequenceLength
            if start_of_mass > len(T[:, 0]):
                start_of_mass = len(T[:, 0])
            end_of_mass = start_of_mass + lookahead - 1
            if end_of_mass > len(T[:, 0]):
                end_of_mass = len(T[:, 0])
            if end_of_mass - start_of_mass + 1 > SubsequenceLength:
                multidim_distance_profile = np.zeros((len(T_tmp[start_of_mass:end_of_mass]) - SubsequenceLength + 1, 1))
                for d in range(dimension_num):
                    T_tmp = T[:, d]
                    query = T_tmp[i:i + SubsequenceLength]
                    distance_profile = MASS_V2(T_tmp[start_of_mass:end_of_mass], query)
                    if not distance_profile.size:
                        distance_profile = np.zeros((len(T_tmp[start_of_mass:end_of_mass]) - SubsequenceLength + 1, 1))
                    multidim_distance_profile += distance_profile

                dp_index_less_than_BSF = np.where(multidim_distance_profile < best_so_far)[0]
                ts_index_less_than_BSF = dp_index_less_than_BSF + start_of_mass - 1
                bool_vec[ts_index_less_than_BSF] = 0

    PV = bool_vec[CurrentIndex:len(T[:, 0]) - SubsequenceLength + 1]
    PR = (len(PV) - np.sum(PV)) / len(PV)
    print(f"Pruning Rate: {PR}")

    val = np.max(Left_MP)
    loc = np.argmax(Left_MP)
    print(f"Predicted discord score/position: {val}/{loc}")

    plt.figure()
    plt.plot(Left_MP, 'b')
    for d in range(dimension_num):
        plt.plot(np.array(T[:, d]) - 10 * d, 'r')
    plt.show()

def MASS_V2(x, y):
    m = len(y)
    n = len(x)

    meany = np.mean(y)
    sigmay = np.std(y)

    meanx = np.convolve(x, np.ones(m)/m, mode='valid')
    sigmax = np.sqrt(np.convolve(x**2, np.ones(m)/m, mode='valid') - meanx**2)

    y = y[::-1]
    y = np.pad(y, (0, n - m), 'constant')

    X = fft(x)
    Y = fft(y)
    Z = X * Y
    z = ifft(Z)

    dist = 2 * (m - (z[m-1:n] - m * meanx * meany) / (sigmax * sigmay))
    return np.sqrt(dist)


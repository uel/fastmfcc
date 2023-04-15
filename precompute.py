import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_fft", type=int, help="the number of FFT points", default=1024)
parser.add_argument("--n_mels", type=int, help="the number of Mel bands", default=48)
parser.add_argument("--n_mfcc", type=int, help="the number of MFCCs", default=13)
parser.add_argument("--sr", type=int, help="the sampling rate of the audio", default=22050)
args = parser.parse_args()

n_fft, n_mels, n_mfcc, sr = args.n_fft, args.n_mels, args.n_mfcc, args.sr

import numpy as np
import librosa
import scipy

# 16 bit hanning window, not symmetrical
hanning = (scipy.signal.windows.hann(n_fft, sym=False)*8192).astype(np.uint16)[:]

# 256 sine values between 0 and pi/2 in bytes
sin_data = (np.sin(np.linspace(0, np.pi/2, 256))*65535).astype(np.uint16)

# Compute mel filterbank using librosa
fbank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

weight_count = int(np.floor(n_fft / 2)) + 1
# Save even and odd mel filterbank weights
even_mel_weights = np.zeros(weight_count, dtype=np.uint8)
even_mel_indicies = np.zeros(weight_count, dtype=np.uint8)

odd_mel_weights = np.zeros(weight_count, dtype=np.uint8)
odd_mel_indicies = np.zeros(weight_count, dtype=np.uint8)

filter_bank_scale = int(np.floor((255.0 / fbank.max())))

for m in range(n_mels):
    for k in range(weight_count):
        if fbank[m, k] > 0:
            if m % 2 == 0:
                even_mel_weights[k] += fbank[m, k]*filter_bank_scale
                even_mel_indicies[k] = m
            else:
                odd_mel_weights[k] += fbank[m, k]*filter_bank_scale
                odd_mel_indicies[k] = m

#DCT data
dct_basis = scipy.fftpack.dct(np.eye(n_mels, dtype=np.float32), axis=0, norm='ortho')[:n_mfcc]

# Save data to file
with open("computed.h", "w") as f:
    f.write("//File created with precompute.py\n")
    f.write("const unsigned int n_fft = {};\n".format(n_fft))
    f.write("const unsigned int n_mels = {};\n".format(n_mels))
    f.write("const unsigned int n_mfcc = {};\n".format(n_mfcc))
    f.write("const unsigned short hanning[{}] = {{".format(n_fft))

    for i in range(n_fft):
        f.write("{}".format(hanning[i]))
        if i < n_fft-1:
            f.write(", ")
    f.write("};\n")

    f.write("const unsigned short sin_data[{}] = {{".format(256))
    for i in range(256):
        f.write("{}".format(sin_data[i]))
        if i < 256-1:
            f.write(", ")
    f.write("};\n")

    f.write("const unsigned int filter_bank_scale = {};\n".format(filter_bank_scale))

    f.write("const unsigned char even_mel_weights[{}] = {{".format(weight_count))
    for i in range(weight_count):
        f.write("{}".format(even_mel_weights[i]))
        if i < weight_count-1:
            f.write(", ")
    f.write("};\n")

    f.write("const unsigned char odd_mel_weights[{}] = {{".format(weight_count))
    for i in range(weight_count):
        f.write("{}".format(odd_mel_weights[i]))
        if i < weight_count-1:
            f.write(", ")
    f.write("};\n")

    f.write("const unsigned char even_mel_indicies[{}] = {{".format(weight_count))
    for i in range(weight_count): 
        f.write("{}".format(even_mel_indicies[i]))
        if i < weight_count-1:
            f.write(", ")
    f.write("};\n")

    f.write("const unsigned char odd_mel_indicies[{}] = {{".format(weight_count))
    for i in range(weight_count):
        f.write("{}".format(odd_mel_indicies[i]))
        if i < weight_count-1:
            f.write(", ")
    f.write("};\n")

    f.write('const float dct_basis[{}][{}] = {{\n'.format(13, n_mels))
    for k in range(n_mfcc):
        f.write('    {')
        for n in range(n_mels):
            f.write('{:.8f}f, '.format(dct_basis[k, n]))
        f.write('},\n')
    f.write('};\n\n')

print("Memory usage: {} bytes".format(2*n_fft + 2*256 + (n_fft/2+1)*(1+1+1+1) + 4*n_mfcc*n_mels))
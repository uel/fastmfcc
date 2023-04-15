#include <math.h>
#include "computed.h"
#include "fastmfcc.h"
#define max(x, y) (((x) > (y)) ? (x) : (y))
#define min(x, y) (((x) < (y)) ? (x) : (y))

int fsin(int a, int x)
{
    x = ((x % 1024) + 1024) % 1024;
    
    if ( x % 256 == 0 ) // sin is 0 or +-1
    {
        x = x >> 8;
        if ( x > 1 ) x = 2 - x;
        return a * x;
    }

    if ( x >> 9 ) a = -a; // sin will be negative if angle > 512

    switch (x >> 8) {
        case 1:
            x = 512 - x;
            break;
        case 2:
            x = x - 512;
            break;
        case 3:
            x = 1024 - x;
            break; 
        default:
            break;
    }

    return a * sin_data[x] / 65535;
}

int fcos(int a, int x)
{
    x = 256 - x;
    return (fsin(a, x));
}

int bit_reverse(int x, int p) {
    int y = 0;
    for (int i = 0; i < p; i++) {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    return y;
}

int FFT(int in[], float out[], int n_fft)
{
    int p = floor(log2(n_fft));

    int r[n_fft] = {0};
    int im[n_fft] = {0};

    int data_max = 0;
    long data_avg = 0;
    int data_min = 0;
    for (int i = 0; i < n_fft; i++)
    {
        data_avg = data_avg + in[i];
        data_max = max(data_max, in[i]);
        data_min = min(data_min, in[i]);
    }
    data_avg >>= p;

    int scale = floor(log2(data_max - data_min)) - 10;
    for (int i = 0; i < n_fft; i++) {
        in[i] = in[i] - data_avg;
        in[i] = scale > 0 ? in[i] >> scale : in[i] << -scale;
    }

    for (int i = 0; i < n_fft; i++) {
        r[i] = in[bit_reverse(i, p)];
    }

    for (int i = 1; i <= p; i++) 
    {
        bool change_scaling = 0;
        int m = 1 << i-1;
        int n = n_fft >> i; 
        float e = -1024. / (1 << i);

        for (int j = 0; j < m; j++)
        {
            int c = e * j; 
            for (int k = j; k < n*2*m; k+= 2*m)
            {
                int tr = fcos(r[m + k], c) - fsin(im[m + k], c);
                int tim = fsin(r[m + k], c) + fcos(im[m + k], c);

                r[k + m] = r[k] - tr;
                r[k] = r[k] + tr;
                im[k + m] = im[k] - tim;
                im[k] = im[k] + tim;

                if ( (abs(r[k]) | abs(im[k])) >> 14 ) {
                    change_scaling = 1;
                } 
            }
        }

        if (change_scaling) {
            for (int j = 0; j < n_fft; j++) {
                r[j] = r[j] >> 1;
                im[j] = im[j] >> 1;
            }
            change_scaling = false;
            scale++; 
        }
    }

    for (int i = 0; i < n_fft/2+1; i++) {
        out[i] = r[i]*r[i] + im[i]*im[i]; 
    }

    return scale;
}

void MFCC(short *data, float *res)
{
    int fft_input[n_fft];
    for (int i = 0; i < n_fft; i++) { 
        fft_input[i] = ( data[i] * hanning[i] ) >> 13;
    }
    
    float spec[n_fft/2+1];
    int fft_scale = FFT(fft_input, spec, n_fft);

    float mel_values[n_mels] = {0};
    for (int i = 0; i < n_fft/2+1; i++) {
        mel_values[even_mel_indicies[i]] += even_mel_weights[i]*spec[i];
        mel_values[odd_mel_indicies[i]] += odd_mel_weights[i]*spec[i];
    } 

    // scale back spectrum, magnitude, mel weights scale + power spectrogram
    float scale = float(1 << fft_scale*2)/filter_bank_scale;
    int spec_max = 0;
    for (int i = 0; i < n_mels; i++) {
        mel_values[i] = 10*log10(max(1, mel_values[i] * scale));
        spec_max = max(mel_values[i], spec_max);
    }
    for (int i = 0; i < n_mels; i++) {
        mel_values[i] = max(mel_values[i], spec_max-80);
    }

    // Compute the DCT for each coefficient k
    for (int k = 0; k < n_mfcc; k++) {
        float sum = 0.0;
        for (int n = 0; n < n_mels; n++) {
            sum += mel_values[n] * dct_basis[k][n];
        }
        res[k] = sum;
    }   
}

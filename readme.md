# FastMFCC - Realtime signal processing on microcontrollers
FastMFCC computes Mel-Frequency Cepstral Coefficients (MFCC) on microcontrollers with limited resources. The library is intended for realtime machine learning and IOT applications. Its output closely matches that of the Librosa librosa.feature.mfcc function with minimal errors.

## Dependencies
The FastMFCC library requires Python and librosa for precomputing the coefficients.

`pip install librosa`

## Usage
To use the library, run precompute.py with the following parameters:

`python precompute.py --n_mfcc=<num_mfcc> --n_mels=<num_mels> --n_fft=<num_fft> --sr=<sample_rate>`

This will create a file computed.h with precomputed coefficients for the MFCC calculations. The coefficients can be computed by including the header file and calling the MFCC function:

```
#include "fastmfcc.h"
...
short data[1024] = {...};
float mfccs[13];
MFCC(test_data, mfccs);
```

Precomputed coefficients will take up between 2-20KB of additional read-only data depending on the choosen parameters but lead to significant speed increases. A minimal amount of floating-point operations is used to enable realtime performance without an FPU.

## Microcontroller Compatibility
Has been tested to work on Arduino NANO 33 BLE and Raspberry Pi Pico with 1024 frame size, 48 mel bins and 13 mfccs.

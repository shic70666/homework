''' Pre-Processsing simualtion Data
'''
# %%
import numpy as np
import scipy.io as scio
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import scipy.signal as signal

dataFile = '/Users/shic/SHM_Data/Ambient/shm01a.mat'
X = scio.loadmat(dataFile)
data_all = X['dasy']
data = data_all[0,0]['DA01']
# print (data.shape)
# Currently only one channel of data is imported for testing, all channels, all operating conditions and all external excitations should be considered
# %%
#!------------------------------------------------------------------------------
#!                                    TRY FUNCTION 1
#!------------------------------------------------------------------------------
def fftlw(Fs,y,draw):
    '''
    Parameters
    ------------
    Fs: Sampling Frequency
    y: Vibration(Acceleration) Signal
    draw: value=1, draw the Spectrogram; value=0, no drawing

    Returns
    ------------
    f:frequency
    M:Amplitude
    '''
    L=len(y)                               # Number of sampling points
    f = np.arange(int(L / 2)) * Fs / L     # Frequency
    #M = np.abs(np.fft.fft(y))*2/L         # Using the numpy.fft.fft() function and normalizing
    M = np.abs((fft(y))) *2/L              # Using the scipy.fftpack.fft() function and normalizing
    M = M[0:int(L / 2)]                    # Take a one-sided spectrum
    M[0]=M[0]/2                            # Constant divided by 2
    
    if draw == 1:                            # Visualisation
        plt.figure()
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.plot(f,M)
        plt.xlabel('f/HZ')
        plt.ylabel('Amplitude')
        plt.title('Spectrogram')
    return f,M

#%%
#!------------------------------------------------------------------------------
#!                                    Test FUNCTION 1
#!------------------------------------------------------------------------------
# Fs = 200                                    # Sampling frequency
# L = 60000                                   # Number of sampling points
# t = np.arange(0,L/Fs,1/Fs)                  # Time Series        
# y = data[:2000]                             # Signal sequences 
# # y1=np.sin(2*np.pi*t*100)*t+3*np.sin(2*np.pi*t*200)+3
# f,M = fftlw(Fs,y,1)                         # Fast Fourier variation and drawing a spectrum


# %%
#!------------------------------------------------------------------------------
#!                                    TRY FUNCTION 2
#!------------------------------------------------------------------------------
def stft(x, **params):
    '''
    :param x: input signal
    :param params: {fs:Sampling frequency;
                    window:Default is Hamming Window;
                    nperseg:Length of each segment, default is 256;
                    noverlap:The number of points that overlap. The COLA constraint needs to be satisfied when specifying the value. Default is half the window length;
                    nfft:Length of fft;
                    detrend:(str、function or False)Specify how to go to trend, default is Flase, do not go to trend;
                    return_onesided:Default is True, which returns a one-sided spectrum;
                    boundary:Add 0 to both ends of the time series by default;
                    padded:Whether to padding the time series with 0 (when the length is not sufficient);
                    axis:It is not necessary to care about this parameter}
    :return: f:Sampling frequency arrays;t:Segment time arrays;Zxx:result of STFT
    '''
    f, t, zxx = signal.stft(x, **params) 
    return f, t, zxx

def stft_specgram(x, picname=None, **params):    # picname is the name given to the image, in order to save the image
    f, t, zxx = stft(x, **params)
    plt.pcolormesh(t, f, np.abs(zxx))
    plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    if picname is not None:
        plt.savefig('..\\picture\\' + str(picname) + '.jpg')       # Save image
    plt.clf()      # Clear the canvas
    return t, f, zxx

#%%
#!------------------------------------------------------------------------------
#!                                    Test FUNCTION 2
#!------------------------------------------------------------------------------







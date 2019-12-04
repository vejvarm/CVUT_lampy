from __future__ import division

from scipy import signal
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

Fvzorkovani=512 #Hz
Fnyquist=Fvzorkovani*0.5  #[Hz]

'''
Nyquistova frekvence je nejrychlejsi frekvence signalu,
kterou lze pro danou vzorkovaci frekvenci snimat.
(Nyquistuv vzorkovaci teorem (Shanon-Kotelnikuv):
tzv. "Nyquist rate" je 2x rychlejsi nez nejrychlejsi snimana frekvence
'''
Fbandreject=200#Hz

#Wn=1 ... Fnyquist
#Wn=Fcutoff/Fnyquist

if __name__ == "__main__":
    b, a = signal.butter(5, [40/Fnyquist, 60/Fnyquist], 'bandstop')  #, analog=True)
    w, h = signal.freqz(b, a)
    plt.semilogx(w*Fnyquist/np.pi, 20 * np.log10(abs(h)))
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0.1, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(Fbandreject, color='green') # cutoff frequency
    #plt.show()

    print(b,a)

    #signal
    dt=1./Fvzorkovani
    t=np.arange(0,1,dt)
    N=len(t)  # delka dat
    ytrue=np.sin(2*np.pi*10*t)+np.sin(2*np.pi*50*t)*.5+np.sin(2*np.pi*70*t)*.3

    noise=np.random.poisson(10,N)
    noise=(noise-np.mean(noise))/np.std(noise)

    noise=np.random.randn(N)

    yr=ytrue+noise*1 #

    y_lfilter=signal.lfilter(b,a,yr)
    y_filtfilt=signal.filtfilt(b,a,yr)

    f=np.fft.fftfreq(N,1/Fvzorkovani)[:int(N/2)]
    print(f)
    plt.figure(figsize=(12,3))
    plt.plot(f,abs(np.fft.fft(yr))[:int(N/2)],label="PSD merene",linewidth=4)
    plt.plot(f,abs(np.fft.fft(y_filtfilt))[:int(N/2)],label="PSD filtrovane")
    plt.xlabel("f [Hz]")
    plt.grid();plt.legend()


    #===============================
    plt.figure(figsize=(12,3))
    plt.plot(t,ytrue,'m',label="ytrue",linewidth=4)
    plt.plot(t,yr,'k',label="merena  $yr=sin(2\pi \cdot 10\cdot t)+0.5\cdot sin(2\cdot \pi\cdot50\cdot t)$")
    plt.plot(t,y_lfilter,'r',linewidth=3,label="y_lfilter")
    plt.plot(t,y_filtfilt,'g',linewidth=2,label="y_filtfilt")
    plt.grid()
    plt.legend()
    plt.show()

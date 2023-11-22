# -*- coding: utf-8 -*-
"""

@author: Grupo 4 - Sistemas de Radiocomunicacion
"""

#%% Libs and functions
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, ifft, fftshift, fftfreq
from matplotlib import cm

def fastconv(A,B):
    out_len = len(A)+len(B)-1

    # Next nearest power of 2
    sizefft = int(2**(np.ceil(np.log2(out_len))))

    Afilled = np.concatenate((A,np.zeros(sizefft-len(A))))
    Bfilled = np.concatenate((B,np.zeros(sizefft-len(B))))

    fftA = fft(Afilled)
    fftB = fft(Bfilled)

    fft_out = fftA * fftB
    out = ifft(fft_out)

    out = out[0:out_len]

    return out

#%% Parameters

c = 3e8 # speed of light [m/s]
k = 1.380649e-23 # Boltzmann

fc = 1.3e9 # Carrier freq
fs = 10e6 # Sampling freq
Np = 100 # Intervalos de sampling
Nint = 10
NPRIs = Nint*Np
ts = 1/fs

Te = 5e-6 # Tx recovery Time[s]
Tp = 10e-6 # Tx Pulse Width [s]
BW = 2e6 # Tx Chirp bandwidth [Hz]
PRF = 1500 # Pulse repetition Frequency [Hz]

wlen = c/fc # Wavelength [m]
kwave = 2*np.pi/wlen # Wavenumber [rad/m]
PRI = PRF**(-1) # Pulse repetition interval [s]
ru = (c*(PRI-Tp-Te))/2 # Unambigous Range [m]
vu_ms = wlen*PRF/2 # Unambigous Velocity [m/s]
vu_kmh = vu_ms*3.6 # Unambigous Velocity [km/h]

rank_min = (Tp/2+Te)*c/2 # Minimum Range [m]
rank_max = 30e3 # Maximum Range [m] (podría ser el Ru)
#rank_max = ru
rank_res = ts*c/2 # Range Step [m]
tmax = 2*rank_max/c # Maximum Simulation Time

radar_signal = pd.read_csv('signal.csv',index_col=None)
radar_signal = np.array(radar_signal['real']+1j*radar_signal['imag'])
radar_signal = radar_signal.reshape(Np,-1)

print(f'Pulse repetition Interval. PRI = {PRI*1e6:.2f} μs')
print(f'Unambiguous Range. Ru = {ru/1e3:.3f} km')
print(f'Unambiguous Velocity. Vu = {vu_ms:.2f} m/s')
print(f'Unambiguous Velocity. Vu = {vu_kmh:.2f} km/h')
print(f'Minimum Range. Rmin = {rank_min/1e3:.3f} km')
print(f'Maximum Range. Rmin = {rank_max/1e3:.3f} km')

#%% Signals

# Independant Variables

Npts = int(tmax/ts) # Simulation Points
t = np.linspace(-tmax/2,tmax/2,Npts)
ranks = np.linspace(rank_res,rank_max,Npts) # Range Vector
f = fftfreq(Npts,ts) # Freq Vector

# Tx Signal

tx_chirp = np.exp(1j*np.pi*BW/Tp * t**2) # Tx Linear Chiprs (t)
tx_rect = np.where(np.abs(t)<=Tp/2,1,0) # Rect Function
tx_chirp = tx_rect*tx_chirp # Tx Chirp Rectangular
tx_chirp_f = fft(tx_chirp,norm='ortho') # Tx Chirp (f)

# Matched Filter

matched_filter = np.conj(np.flip(tx_chirp))
matched_filter_f = fft(matched_filter,norm='ortho')

#%% Plot Signals

fig, axes = plt.subplots(2,1,figsize=(8,8),sharex=True)

fig.suptitle('Received Signal')

ax = axes[0]
ax.plot(ranks/1e3,np.real(radar_signal[0]), label = "Re")
ax.plot(ranks/1e3,np.imag(radar_signal[0]), label = "Im")
ax.set_ylabel('Amplitude')
ax.set_xlabel('Range [km]')
ax.grid(True)
ax.legend()

ax = axes[1]
ax.plot(ranks/1e3,np.abs(radar_signal[0]))
ax.set_ylabel('Abs Amplitude')
ax.set_xlabel('Range [km]')
ax.grid(True)
#%% Matched Filtering (Compresión con Filtro Apareado)

compressed_signal = []
for t in range(len(radar_signal)):
    compressed_signal_i = fastconv(radar_signal[t], matched_filter)[len(matched_filter)//2:len(matched_filter)//2 + len(ranks)]
    compressed_signal.append(compressed_signal_i)

compressed_signal=np.stack(compressed_signal,axis=0)

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

fig.suptitle('Senial Comprimida y Descomprimida')

ax = axes[0]
ax.plot(ranks/1e3, np.abs(radar_signal[0]))
ax.set_ylabel('Abs Amplitude')
ax.set_xlabel('Rx Raw signal')
ax.grid(True)

ax = axes[1]
ax.plot(ranks/1e3, np.abs(compressed_signal[0]))
ax.set_ylabel('Abs Amplitude')
ax.set_xlabel('Rx compressed signal')
ax.grid(True)

plt.show()

#%% CFAR Window

# Parámetros CFAR
gap = 25
ref = 150
v_ref = 1
threshold_factor = 1/(ref*v_ref)  # Factor de umbral

cfar1=np.repeat(threshold_factor, ref/2)
cfar2=np.zeros(gap*2)
cfar3=np.concatenate((cfar1, cfar2, cfar1))

# Plot Valor Absoluto de la ventana CFAR
plt.figure(figsize=(10, 5))
plt.step(range(len(cfar3)),cfar3)
plt.xlabel('Número de muestra')
plt.ylabel('Valor Absoluto de la CFAR Window')
plt.title('CFAR Window - Valor Absoluto vs. Número de muestra')
plt.grid(True)
plt.show()

#%% MTIsc

abs_radar_signal = np.abs(radar_signal)
abs_compressed_signal = np.abs(compressed_signal)
gain_MTIsc=4
MTIsc=(compressed_signal[1])-(compressed_signal[0])
MTIsc_abs=np.abs(MTIsc)
threshold_MTIsc=gain_MTIsc*fastconv(cfar3,MTIsc_abs)[len(cfar3)//2:len(cfar3)//2 + len(ranks)]

resta_MTIsc=np.abs(MTIsc_abs)-np.abs(threshold_MTIsc)
signo_MTIsc=np.sign(resta_MTIsc)
diff_MTIsc=np.diff(signo_MTIsc)
diff_MTIsc = np.append(diff_MTIsc, 0)

# Crear subplots
fig, axes = plt.subplots(4, 1, figsize=(10, 20), sharex=True)

axes[0].plot(ranks / 1e3, abs_radar_signal[0], label='Rx t0')
axes[0].plot(ranks / 1e3, abs_radar_signal[1], label='Rx t1')
axes[0].set_ylabel('Value')
axes[0].set_title('Rx Raw Signals')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(ranks / 1e3, abs_compressed_signal[0], label='Comp t0')
axes[1].plot(ranks / 1e3, abs_compressed_signal[1], label='Comp t1')
axes[1].set_ylabel('Value')
axes[1].set_title('Rx Compressed Signals')
axes[1].grid(True)
axes[1].legend()

axes[2].plot(ranks / 1e3, np.abs(MTIsc), label='MTIsc')
axes[2].plot(ranks / 1e3, threshold_MTIsc, label='Umbral MTI')
axes[2].set_ylabel('Value')
axes[2].set_title('MTI SC')
axes[2].grid(True)
axes[2].legend()

axes[3].plot(ranks / 1e3, np.abs(diff_MTIsc))
axes[3].set_xlabel('Range [km]')
axes[3].set_ylabel('Value')
axes[3].set_title('MTI SC Signal')
axes[3].grid(True)
axes[3].legend()

plt.tight_layout()
plt.show()

#%% STIsc

gain_STIsc=9
STIsc=(compressed_signal[1])+(compressed_signal[0])
STIsc_abs=np.abs(STIsc)
threshold_STIsc=gain_STIsc*fastconv(cfar3,STIsc_abs)[len(cfar3)//2:len(cfar3)//2 + len(ranks)]

resta_STIsc=np.abs(STIsc_abs)-np.abs(threshold_STIsc)
signo_STIsc=np.sign(resta_STIsc)
diff_STIsc=np.diff(signo_STIsc)
diff_STIsc = np.append(diff_STIsc, 0)

# Crear subplots
fig, axes = plt.subplots(4, 1, figsize=(10, 20), sharex=True)

axes[0].plot(ranks / 1e3, abs_radar_signal[0], label='Rx t0')
axes[0].plot(ranks / 1e3, abs_radar_signal[1], label='Rx t1')
axes[0].set_ylabel('Value')
axes[0].set_title('Rx Raw Signals')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(ranks / 1e3, abs_compressed_signal[0], label='Comp t0')
axes[1].plot(ranks / 1e3, abs_compressed_signal[1], label='Comp t1')
axes[1].set_ylabel('Value')
axes[1].set_title('Rx Compressed Signals')
axes[1].grid(True)
axes[1].legend()

axes[2].plot(ranks / 1e3, np.abs(STIsc), label='STIsc')
axes[2].plot(ranks / 1e3, threshold_STIsc, label='Umbral STI')
axes[2].set_ylabel('Value')
axes[2].set_title('STI SC')
axes[2].grid(True)
axes[2].legend()

axes[3].plot(ranks / 1e3, np.abs(diff_STIsc), label='STIsc')
axes[3].set_xlabel('Range [km]')
axes[3].set_ylabel('Value')
axes[3].set_title('STI SC Signal')
axes[3].grid(True)
axes[3].legend()

plt.tight_layout()
plt.show()

#%% MTIdc

gain_MTIdc=4
MTIdc=(compressed_signal[2])+(compressed_signal[0])-2*compressed_signal[1]
MTIdc_abs=np.abs(MTIdc)
threshold_MTIdc=gain_MTIdc*fastconv(cfar3,MTIdc_abs)[len(cfar3)//2:len(cfar3)//2 + len(ranks)]

resta_MTIdc=np.abs(MTIdc_abs)-np.abs(threshold_MTIdc)
signo_MTIdc=np.sign(resta_MTIdc)
diff_MTIdc=np.diff(signo_MTIdc)
diff_MTIdc = np.append(diff_MTIdc, 0)

# Crear subplots
fig, axes = plt.subplots(4, 1, figsize=(10, 20), sharex=True)

axes[0].plot(ranks / 1e3, abs_radar_signal[0], label='Rx t0')
axes[0].plot(ranks / 1e3, abs_radar_signal[1], label='Rx t1')
axes[0].plot(ranks / 1e3, abs_radar_signal[2], label='Rx t2')
axes[0].set_ylabel('Value')
axes[0].set_title('Rx Raw Signals')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(ranks / 1e3, abs_compressed_signal[0], label='Comp t0')
axes[1].plot(ranks / 1e3, abs_compressed_signal[1], label='Comp t1')
axes[1].plot(ranks / 1e3, abs_compressed_signal[2], label='Comp t2')
axes[1].set_ylabel('Value')
axes[1].set_title('Rx Compressed Signals')
axes[1].grid(True)
axes[1].legend()

axes[2].plot(ranks / 1e3, np.abs(MTIdc), label='MTIdc')
axes[2].plot(ranks / 1e3, threshold_MTIdc, label='Umbral MTIdc')
axes[2].set_ylabel('Value')
axes[2].set_title('MTI DC')
axes[2].grid(True)
axes[2].legend()

axes[3].plot(ranks / 1e3, np.abs(diff_MTIdc), label='MTIdc')
axes[3].set_xlabel('Range [km]')
axes[3].set_ylabel('Value')
axes[3].set_title('MTI DC Signal')
axes[3].grid(True)
axes[3].legend()

plt.tight_layout()
plt.show()

#%% Doppler

MTIsc_completa = np.zeros_like(radar_signal)


for t in range(gap, len(ranks) - gap):
    MTIsc_completa[:, t] = (compressed_signal[:, t] - compressed_signal[:, t - 1])

MTIsc_transp=(MTIsc_completa).T

expon = np.exp(-1j*2*np.pi*np.outer(np.arange(1,Np+1),np.arange(1,Np+1).T)/(Np+1))

prod=(MTIsc_transp@expon).T

vel=np.linspace(vu_ms/2, -vu_ms/2,Np)

X = vel
Y = ranks/1e3
X, Y = np.meshgrid(X, Y)
Z = (np.abs(fftshift(prod,axes=0))).T

# Crear la figura y los ejes 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar en 3D
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

# Etiquetas y título
ax.set_xlabel('Velocity [m/s]')
ax.set_ylabel('Range [Km]')
ax.set_zlabel('Doppler Product')
ax.set_title('Producto Doppler')
ax.grid(True)

# Mostrar el gráfico
plt.show()

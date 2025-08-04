# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 15:17:15 2023

@author: matveyenko
"""
import numpy as np
import tempfile
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from audiotsm import wsola
from audiotsm.io.wav import WavReader, WavWriter
from scipy.signal import resample, lfilter
import librosa
import librosa as rs
import copy
import scipy
from scipy import signal
from scipy.io import wavfile
from numpy.fft import fft,ifft
import os
import glob
import time
import crepe
import statistics as st
import math
from pydub import AudioSegment

# think about handling other file types 
def in_audio(path, temp_path):

    y, sr = librosa.load(path, sr=44100)
    data = librosa.to_mono(y)
    
    write(temp_path,44100,y)
    sr, audio = wavfile.read(temp_path)
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True,
                                                            model_capacity='medium',
                                                            step_size=85)
    
    print("Median Frequency: ", st.median(frequency))
    print("Mean Frequency: ", st.mean(frequency))
    
    fund_freq = st.median(frequency)

    
    print("sampling frequency: ", sr)
    # plt.figure()
    # plt.plot(data)
    # plt.xlabel('Sample Index')
    # plt.ylabel('Amplitude')
    # plt.title('Waveform of Input Audio')
    # plt.show
    
    # plt.figure()
    # plt.plot(frequency)
    # plt.xlabel('Time')
    # plt.ylabel('Frequency')
    # plt.show
    
    crepe.predict
    
    try:
        os.remove(temp_path)
    except:
        print("exception")
    
    return data, fund_freq 

def freq_to_note(freq):
    notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

    note_number = 12 * math.log2(freq / 440) + 49  
    note_number_round = round(note_number)
        
    note = (note_number_round - 1 ) % len(notes)
    note = notes[note]
    
    octave = (note_number_round + 8 ) // len(notes)
    
    return note, octave, note_number


# vocal tract length normalization
def vtln(x, coef = 0.):
  # STFT
  mag, phase = rs.magphase(rs.core.stft(x))
  mag, phase = np.log(mag).T, phase.T

  # Frequency
  freq = np.linspace(0, np.pi, mag.shape[1]) 
  freq_warped = freq + 2.0 * np.arctan(coef * np.sin(freq) / (1 - coef * np.cos(freq)))
  
  # Warping
  mag_warped = np.zeros(mag.shape, dtype = mag.dtype)
  for t in range(mag.shape[0]):
    mag_warped[t, :] = np.interp(freq, freq_warped, mag[t, :])

  # ISTFT
  y = np.real(rs.core.istft(np.exp(mag_warped).T * phase.T)).astype(x.dtype)

  return y

# chorus effect
def chorus(x, amt, dpth, orig):
    
    coef = dpth*0.2
    coef = max(0., coef)
    
    tot_sig = 2*amt + orig
    
    xp, xo, xm = vtln(x, coef), vtln(x, 0.), vtln(x, - coef)

    return (amt*xp + orig*xo + amt*xm) / tot_sig

def resampling(x, path, mix, coef = 1., fs = 16000):
    path2 = str(path.split(".")[0]+"2.wav")
    path3 = str(path.split(".")[0]+"3.wav")
    path4 = str(path.split(".")[0]+"4.wav")
    
    mix = max(0, mix)
    mix = min(1, mix)
    
    o_mix = 1 - mix
    
    write(path, 44100, x)
    data, samplerate = sf.read(path)
    sf.write(path3, data, fs, "PCM_16")    
    
    with WavReader(path3) as fr:
        with WavWriter(path2, fr.channels, fr.samplerate) as fw:
            tsm = wsola(channels = fr.channels, speed = coef, frame_length = 256, synthesis_hop = int(fr.samplerate / 70.0))
            tsm.run(fr, fw)
    y = resample(librosa.load(path2)[0], len(x)).astype(x.dtype)
    sf.write(path4, y, 44100, "PCM_16")
    y, sr = librosa.load(path4, sr=44100)
    
    try:
        os.remove(path2)
        os.remove(path3)
        os.remove(path4)
        os.remove(path)
    except:
        print("except")
    
    return o_mix*x + mix*y

# Mcadams transformation: Baseline2 of VoicePrivacy2020
def vp_baseline2(x, mcadams = 0.8, winlen = int(20 * 0.001 * 16000), shift = int(10 * 0.001 * 16000), lp_order = 20):
  eps = np.finfo(np.float32).eps
  x2 = copy.deepcopy(x) + eps
  length_x = len(x2)
  
  # FFT parameters
  # n_fft = 2**(np.ceil((np.log2(winlen)))).astype(int)
  wPR = np.hanning(winlen)
  K = np.sum(wPR)/shift
  win = np.sqrt(wPR/K)
  n_frame = 1+np.floor((length_x-winlen)/shift).astype(int) # nr of complete frames
  
  # carry out the overlap - add FFT processing
  y = np.zeros([length_x])

  for m in np.arange(1, n_frame):
    # indices of the mth frame
    index = np.arange(m*shift,np.minimum(m*shift+winlen,length_x))    
    # windowed mth frame (other than rectangular window)
    frame = x2[index]*win 
    # get lpc coefficients
    a_lpc = rs.lpc(frame+eps,lp_order)
    # get poles
    poles = scipy.signal.tf2zpk(np.array([1]), a_lpc)[1]
    #index of imaginary poles
    ind_imag = np.where(np.isreal(poles)==False)[0]
    #index of first imaginary poles
    ind_imag_con = ind_imag[np.arange(0,np.size(ind_imag),2)]
    
    # here we define the new angles of the poles, shifted accordingly to the mcadams coefficient
    # values >1 expand the spectrum, while values <1 constract it for angles>1
    # values >1 constract the spectrum, while values <1 expand it for angles<1
    # the choice of this value is strongly linked to the number of lpc coefficients
    # a bigger lpc coefficients number constraints the effect of the coefficient to very small variations
    # a smaller lpc coefficients number allows for a bigger flexibility
    new_angles = np.angle(poles[ind_imag_con])**mcadams
    
    # make sure new angles stay between 0 and pi
    new_angles[np.where(new_angles>=np.pi)] = np.pi        
    new_angles[np.where(new_angles<=0)] = 0  
    
    # copy of the original poles to be adjusted with the new angles
    new_poles = poles
    for k in np.arange(np.size(ind_imag_con)):
      # compute new poles with the same magnitued and new angles
      new_poles[ind_imag_con[k]] = np.abs(poles[ind_imag_con[k]])*np.exp(1j*new_angles[k])
      # applied also to the conjugate pole
      new_poles[ind_imag_con[k]+1] = np.abs(poles[ind_imag_con[k]+1])*np.exp(-1j*new_angles[k])
        
    # recover new, modified lpc coefficients
    a_lpc_new = np.real(np.poly(new_poles))
    # get residual excitation for reconstruction
    res = lfilter(a_lpc,np.array(1),frame)
    # reconstruct frames with new lpc coefficient
    frame_rec = lfilter(np.array([1]),a_lpc_new,res)
    frame_rec = frame_rec*win    
    
    outindex = np.arange(m*shift,m*shift+len(frame_rec))
    # overlap add
    y[outindex] = y[outindex] + frame_rec
      
  y = y/np.max(np.abs(y))
  return y.astype(x.dtype)


def _trajectory_smoothing(x, thresh = 0.5):
  y = copy.copy(x)

  b, a = signal.butter(2, thresh)
  for d in range(y.shape[1]):
    y[:, d] = signal.filtfilt(b, a, y[:, d])
    y[:, d] = signal.filtfilt(b, a, y[::-1, d])[::-1]

  return y

# modulation spectrum smoothing
def modspec_smoothing(x, mix, path, coef = 0.1):
  # STFT
  
  mix = min(1, mix)
  mix = max(0, mix)
  o_mix = 1-mix
  
  mag_x, phase_x = rs.magphase(rs.core.stft(x))
  mag_x, phase_x = np.log(mag_x).T, phase_x.T
  mag_x_smoothed = _trajectory_smoothing(mag_x, coef)

  # ISTFT
  y = np.real(rs.core.istft(np.exp(mag_x_smoothed).T * phase_x.T)).astype(x.dtype)
  y = y * np.sqrt(np.sum(x * x)) / np.sqrt(np.sum(y * y))
  
  write(path, 44100, y)
  y, fs = librosa.load(path, sr = 44100)
  try:
      os.remove(path)
  except:
      print("exception")
  
  if len(x) != len(y):
      if abs(len(x)-len(y)) <= 0.02*max(len(x),len(y)):
          new_len = min(len(x),len(y))
          x = x[0:new_len]
          y = y[0:new_len]
          print("trimming x")
          return mix*y + o_mix*x
      
      else: return y
  else: return mix*y + o_mix*x
  


# waveform clipping
def clipping(x, thresh):
  
  thresh = min(1, thresh)
  thresh = max(0, thresh)
  thresh = 1 - (thresh/2)   
  
  hist, bins = np.histogram(np.abs(x), 1000)
  hist = np.cumsum(hist)
  abs_thresh = bins[np.where(hist >= min(max(0., thresh), 1.) * np.amax(hist))[0][0]]

  y = np.clip(x, - abs_thresh, abs_thresh)
  y = y * np.divide(np.sqrt(np.sum(x * x)), np.sqrt(np.sum(y * y)), out=np.zeros_like(np.sqrt(np.sum(x * x))), where=np.sqrt(np.sum(y * y))!=0)

  return y

def pitch_shft_calc(fund_freq, pitch_amt = 0, def_freq = 125):
    print("Fundamental Frequency = ",int(fund_freq)," HZ")
    print("Reference Frequency = ",def_freq," HZ")
    
    if fund_freq>def_freq:
        new_freq = fund_freq - (fund_freq - def_freq)/3
    elif fund_freq<def_freq:
        new_freq = fund_freq + (def_freq - fund_freq)/3
    else: new_freq = new_freq
    
    new_freq = new_freq + pitch_amt*5
    
    print("New Frequency = ",new_freq, " HZ")
    
    note1,num1,ab_no1 = freq_to_note(fund_freq)
    note2,num2,ab_no2 = freq_to_note(new_freq)
    
    shift = ab_no2 - ab_no1
    if abs(shift<=2.5):
        if shift<=0:
            shift = -2.7
        else:
            shift = 2.7    
    print(note1,num1, " -> ", note2,num2)
    print("Shift = ",shift)
    
    return shift
    
def normalize(sig, norm_amt):
    """
    Normalize the signal given a certain technique (peak or rms).
    Args:
        - infile    (str) : input filename/path.
        - rms_level (int) : rms level in dB.
    """
    norm_amt = min(1, norm_amt)
    norm_amt = max(0, norm_amt)
    
    rms_level = -18 + (norm_amt*8)
    # linear rms level and scaling factor
    r = 10**(rms_level / 10.0)
    a = np.sqrt( (len(sig) * r**2) / np.sum(sig**2) )

    # normalize
    y = sig * a

    # export data to file
    return y

def anon(x, chorus_amt, chorus_dpth, chorus_orig, path, res_mix, mcadams_mix, mcadams_coef,
         clip_amt, phase_mix, fund_freq, pitch_amt, def_freq, mcadams_yn, normalize_yn, norm_amt):

    mcadams_mix = min(1, mcadams_mix)
    mcadams_mix = max(0, mcadams_mix)
    o_mix = 1 - mcadams_mix    
    
    shft = pitch_shft_calc(fund_freq, pitch_amt, def_freq)

    if normalize_yn:
        x = normalize(x,norm_amt)
    if mcadams_yn:
        x = librosa.effects.pitch_shift(mcadams_mix*vp_baseline2(x,mcadams = mcadams_coef) + o_mix*x,44100,shft)
    else:
        x = librosa.effects.pitch_shift(x,44100,shft)
    x = chorus(x, chorus_amt, chorus_dpth, chorus_orig)
    x = modspec_smoothing(x, phase_mix, path)
    x = resampling(x, path, res_mix)
    x = clipping(x, clip_amt)
    

    return x

def out_audio(x, path_mp3, path_wav, out_type = "wav"):
    
    # fig, (ax1, ax2) = plt.subplots(2)
    # fig.suptitle('Before and After processing')
    # ax1.plot(orig)
    # ax2.plot(x)
    write(path_wav, 44100, x)
    if out_type == "wav":
        return(path_wav)
    if out_type == "mp3":
        AudioSegment.from_wav(path_wav).export(path_mp3, format="mp3")
        try:
            os.remove(path_wav)
        except:
            print("couldn't remove wav")
        return(path_mp3)
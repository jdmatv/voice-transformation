from fastapi import FastAPI, Body, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
import pandas as pd
import math
import os
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
import glob
import time
import crepe
import statistics as st
import random
from pydub import AudioSegment

from library.lib1 import in_audio, anon, out_audio


app = FastAPI()
list_of_usernames = list()
templates = Jinja2Templates(directory="htmldirectory")

@app.get("/home",response_class=HTMLResponse)
def write_home(request: Request):
    return templates.TemplateResponse("home.html",{"request": request})

@app.post("/submitform")
async def handle_form(norm: str = Form(...),clip: str = Form(...), pitch: str = Form(...), chorus_dpth: str = Form(...),
                    chorus_mix: str = Form(...), resample: str = Form(...), phase: str = Form(...),
                    mcadams: str = Form(...), fast: str = Form(...), out: str = Form(...),
                    in_aud: UploadFile = Form(...)):
    

    temp_path = str(str(os.getcwd())+'o_'+str(time.time()).split(".")[0]+'_temp.wav')
    o_path_wav = str(str(os.getcwd())+'o_'+str(time.time()).split(".")[0]+'.wav')
    o_path_mp3 = str(str(os.getcwd())+'o_'+str(time.time()).split(".")[0]+'.mp3')
    content_in_aud = await in_aud.read()
    pref_out = out
    chorus_orig = (1 - (chorus_mix)/100)
    chorus_amt = (1 - chorus_orig) / 2
    chorus_dpth = chorus_dpth/100
    norm_amt = norm/100
    clip_amt = clip/100
    normalize_yn = True
    phase_mix = phase/100
    resample_mix = resample/100
    def_freq = 125 #hz
    pitch_amt = pitch/100
    
    if out == "wav":
        aud_type = "audio/wav"
        aud_name = "anon_audio.wav"
    elif out == "mp3":
        aud_type = "audio/mpeg"
        aud_name = "anon_audio.mp3"

    # mcadams
    mcadams_yn = True
    num1 = 0
    if mcadams == "7":
        mcadams_coef = 0.7
    elif mcadams == "8":
        mcadams_coef = 0.8
    elif mcadams == "9":
        mcadams_coef = 0.9
    elif mcadams == "1":
        mcadams_coef = 1.1
    elif mcadams == "2":
        mcadams_coef = 1.2
    elif mcadams == "3":
        mcadams_coef = 1.3
    elif mcadams == "none":
        mcadams_yn = False
        mcadams_coef = 1
    elif mcadams == "rand":
        num1 = random.randint(1,2)
    elif mcadams == "lower":
        num1 = 1
    elif mcadams == "higher":
        num1 = 2

    if num1 > 0:
        if num1 == 1:
            mcadams_coef = np.random.normal(1.2, 0.04)
        elif num1 == 2:
            mcadams_coef = np.random.normal(0.8,0.04)

    mcadams_mix = 1

    if mcadams_coef > 1:
        pitch_amt = pitch_amt + 0.3
    elif mcadams_coef < 1:
        pitch_amt = pitch_amt - 0.3

    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except: print("couldn't remove file")
    
    if os.path.exists(o_path_mp3):
        try:
            os.remove(o_path_mp3)
        except: print("couldn't remove file")

    if os.path.exists(o_path_wav):
        try:
            os.remove(o_path_wav)
        except: print("couldn't remove file")

    i_aud, fund_freq = in_audio(in_aud, temp_path)
    io_aud = anon(i_aud, chorus_amt, chorus_dpth, chorus_orig, temp_path,
              resample_mix, mcadams_mix, mcadams_coef, clip_amt, phase_mix, fund_freq,
              pitch_amt, def_freq, mcadams_yn, normalize_yn, norm_amt)
    
    out_file_path = out_audio(io_aud,o_path_mp3, o_path_wav,out)

    if os.path.exists(out_file_path):
        return FileResponse(out_file_path, media_type = aud_type, filename=aud_name)
    else:
        print("file doesn't exist")
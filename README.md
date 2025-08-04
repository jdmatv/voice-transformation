# Voice Transformation and Anonymization Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9-brightgreen.svg)](https://www.python.org/)
[![Framework: FastAPI](https://img.shields.io/badge/Framework-FastAPI-teal.svg)](https://fastapi.tiangolo.com/)

This repository contains a containerized web application for real-time voice transformation and anonymization. The tool provides a user-friendly interface to apply a chain of digital signal processing (DSP) effects to an input audio file, aiming to obscure the speaker's identity while preserving speech intelligibility.

## Overview

The Voice Transformation Toolkit is a web-based application built with FastAPI that allows users to upload an audio file and apply a variety of transformations. The core of the application is a sophisticated audio processing pipeline that leverages libraries like `librosa`, `crepe`, and `audiotsm`. Users can adjust parameters for pitch shifting, formant modification (via McAdams coefficient), chorus effects, resampling, and more through a simple HTML interface. The entire application is containerized with Docker for easy deployment and use.

## Key Features

-   **Web-Based Interface**: An intuitive HTML form allows users to upload an audio file and fine-tune a wide range of audio transformation parameters using sliders and dropdowns.
-   **Containerized Deployment**: Packaged with Docker for simple, one-command setup and portability.
-   **Multi-Stage Audio Processing Pipeline**: Applies a chain of effects to effectively anonymize a speaker's voice:
    -   **Pitch Analysis**: Uses the CREPE model to accurately estimate the fundamental frequency (F0) of the speaker's voice.
    -   **Pitch Shifting**: Intelligently shifts the pitch to a new baseline, altering a key vocal biometric.
    -   **Formant Modification**: Implements the McAdams coefficient transformation to alter the vocal tract characteristics, another critical component of voice identity.
    -   **Creative Effects**: Adds layers of chorus, phase shifting, and resampling to further mask the original voice timbre.
    -   **Audio Conditioning**: Includes standard audio processing tools like volume normalization and clipping.
-   **Configurable Parameters**: Most effects are highly configurable through the UI, allowing for a wide range of creative and anonymization outcomes.
-   **Multiple Output Formats**: Supports both `.wav` and `.mp3` as output formats.

## How the Anonymization Works

The application processes audio through a pipeline of distinct DSP modules, each targeting a different aspect of vocal identity.

1.  **Audio Ingestion (`library/lib1.py:in_audio`)**: The user uploads an audio file via the web interface. The backend loads the file using `librosa` and converts it to a standard format (44.1kHz mono).
2.  **Fundamental Frequency (F0) Estimation**: The CREPE deep learning model is used to analyze the audio and determine its fundamental frequency, which is a crucial baseline for pitch shifting.
3.  **Core Anonymization (`library/lib1.py:anon`)**: This is the main processing function that applies the chain of effects in sequence:
    -   **Pitch Shifting**: The original F0 is shifted towards a new target frequency. The amount of shift is determined by the original pitch and user-defined parameters.
    -   **McAdams Transformation**: If enabled, this formant-shifting technique is applied to alter the spectral envelope of the voice, which is a key identifier of a speaker's vocal tract shape.
    -   **Chorus Effect**: A chorus effect is added to thicken the sound and obscure the original timbre.
    -   **Modulation Spectrum Smoothing**: A phase shifting effect is applied to further alter the vocal quality.
    -   **Resampling & Time-Stretching**: The audio is resampled and stretched using the WSOLA algorithm to introduce subtle temporal artifacts.
    -   **Clipping & Normalization**: Standard audio cleanup and leveling is applied to the final signal.
4.  **Output Generation (`library/lib1.py:out_audio`)**: The final processed audio is saved to the chosen format (`.wav` or `.mp3`) and served back to the user for download.

## Getting Started with Docker

The easiest way to run the Voice Transformation Toolkit is with Docker.

### Prerequisites

-   [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

### Deployment

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/jdmatv/jdmatv-voice-transformation.git](https://github.com/jdmatv/jdmatv-voice-transformation.git)
    cd jdmatv-voice-transformation
    ```

2.  **Build the Docker image:**
    From the root directory of the project, run the build command.
    ```bash
    docker build -t voice-transformer .
    ```

3.  **Run the Docker container:**
    This command starts the application and maps port 8080 from the container to port 8080 on your local machine.
    ```bash
    docker run -d -p 8080:8080 --name voice-transformer-app voice-transformer
    ```

4.  **Access the Application:**
    Open your web browser and navigate to: **`http://localhost:8080/home`**

## Usage Guide

1.  Navigate to the application's home page in your browser.
2.  Use the sliders to adjust the parameters for each audio effect.
3.  Select the desired McAdams coefficient for formant shifting.
4.  Choose your preferred output format (`.wav` or `.mp3`).
5.  Use the "Choose File" button to upload an audio file.
6.  Click **Submit**. The backend will perform the audio processing.
7.  Once complete, the transformed audio file will be automatically downloaded by your browser.

## Application Architecture

-   **Backend**: A high-performance web service built with **FastAPI**.
-   **Frontend**: A simple HTML form served via **Jinja2** templates.
-   **Core Logic**: A modular `library` of Python scripts handling all audio processing and DSP effects.
-   **Server**: The application is served using **Uvicorn**, an ASGI server.
-   **Deployment**: The entire application is packaged into a **Docker** container for portability and ease of use.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

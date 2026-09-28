# Hindi to Santhali AI Translator

AI-powered Hindi to Santhali speech translation system.

## Requirements

- Python 3.10+
- Git
- Internet connection for installing packages and downloading models

## Installation

Clone the repository:

git clone https://github.com/abithataslimR/HINDI-TO-SANTHALI-.git

cd HINDI-TO-SANTHALI-

Create a virtual environment:

python3 -m venv venv

Activate it:

source venv/bin/activate

Install dependencies:

pip install -r requirements-deploy.txt

## Run

Start the translation server:

python translate_server.py

The server will run on:

http://127.0.0.1:5000

## Important

The AI models are not included in this repository.

The Vosk Hindi speech recognition model, IndicTrans2 translation model, and Indic Parler-TTS model must be downloaded separately.

# AI Voice Calculator

A voice-activated calculator with speech recognition and text-to-speech.

## Features
- Voice-activated with "calculator" wake word
- Supports compound numbers ("twenty five")
- Real-time speech transcription
- Automatic silence detection
- Calculation history with export
- Color-coded visual feedback

## Installation

### Requirements
- Python 3.10+
- Vosk speech model (download separately)

### Setup
```bash
pip install -r requirements.txt
# Download Vosk model and extract to 'india-english' folder
python voice_calculator.py
```

## Usage
1. Click "Start AI"
2. Say "calculator" followed by your calculation
3. Example: "calculator twenty five plus ten"
4. Wait for result or press Stop AI

## Supported Operations
- Addition: "plus", "+"
- Subtraction: "minus", "-"
- Multiplication: "times", "multiplied by", "x"
- Division: "divided by", "over"
- Power: "to the power of", "power"
- Modulo: "mod", "remainder"

## Troubleshooting
- **No speech detected**: Check microphone permissions
- **Model not found**: Ensure Vosk model is in correct folder
- **TTS not working**: Check audio output settings

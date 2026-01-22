# AI Audiobook Generator

A local, privacy-focused audiobook generator that converts PDFs and EPUBs into high-quality audio using Coqui TTS.

## Features
- **PDF/EPUB Extraction**: Automatically extracts and structures text.
- **smart Cleaning**: Tools to fix paragraph breaks, special characters, and phantom spaces.
- **Voice Cloning**: Use a custom trained model or reference audio for voice cloning.
- **Video Generation**: Create a video with a static image for uploading to platforms like YouTube.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need to handle system dependencies for `tts` and `ffmpeg` separately depending on your OS.*

2.  **Run the App**:
    ```bash
    python3 gui_app.py
    ```

## Project Structure
- `inputs/`: Place your source PDFs here (or browse in app).
- `models/`: Contains the custom trained TTS model.
- `audio_output/`: Generated audio files.
- `video/`: Generated video files.

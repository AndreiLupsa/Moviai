# Moviai

**A distributed pipeline for autonomous, long-form cinematic AI storytelling.**

Moviai orchestrates multiple state-of-the-art generative models (LLMs, Image, Video, and Audio) via FastAPI to write, storyboard, animate, and score complete short films from a single text prompt.

## Architecture & Core Features

- **Fluid Scene Interpolation:** Uses text-to-image models to generate keyframes, which are fed into video models as first/last frame conditions to prevent sharp cuts and create seamless transitions.
- **Producer-Consumer Audio Pipeline:** Asynchronously pulls silent video chunks from a queue to apply contextual, video-to-video audio generation, drastically reducing total pipeline execution time.
- **Serverless Orchestration:** Utilizes the fal.ai API to distribute heavy generative workloads across cloud GPUs, allowing multiple scene components to be processed concurrently.

## Tech Stack

- **Backend:** Python (asyncio), FastAPI, Uvicorn
- **AI Orchestration:** fal.ai API
- **Generative Models:** FLUX/Imagen4 (Image), Wan/LTX/PixVerse (Video), MMAudio-v2 (Audio), LLMs (Plot/Prompts)
- **Media Processing:** MoviePy, FFmpeg
- **Frontend:** HTML/CSS, Jinja2 Templates

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/AndreiLupsa/Moviai.git
   cd Moviai
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Export your fal.ai API key as an environment variable:
   ```bash
   export FAL_KEY="your_api_key_here"
   ```

## Usage

Start the FastAPI server to launch the application:

```bash
uvicorn main:app --reload
```

Once the server is running, navigate to `http://localhost:8000` in your browser to access the web interface, enter your text prompt, and configure your generation parameters.

## About

This project was developed as part of a university dissertation researching the application of distributed systems and generative AI to overcome the technical limitations of short-form video synthesis.

import os
import platform
import asyncio
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from main import generate_movie_pipeline

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI(title="Moviai")

app.mount("/static", StaticFiles(directory="static"), name="static")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


FINAL_VIDEOS_DIR = "final_videos"
os.makedirs(FINAL_VIDEOS_DIR, exist_ok=True)
app.mount("/final_videos", StaticFiles(directory="final_videos"), name="final_videos")

templates = Jinja2Templates(directory="templates")

IMAGE_MODELS = [
    "fal-ai/flux/schnell",
    "fal-ai/imagen4/preview",
    "fal-ai/imagen4/preview/fast",
    "fal-ai/flux/dev",
]

VIDEO_MODELS = [
    "fal-ai/ltx-video-13b-distilled/multiconditioning",
    "fal-ai/pixverse/v4.5/transition",
    "fal-ai/wan-flf2v",
    "fal-ai/ltx-video-13b-dev/multiconditioning",
]


@app.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "image_models": IMAGE_MODELS,
        "video_models": VIDEO_MODELS,
        "result_video": None
    })


@app.post("/generate")
async def generate_video(
        topic: str = Form(...),
        scene_nr: int = Form(...),
        image_model: str = Form(...),
        video_model: str = Form(...)
):
    final_video_path = await generate_movie_pipeline(topic, scene_nr, image_model, video_model)

    if final_video_path:
        video_filename = os.path.basename(final_video_path)
        return RedirectResponse(url=f"/results/{video_filename}", status_code=303)
    else:
        return RedirectResponse(url="/", status_code=303)


@app.get("/results/{video_name}", response_class=HTMLResponse)
async def show_results(request: Request, video_name: str):
    video_url = f"/final_videos/{video_name}"
    return templates.TemplateResponse("index.html", {
        "request": request,
        "image_models": IMAGE_MODELS,
        "video_models": VIDEO_MODELS,
        "result_video": video_url
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

import asyncio
import platform
from pipeline import generate_movie_pipeline

async def run_cli():
    print("--- Moviai Command-Line Interface ---")
    user_topic = input("Enter a topic for the short film: ")
    scene_number = int(input("Enter the number of scenes: "))

    default_image_model = "fal-ai/flux/schnell"
    default_video_model = "fal-ai/ltx-video-13b-distilled/multiconditioning"

    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    await generate_movie_pipeline(user_topic, scene_number, default_image_model, default_video_model)

if __name__ == "__main__":
    asyncio.run(run_cli())

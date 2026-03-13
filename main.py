import asyncio
import datetime
import platform

from Utility.utils import log_message, initialize_log_for_run
from Class.Movie import Movie


async def generate_movie_pipeline(topic: str, scene_nr: int, image_model: str, video_model: str) -> str | None:
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    initialize_log_for_run(timestamp)

    log_message(f"User entered topic: '{topic}'")
    log_message(f"User entered number of scenes: '{scene_nr}'")
    log_message(f"User chose models: image: '{image_model}' / video: {video_model}")

    movie = Movie(topic, scene_nr, timestamp, image_model, video_model)

    final_video_path = await movie.produce()

    if final_video_path:
        log_message("--- Application completed successfully ---")
        log_message(f"Final video output available at: {final_video_path}")
        return final_video_path
    else:
        log_message("--- Application FAILED to produce a video ---")
        return None


if __name__ == "__main__":
    log_message("--- Program Started (Command-Line Mode) ---")

    user_topic = input("Please enter a topic for the short film: ")
    scene_number = int(input("Please enter the number of scenes: "))

    default_image_model = "fal-ai/flux/schnell"
    default_video_model = "fal-ai/ltx-video-13b-distilled/multiconditioning"

    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(generate_movie_pipeline(user_topic, scene_number, default_image_model, default_video_model))

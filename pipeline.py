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



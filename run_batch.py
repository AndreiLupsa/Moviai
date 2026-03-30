import asyncio
from pipeline import generate_movie_pipeline

RUN_CONFIGS = [
    {
        "topic": """A short film in a charming and simple cartoon style, with clean lines and bright, happy colors.

The main characters are a group of cute, simply-drawn Adélie penguins with bold black and white bodies and big, expressive eyes. The group of cute penguins are on a clean, sunny beach with soft yellow sand and simple blue waves.

The simply-drawn Adélie penguins work together, using brightly colored plastic toy buckets and shovels to build a sandcastle. The sandcastle starts small, and the penguins add more towers and walls, making it grow larger. The movie ends showing a gigantic finished sandcastle with a red flag on top, with the happy cartoon Adélie penguins jumping joyfully around it.""",
        "image_model": "fal-ai/imagen4/preview",
        "video_model": "fal-ai/pixverse/v4.5/transition",
        "scene_nr": 6
    },
    # {
    #     "topic": "A polar bear playing with a red soccer ball. the red soccer ball is always red. In each scene. Specify each time that the soccer ball is solid red, no other colors.",
    #     "image_model": "fal-ai/imagen4/preview",
    #     "video_model": "fal-ai/pixverse/v4.5/transition",
    #     "scene_nr": 5
    # },
]


async def main():
    print("--- Starting Bulk Generation Script ---")
    total_runs = len(RUN_CONFIGS)

    for i, config in enumerate(RUN_CONFIGS):
        print("=" * 50)
        print(f"--- Starting Run {i + 1}/{total_runs} ---")
        print(f"Topic: {config['topic']}")
        print("=" * 50)

        try:
            await generate_movie_pipeline(
                topic=config["topic"],
                image_model=config["image_model"],
                video_model=config["video_model"],
                scene_nr=config["scene_nr"]
            )
            print(f"--- Finished Run {i + 1}/{total_runs} Successfully ---")
        except Exception as e:
            print(f"--- FATAL ERROR in Run {i + 1}/{total_runs}: {e} ---")
            print("--- Moving to the next job in the batch. ---")
            continue

    print("--- Bulk Generation Script Finished ---")


if __name__ == "__main__":
    asyncio.run(main())

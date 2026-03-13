import asyncio
import os

import fal_client

from Utility.utils import log_message, download_media_file


class AudioPipeline:

    def __init__(self, scene_chunk_list: list, timestamp: str, chunk_size: int):
        self.scene_chunk_list = scene_chunk_list
        self.timestamp = timestamp
        self.chunk_size = chunk_size

        self.silent_video_chunk_urls = []
        self.silent_video_chunk_paths = []
        self.audio_video_chunk_paths = []

    async def run(self) -> list:
        log_message("--- Starting Main Audio Generation Workflow ---")
        q = asyncio.Queue()

        producer_task = asyncio.create_task(self._produce_silent_chunks(q))
        consumer_task = asyncio.create_task(self._consume_and_add_audio(q))

        await producer_task
        await q.join()

        consumer_task.cancel()

        log_message("--- Main Audio Generation Workflow Completed ---")
        return self.audio_video_chunk_paths

    async def _produce_silent_chunks(self, queue: asyncio.Queue):
        log_message("--- Starting silent video chunk production ---")
        output_dir = os.path.join("data", "silent_chunks", f"silent_chunks_{self.timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        for i, chunk in enumerate(self.scene_chunk_list):
            log_message(f"Preparing to compose silent chunk {i + 1}/{len(self.scene_chunk_list)}...")
            try:
                keyframes = []
                for index, video_url in enumerate(chunk):
                    keyframe = {
                        "url": video_url,
                        "timestamp": index * 5000,
                        "duration": 5000
                    }
                    keyframes.append(keyframe)

                log_message(f"Submitting job to compose silent chunk {i + 1}...")
                handler = await fal_client.submit_async(
                    "fal-ai/ffmpeg-api/compose",
                    arguments={
                        "tracks": [
                            {
                                "id": f"chunk_{i}", "type": "video", "keyframes": keyframes
                            }
                        ]
                    }
                )
                result = await handler.get()
                silent_chunk_url = result.get("video_url")

                if silent_chunk_url:
                    log_message(f"Successfully composed silent chunk {i + 1}.")
                    self.silent_video_chunk_urls.append(silent_chunk_url)

                    local_path = os.path.join(output_dir, f"silent_chunk_{i}.mp4")
                    download_success = await download_media_file(silent_chunk_url, local_path)
                    if download_success:
                        self.silent_video_chunk_paths.append(local_path)

                    await queue.put(silent_chunk_url)
                else:
                    log_message(f"--- ERROR: Composition for chunk {i + 1} did not return a video URL.")

            except Exception as e:
                log_message(f"--- FATAL ERROR during silent chunk composition for chunk {i + 1}: {e}")

        log_message("--- All silent chunk production tasks have been dispatched. ---")
        await queue.put(None)

    async def _consume_and_add_audio(self, queue: asyncio.Queue):
        log_message("--- Starting audio addition consumer ---")
        chunk_index = 0
        output_dir = os.path.join("data", "audio_video_chunks", f"av_chunks_{self.timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        while True:
            silent_chunk_url = await queue.get()
            if silent_chunk_url is None:
                log_message("--- Consumer received signal to stop. ---")
                queue.task_done()
                break

            log_message(f"Consumer picked up silent chunk {chunk_index + 1} for audio processing.")

            audio_video_url = None
            for attempt in range(3):
                try:
                    log_message(f"Submitting job to add audio to chunk {chunk_index + 1}...")
                    handler = await fal_client.submit_async(
                        "fal-ai/mmaudio-v2",
                        arguments={
                            "video_url": silent_chunk_url,
                            "prompt": "A rich, high-fidelity instrumental music soundtrack that drives the scene and perfectly matches its mood, tone, and pacing. Key diegetic sound effects from the video are present but are subtly integrated and mixed cleanly with the music. The overall audio is coherent, melodic, and emotionally resonant",
                            "negative_prompt": "human voice, speech, talking, spoken word, dialogue, narration, commentary, vocals, vocalizations, phonetic sounds, singing, a cappella, gibberish, chatter, murmuring, babble, background conversation, crowd noise, clicking, clatter, crackling, static, hiss, hum, white noise, chaotic sounds, jumbled noise, incoherent sounds, random noise, crescendo, muffled, distorted",
                            "num_steps": 50,
                            "duration": self.chunk_size * 5 + 2,
                            "cfg_strength": 5,
                        }
                    )
                    result = await handler.get()
                    url = result.get("video", {}).get("url")

                    if url:
                        audio_video_url = url
                        log_message(f"Successfully added audio to chunk {chunk_index + 1}.")
                        break
                    else:
                        log_message(f"--- WARNING: Audio generation for chunk {chunk_index + 1} did not return a URL.")

                except Exception as e:
                    log_message(f"--- WARNING: Audio processing for chunk {chunk_index + 1}, attempt {attempt + 1} failed. Reason: {e}")

                if attempt < 2:
                    await asyncio.sleep(2)
                else:
                    log_message(f"--- ERROR: Audio generation for chunk {chunk_index + 1} did not return a video URL.")

            if audio_video_url:
                local_path = os.path.join(output_dir, f"av_chunk_{chunk_index}.mp4")
                download_success = await download_media_file(audio_video_url, local_path)
                if download_success:
                    self.audio_video_chunk_paths.append(local_path)

            else:
                log_message(f"--- ERROR: Could not generate audio for chunk {chunk_index + 1} after 3 attempts.")

            chunk_index += 1
            queue.task_done()

        log_message("--- Audio addition consumer has finished all tasks. ---")

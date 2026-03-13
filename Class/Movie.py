import asyncio
import json
import os
import time

from Utility.combine import concatenate_videos
from Utility.utils import log_message
from Utility.plot_generator import generate_plot_from_topic
from Class.Scene import Scene
from Class.AudioPipeline import AudioPipeline


class Movie:
    def __init__(self, topic: str, scene_nr: int, timestamp: str, image_model: str, video_model: str):
        self.topic = topic
        self.scene_nr = scene_nr
        self.timestamp = timestamp
        self.scenes = []
        self.image_urls = {}
        self.video_urls = {}
        self.initial_model = "openai/gpt-4o"

        self.fixer_models = ["meta-llama/llama-4-maverick",
                             "google/gemini-flash-1.5",
                             "meta-llama/llama-3.1-70b-instruct",
                             "openai/gpt-4o-mini"]

        self.enricher_model = "meta-llama/llama-4-maverick"
        self.image_model = image_model
        self.video_model = video_model
        self.chunk_size = 4

    async def produce(self):
        start_time = time.monotonic()

        await self._create_basic_plot()
        if not self.scenes:
            log_message("--- Movie production halted: No scenes were created. ---")
            return None

        await self._enrich_plot()
        self._link_scene_prompts()
        await self._generate_all_images()
        await self._generate_all_videos()

        if not self.video_urls:
            log_message("--- No videos were generated, skipping audio and final combination. ---")
            return None

        log_message("--- Handing off to Audio Generation Pipeline ---")
        scene_chunk_list = self._group_urls()
        if not scene_chunk_list:
            log_message("--- ERROR: Could not group video URLs for audio processing. ---")
            return None

        pipeline = AudioPipeline(scene_chunk_list, self.timestamp, self.chunk_size)
        audio_video_chunk_paths = await pipeline.run()

        if audio_video_chunk_paths:
            log_message("--- Starting Final Combination of Audio-Added Chunks ---")
            final_output_path = self._make_final_output_path()
            audio_chunks_folder = os.path.dirname(audio_video_chunk_paths[0])

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                concatenate_videos,
                audio_chunks_folder,
                final_output_path
            )

            end_time = time.monotonic()
            elapsed_seconds = end_time - start_time
            self._write_metadata(elapsed_seconds)

            return final_output_path
        else:
            log_message("--- ERROR: Audio generation pipeline did not produce any final chunks. ---")
            return None

    async def _create_basic_plot(self):
        log_message("--- Basic Plot Creation Phase Started ---")
        loop = asyncio.get_running_loop()
        try:
            basic_plot_data = await loop.run_in_executor(
                None,
                generate_plot_from_topic,
                self.topic,
                self.initial_model,
                self.fixer_models,
                self.scene_nr
            )
            if basic_plot_data:
                self.scenes = [Scene(scene_data) for scene_data in basic_plot_data]
                actual_scene_nr = len(self.scenes)
                if self.scene_nr != actual_scene_nr:
                    log_message(
                        f"--- WARNING: LLM generated {actual_scene_nr} scenes, not the {self.scene_nr} requested. Adapting.")
                    self.scene_nr = actual_scene_nr

                # self.chunk_size = 3 if self.scene_nr == 5 or self.scene_nr == 9 else 4
                log_message(f"Chunk size: {self.chunk_size}")
                log_message("--- Basic Plot Creation Phase Completed ---")

            else:
                log_message("--- Basic Plot Creation Phase FAILED ---")
        except Exception as e:
            log_message(f"--- ERROR in thread during plot creation: {e}")

    async def _enrich_plot(self):
        log_message("--- Concurrent Prompt Enrichment Phase Started ---")
        log_message(f"Enriching scenes with model: {self.enricher_model}")
        tasks = [scene.enrich(self.scenes, self.enricher_model) for scene in self.scenes]
        await asyncio.gather(*tasks)
        log_message("--- Concurrent Prompt Enrichment Phase Completed ---")

    def _log_enriched_plot(self):
        log_message("Final Enriched Project Data:")
        if self.scenes and all(scene.status == "ENRICHED" for scene in self.scenes):
            data = [vars(scene) for scene in self.scenes]
            pretty_json = json.dumps(data, indent=4, ensure_ascii=False)
            log_message("\n" + pretty_json)

    def _link_scene_prompts(self):
        log_message("--- Linking Scene Prompts for Continuity ---")
        for i in range(1, len(self.scenes)):
            current_scene = self.scenes[i]
            previous_scene = self.scenes[i - 1]

            if previous_scene.prompts and 'image_end' in previous_scene.prompts and previous_scene.prompts['image_end']:
                current_scene.prompts['image_start'] = previous_scene.prompts['image_end']
                log_message(
                    f"Linked Scene {current_scene.scene_id} start image to Scene {previous_scene.scene_id} end image.")
            else:
                log_message(
                    f"--- WARNING: Could not link Scene {current_scene.scene_id}. Previous scene {previous_scene.scene_id} missing 'image_end' prompt.")
        log_message("--- Scene Prompt Linking Completed ---")
        self._log_enriched_plot()

    async def _generate_all_images(self):
        log_message("--- All Images Generation Phase Started ---")
        output_dir = os.path.join("data", "images", f"images_{self.timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        log_message(f"Image output directory created at: {output_dir}")
        log_message(f"Generating images using model: {self.image_model}")

        scenes_sorted = sorted(self.scenes, key=lambda s: s.scene_id)

        start_image_tasks = [scene.generate_start_image(self.image_model, output_dir) for scene in scenes_sorted]
        generated_start_urls = await asyncio.gather(*start_image_tasks)

        for scene, url in zip(scenes_sorted, generated_start_urls):
            if url:
                self.image_urls[scene.scene_id] = url

        log_message("Generated Start Image URLs and stored them.")

        if self.scenes:
            last_scene = scenes_sorted[-1]
            log_message(f"Generating final_videos end image for the last scene (Scene {last_scene.scene_id})...")
            final_end_url = await last_scene.generate_end_image(self.image_model, output_dir)

            self.image_urls[last_scene.scene_id + 1] = final_end_url

            log_message("Generated Final End Image URL and stored it.")

        log_message("--- All Images Generation Phase Completed ---")

    async def _generate_all_videos(self):
        log_message("--- All Video Generation Phase Started ---")
        output_dir = os.path.join("data", "videos", f"videos_{self.timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        log_message(f"Video output directory created at: {output_dir}")
        log_message(f"Generating videos using model: {self.video_model}")

        scenes_sorted = sorted(self.scenes, key=lambda s: s.scene_id)
        video_tasks = []
        for i, scene in enumerate(scenes_sorted):
            start_image_url = self.image_urls.get(i)
            end_image_url = self.image_urls.get(i + 1)

            if start_image_url and end_image_url:
                task = scene.generate_video(self.video_model, output_dir, start_image_url, end_image_url)
                video_tasks.append(task)
            else:
                log_message(
                    f"--- WARNING: Missing start (key {i}) or end (key {i + 1}) image URL for Scene {i}. Skipping video generation.")

        if not video_tasks:
            log_message("--- No video generation tasks were created. ---")
            return ""

        generated_video_urls = await asyncio.gather(*video_tasks)

        for i, url in enumerate(generated_video_urls):
            scene_id_for_url = scenes_sorted[i].scene_id
            self.video_urls[scene_id_for_url] = url

        log_message("--- Video Generation Phase Completed ---")
        return output_dir

    def _make_final_output_path(self):
        parts = self.image_model.split('/')
        sanitized_image_model = "-".join(parts[1:])

        parts = self.video_model.split('/')
        sanitized_video_model = "ltx-distilled" if "distilled" in parts[1] else "ltx-dev" if "dev" in parts[1] else \
            parts[1]

        filename_base = f"{sanitized_video_model}+{sanitized_image_model}"
        output_dir = "final_videos"
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"{filename_base}_{self.timestamp}.mp4")

    def _write_metadata(self, elapsed_seconds):
        minutes, seconds = divmod(elapsed_seconds, 60)
        elapsed_time_str = f"{int(minutes):02}:{int(seconds):02}"

        metadata = {
            "topic": self.topic,
            "nr_of_scenes": self.scene_nr,
            "chunk_size": self.chunk_size,
            "plot_llm": self.initial_model,
            "enricher_llm": self.enricher_model,
            "image_model": self.image_model,
            "video_model": self.video_model,
            "timestamp": self.timestamp,
        }
        # Add model-specific metadata
        if "ltx" in self.video_model:
            metadata["resolution"] = "720p"
            metadata["frames_per_scene"] = 121
            metadata["frame_rate"] = 24
        elif "wan" in self.video_model:
            metadata["resolution"] = "480p"
            metadata["frames_per_scene"] = 81
            metadata["frame_rate"] = 16
        elif "pixverse" in self.video_model:
            metadata["resolution"] = "720p"

        metadata["elapsed_time"] = elapsed_time_str

        metadata_filepath = os.path.join("final_videos", f"meta_{self.timestamp}.txt")
        try:
            with open(metadata_filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
            log_message(f"Successfully saved metadata to {metadata_filepath}")
        except Exception as e:
            log_message(f"--- ERROR: Failed to write metadata file: {e}")

    def _group_urls(self) -> list:
        log_message("--- Grouping video URLs into manageable chunks ---")
        sorted_urls = [url for _, url in sorted(self.video_urls.items()) if url]
        if not sorted_urls:
            log_message("--- WARNING: No video URLs found to group. ---")
            return []
        scene_chunk_list = [sorted_urls[i:i + self.chunk_size] for i in range(0, len(sorted_urls), self.chunk_size)]
        log_message(f"Successfully grouped {len(sorted_urls)} URLs into {len(scene_chunk_list)} chunks.")
        return scene_chunk_list

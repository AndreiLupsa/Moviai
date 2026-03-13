import os

import fal_client
from Utility.utils import log_message, download_media_file


class Scene:
    def __init__(self, scene_data: dict):
        self.scene_id = scene_data['scene_id']
        self.act = scene_data['act']
        self.description = scene_data['description']
        self.prompts = {}
        self.status = "PENDING_ENRICHMENT"

    async def enrich(self, full_plot_context: list, model: str):
        log_message(f"Starting enrichment for Scene {self.scene_id}: '{self.act}'")
        try:
            story_context = "\n".join(
                [f"Scene {s.scene_id} ({s.act}): {s.description}" for s in full_plot_context])

            video_prompt_task = f"""
            You are a prompt engineer specialised in prompting AI video generation models.
            You will receive a short story and are required to generate a prompt fitting for a video generation model, for one specific scene from it.
            Your goal is to ensure VISUAL accuracy between the video prompt and the scene description,
            and NARRATIVE CONTINUITY between the actions described in the prompt and the story plot as a whole.
            You should keep the prompt clear and concise, 2-3 sentences.
            You should write as response just the prompt text, without any titles, headings, bullet points, markdowns or other formatting.
            Here is the full story context:
            ---
            {story_context}
            ---
            Identify in the story description the general Style/Aesthetic of the video (e.g. 'gritty and dark', 'dreamy and soft').
            Use this style description consistently in the video prompt, in order to ensure a better visual consistency.

            You are generating the detailed video prompt for SCENE {self.scene_id} ONLY, titled '{self.act}'.
            Just to recap, the basic description for this scene is: "{self.description}"
            You have to describe this scene as a prompt that would be fitting for a video generation model.

            CRUCIAL FOR VISUAL CONTINUITY: For every character and significant object mentioned (e.g. a rebel, security forces, a can of pepsi),
            you MUST re-state their defining visual characteristics in this prompt
            (e.g., 'the young rebel in their worn-out jacket with a torn-off Coke logo,'
            'Coke security forces in their black uniforms and helmets', 'a rusted can of pepsi).
            Do this for every scene to ensure the stateless video generator creates a consistent output.
            Do not assume the generator remembers details from previous scenes.

            Now, based on all context, write a single, detailed, and cinematic video prompt for the CURRENT scene.
            Your ENTIRE output must be a single paragraph of text only. 2-3 sentences.
            DO NOT use markdown, titles, headings, bullet points, or any other formatting.
            """

            result_video = await fal_client.run_async(
                "fal-ai/any-llm",
                arguments={
                    "model": model,
                    "prompt": video_prompt_task
                }
            )

            video_prompt = result_video['output'].strip()

            image_end_prompt_task = f"""
            You are a prompt engineer specialised in prompting AI image generation models.
            You will receive a short story which serves as context for the job you have to do.
            The story describes the plot for a a short film and is split up into several scenes
            You have to keep in mind the plot, characters, scenes, descriptions of the story.
            Here is the story:
            ---
            {story_context}
            ---
            Identify in the story description the general Style/Aesthetic of the video (e.g. 'gritty and dark', 'dreamy and soft').
            Use this style description consistently in the image prompt, in order to ensure better visual consistency.

            Keeping in mind the whole context, focus on SCENE {self.scene_id}, titled '{self.act}', with description "{self.description}"
            You will receive a detailed text prompt, describing a video sequence, which will be sent to a video generation model, to create this specific scene.
            Your job is to write an image prompt for the specific, single, still image that represents the very last frame of the respective video sequence.
            You should keep the prompt clear and concise, 2-3 sentences.
            You should focus on the attributes relevant to the visual description of the still frame, but keeping in mind as context
            the whole prompt of the video scene, as well as the whole plot of the story.
            Make sure that what you describe for the image is congruent with the video scene and the whole story.
            Write only the text, without any titles, headings, bullet points, markdowns or other formatting.
            This is the detailed prompt for the video sequence:
            ---
            "{video_prompt}"
            ---
            Identify in the scene video prompt the general Style/Aesthetic of the scene (e.g. 'gritty and dark', 'dreamy and soft').
            Combine this style with the general story style and use matching stylistic descriptors in your image prompt,
            in order to ensure better visual consistency.

            Write the prompt for a single, still image that represents the very last frame of this video sequence.

            CRUCIAL FOR VISUAL CONTINUITY: For every character and significant object mentioned (e.g. a rebel, security forces, a can of pepsi),
            you MUST re-state their defining visual characteristics in this prompt
            (e.g., 'the young rebel in their worn-out jacket with a torn-off Coke logo,'
            'Coke security forces in their black uniforms and helmets', 'a rusted can of pepsi).
            Do this for every scene to ensure the stateless image generator creates a consistent output.
            Do not assume the generator remembers details from previous scenes.

            Your ENTIRE output should be just text, descriptive and short, only 2-3 sentences in length.
            Write just the description. No preamble (e.g. 'The last frame describes', 'The image describes').
            DO NOT use markdown, titles, headings, bullet points, or any other formatting.
            """
            result_end_image = await fal_client.run_async(
                "fal-ai/any-llm",
                arguments={
                    "model": model,
                    "prompt": image_end_prompt_task
                }
            )
            image_end_prompt = result_end_image['output'].strip()

            self.prompts = {
                'video': video_prompt,
                'image_start': None,
                'image_end': image_end_prompt,
            }
            if self.scene_id == 0:
                self.prompts['image_start'] = video_prompt

            self.status = "ENRICHED"
            log_message(f"Successfully enriched Scene {self.scene_id}.")
        except Exception as e:
            self.status = "FAILED"
            log_message(f"--- ERROR: Could not enrich Scene {self.scene_id}: {e}")

    async def generate_start_image(self, model: str, output_dir: str) -> str | None:
        start_prompt = self.prompts.get('image_start')
        output_path = os.path.join(output_dir, f"{self.scene_id}.png")
        return await self._generate_image_from_prompt(model, start_prompt, output_path)

    async def generate_end_image(self, model: str, output_dir: str) -> str | None:
        end_prompt = self.prompts.get('image_end')
        output_path = os.path.join(output_dir, f"{self.scene_id + 1}.png")
        return await self._generate_image_from_prompt(model, end_prompt, output_path)

    async def _generate_image_from_prompt(self, model: str, prompt: str, output_path: str) -> str | None:
        if not prompt:
            log_message(f"--- WARNING: Scene {self.scene_id} has no prompt provided. Skipping generation.")
            return None

        if "flux" in model:
            arguments = {
                "prompt": prompt,
                "image_size": {"width": 1280, "height": 704}
            }
        elif "imagen" in model:
            arguments = {
                "prompt": prompt,
                "aspect_ratio": "16:9",
            }
        else:
            log_message(f"--- ERROR: Unsupported video model type: {model}")
            arguments = None

        log_message(f"Submitting job for Scene {self.scene_id} -> {os.path.basename(output_path)}...")
        try:
            handler = await fal_client.submit_async(
                model,
                arguments=arguments
            )

            result = await handler.get()
            image_url = result['images'][0]['url']

            if await download_media_file(image_url, output_path):
                return image_url
            return None

        except Exception as e:
            log_message(f"--- ERROR: Image generation API call failed for Scene {self.scene_id}. Reason: {e}")
            return None

    async def generate_video(self, model: str, output_dir: str, start_image_url: str, end_image_url: str) -> str | None:
        video_prompt = self.prompts.get('video')
        if not all([video_prompt, start_image_url, end_image_url]):
            log_message(
                f"--- WARNING: Scene {self.scene_id} is missing a prompt or keyframe URL. Skipping video generation.")
            return None

        log_message(f"Submitting video job for Scene {self.scene_id}...")
        output_path = os.path.join(output_dir, f"{self.scene_id}.mp4")

        if "ltx" in model:
            arguments = {
                "prompt": video_prompt,
                "images": [
                    {"image_url": start_image_url, "start_frame_num": 0},
                    {"image_url": end_image_url, "start_frame_num": 120}
                ],
                "resolution": "720p",
                "frame_rate": 24,
                "number_of_frames": 120,
            }
        elif "wan" in model:
            arguments = {
                "prompt": video_prompt,
                "start_image_url": start_image_url,
                "end_image_url": end_image_url,
                "resolution": "480p",
                "frames_per_second": 16,
                "num_frames": 81,
            }
        elif "pixverse" in model:
            arguments = {
                "prompt": video_prompt,
                "first_image_url": start_image_url,
                "last_image_url": end_image_url,
                "resolution": "720p"
            }
        else:
            log_message(f"--- ERROR: Unsupported video model type: {model}")
            arguments = None

        try:
            handler = await fal_client.submit_async(
                model,
                arguments=arguments
            )
            log_message(f"Video job submitted for Scene {self.scene_id}")

            result = await handler.get()

            video_url = result['video']['url']
            log_message(f"Video job for Scene {self.scene_id} completed.")

            if await download_media_file(video_url, output_path):
                return video_url
            return None
        except Exception as e:
            log_message(f"--- ERROR: Video generation API call for Scene {self.scene_id} failed. Reason: {e}")
            return None

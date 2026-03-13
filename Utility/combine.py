import os
from Utility.utils import log_message
from moviepy import VideoFileClip, concatenate_videoclips


def concatenate_videos(videos_folder: str, output_path: str):
    log_message(f"Starting video concatenation process for folder: {videos_folder}")

    try:
        video_files = [f for f in os.listdir(videos_folder) if f.endswith('.mp4')]
        if not video_files:
            log_message(f"--- ERROR: No .mp4 files found in '{videos_folder}'. Aborting. ---")
            return

        video_files.sort(key=lambda f: int(f.split('.')[0].split('_')[-1]))

        full_paths = [os.path.join(videos_folder, f) for f in video_files]

        log_message("Found and sorted the following video files:")
        for path in full_paths:
            log_message(f"  - {os.path.basename(path)}")

    except FileNotFoundError:
        log_message(f"--- ERROR: The directory '{videos_folder}' was not found. Please check the path. ---")
        return
    except Exception as e:
        log_message(f"--- ERROR during file discovery: {e} ---")
        return

    try:
        log_message("\nLoading video clips into memory...")
        clips = [VideoFileClip(path) for path in full_paths]

        log_message("Concatenating clips... (This may take a moment)")
        final_clip = concatenate_videoclips(clips, method="compose")

        log_message(f"Writing final video to: {output_path}")
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        for clip in clips:
            clip.close()
        final_clip.close()

        log_message("\n--- Process Completed Successfully! ---")
        log_message(f"Final video saved as '{output_path}'")

    except Exception as e:
        log_message(f"--- ERROR during video processing: {e} ---")
        log_message("Please ensure 'moviepy' is installed correctly (`pip install moviepy`).")
        log_message("If you are on a system without a graphical interface, you might also need to install 'ffmpeg'.")

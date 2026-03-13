import fal_client
import json
import time
from Utility.utils import log_message


def _clean_llm_output(text: str) -> str:
    try:
        start_index = text.index('[')
        end_index = text.rindex(']')
        return text[start_index:end_index + 1]
    except ValueError:
        log_message("Warning: Could not find JSON array brackets in LLM output.")
        return text


# Please generate simple, clear, short (1-2 sentences) visual descriptions for each of the scenes.
# Keep the scene descriptions simple, preferably one character, (maximum 2) per scene and one main, central action happening.
def _get_initial_plot(topic: str, model: str, scene_nr: int) -> str | None:
    prompt = f"""
    You are a movie director and are tasked to create a simple story for a short film, in {scene_nr} scenes, about "{topic}".
    
    IMPORTANT: The story MUST have EXACTLY {scene_nr} scenes. Not more, not less. Your entire response must be a JSON array containing exactly {scene_nr} objects.
    
    The movie is separated into 5 acts: Introduction, Incident, Action, Climax, Conclusion, in this chronological order.
    Each scene will have a field titled "act" which has to be one of the 5 possible ones. You will have to assign to each scene a representative act.
    For each scene, assign it an appropriate act and describe its events in accordance with how the plot should progress in the given act.
    If the plot has a number of scenes multiple of 5, assign the scenes equally for each act. Otherwise, prioritize the 'Action' act to receive more scenes.
    
    Please generate simple, clear, short (1-2 sentences) visual descriptions for each of the scenes.
    Keep the scene descriptions simple, preferably one character, (maximum 2) per scene and one main, central action happening.
    
    Respond in JSON format, without any preamble, not even the word "json".
    Your entire response must be a single, valid JSON array of objects.
    Each object in the array must have only two keys: "act" and "description".
    """
    log_message(f"Generating initial script for topic: '{topic}' using model: {model}")
    try:
        result = fal_client.run(
            "fal-ai/any-llm",
            arguments={
                "model": model,
                "prompt": prompt
            }
        )
        log_message(f"Received raw response from LLM: {model}")
        return result["output"]
    except Exception as e:
        log_message(f"--- ERROR: API call failed during initial script generation: {e}")
        return None


def _parse_or_fix_json(llm_output_string: str, fix_models: list) -> list | None:
    log_message("Attempting to parse LLM output as JSON...")
    cleaned_string = _clean_llm_output(llm_output_string)

    try:
        parsed_data = json.loads(cleaned_string)
        log_message("Successfully parsed LLM output after initial cleaning.")
        return parsed_data
    except json.JSONDecodeError:
        log_message("Parsing failed due to internal syntax error. Entering multi-model self-healing routine...")
        current_string_to_fix = cleaned_string
        for attempt, model_name in enumerate(fix_models):
            log_message(f"Self-healing attempt {attempt + 1} using model: {model_name}...")
            fix_prompt = f"""
            The following text contains a JSON syntax error (e.g., a missing comma or quote).
            Please correct the syntax and provide ONLY the valid JSON array, with no other text or explanation.
            Invalid JSON text to fix:
            ---
            {current_string_to_fix}
            ---
            """
            fix_result = fal_client.run(
                "fal-ai/any-llm",
                arguments={
                    "model": model_name,
                    "prompt": fix_prompt
                },
            )
            fixed_string = fix_result["output"]
            cleaned_fixed_string = _clean_llm_output(fixed_string)
            try:
                parsed_data = json.loads(cleaned_fixed_string)
                log_message(f"Self-healing successful on attempt {attempt + 1} with model {model_name}.")
                return parsed_data
            except json.JSONDecodeError:
                log_message(f"Self-healing attempt with {model_name} also failed.")
                current_string_to_fix = cleaned_fixed_string
                time.sleep(1)
            except Exception as e:
                log_message(f"--- ERROR: API call failed during fix attempt with {model_name}: {e}")

    log_message("--- FATAL ERROR: Could not get valid JSON after all retries.")
    return None


def _add_scene_ids(script_data: list) -> list:
    for i, scene in enumerate(script_data):
        scene["scene_id"] = i
    return script_data


def generate_plot_from_topic(topic: str, initial_model: str, fix_models: list, scene_nr: int) -> list | None:
    raw_script = _get_initial_plot(topic, initial_model, scene_nr)
    if raw_script:
        project_data = _parse_or_fix_json(raw_script, fix_models)
        if project_data:
            final_data = _add_scene_ids(project_data)
            pretty_json = json.dumps(final_data, indent=4, ensure_ascii=False)
            log_message("Basic Plot Data:\n" + pretty_json)
            return final_data
    return None

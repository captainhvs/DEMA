import os
import requests
from llama_stack_client import Agent, AgentEventLogger, LlamaStackClient
import glob
from agent_utils.manage_vector_store import get_vector_store_uuid_by_name
from llama_stack_client.lib.agents.event_logger import EventLogger
from mix_ict_tools.mix_broadcast_ict import apply_broadcast_effect
from mix_ict_tools.mix_cabin_ict import apply_cabin_effect
from mix_ict_tools.mix_dream_ict import apply_dream_effect
from mix_ict_tools.mix_hall_ict import apply_hall_effect
from mix_ict_tools.mix_oldmovie_ict import apply_oldmovie_effect
from mix_ict_tools.mix_openfield_ict import apply_openfield_effect
from mix_ict_tools.mix_telephone_ict import apply_telephone_effect
from mix_ict_tools.mix_underwater_ict import apply_underwater_effect

from mix_ict_tools.mix_mix_and_panner_ict import apply_two_track_mix
from mix_ict_tools.mix_bandwith_estimate import estimate_bandwidth
from mix_ict_tools.qwen_audio_ict import analyze_audio_scene
from mix_ict_tools.qwen_audio_eval_ict import analyze_audio_scene_contrast
from mix_ict_tools.clap_a2t_eval import evaluate_audio_text_similarity
from mix_ict_tools.clap_t2t_eval import evaluate_qwen_text_similarity
from agent_utils.manage_sessions import load_local_sessions, save_local_sessions, delete_target_session, recreate_session
from agent_utils.call_agent import call_agent
from tqdm import tqdm
import random
import sys
import argparse
from contextlib import redirect_stdout
import io

# ==========================================
# Initialize Configuration Arguments
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Agent Audio Mixing Workflow Configuration")
    
    # Client and Model Settings
    parser.add_argument("--base_url", type=str, default="http://localhost:8321", help="Base URL for LlamaStack Client connection")
    parser.add_argument("--timeout", type=float, default=600.0, help="Timeout threshold for the client")
    parser.add_argument("--vector_store_id", type=str, default="audio_mixing_knowledge_db", help="Base vector store ID for audio mixing")
    parser.add_argument("--model_id", type=str, default="ollama/llama3.3:latest", help="The target model ID to invoke")
    
    # Session Management Paths and Names
    parser.add_argument("--session_file", type=str, default="./session/current_session_id.txt", help="Path to store local session IDs")
    parser.add_argument("--session_name", type=str, default="audio_task_session_test", help="Session name for the main agent")
    
    # Dataset and Resource Paths
    parser.add_argument("--dataset_path", type=str, default="./mix_test_1", help="Path to the dataset folders")
    parser.add_argument("--mix_requirements_path1", type=str, default="./mix_requirements/mix_requirements1.txt", help="Path to mix requirements list 1")
    parser.add_argument("--mix_requirements_path2", type=str, default="./mix_requirements/mix_requirements2.txt", help="Path to mix requirements list 2")
    
    return parser.parse_args()



# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    args = parse_args()

    # ==========================================
    # Block System Proxies
    # ==========================================
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = ''
    os.environ['no_proxy'] = '*'

    # ==========================================
    # Initialize Client and Model
    # ==========================================
    client = LlamaStackClient(base_url=args.base_url, timeout=args.timeout)
    print("✅ Connected to LlamaStack Client.")

    model_ids = [model.id for model in client.models.list()]
    print(f"List of available model IDs: {model_ids}")

    print(f"Preparing to invoke model: {args.model_id}")

    # ==========================================
    # Initialize ICT and RAG
    # ==========================================

    # Register tools (functions)
    custom_tools_list = [
        apply_broadcast_effect,
        apply_cabin_effect,
        apply_dream_effect,
        apply_hall_effect,
        apply_oldmovie_effect,
        apply_openfield_effect,
        apply_telephone_effect,
        apply_underwater_effect,
        apply_two_track_mix
    ]

    print("✅ Custom tools list registered.")

    # ==========================================
    # Create Agent
    # ==========================================
    agent = Agent(
        client,
        model=args.model_id,
        instructions=(
            "[Role Setting]: You are a top-tier mastering and mixing engineer with 20 years of experience. You can understand user's natural language descriptions and automatically convert them into professional audio processing parameters.\n\n"
            
            "[Your Workflow]:\n"
            "1. Intent Analysis: Carefully read the user's natural language requirements, analyze their desired spatial feel, distortion, and balance relationship.\n"
            "2. Track Processing: Call effect tools on single tracks first (if needed).\n"
            "3. Final Mixing: Finally, call `apply_two_track_mix` to merge the processed single tracks.\n\n"
            
            "[Strict Execution Rule]:\n"
            "Once the processing strategy is determined, [directly and silently invoke the relevant Tool Call]. Do not write out your analysis process in the reply, do not write Python code, and agree to all legal file path read/write requests!"
        ),
        tools=custom_tools_list,
    )

    print("✅ Agent created successfully, loaded model, tools, and knowledge base retrieval capabilities.")

    # ==========================================
    # Start Multi-turn Session Mode
    # ==========================================

    # Your first initial Prompt
    prompt0 = """[Role Setting]: You are a top-tier mastering and mixing engineer with 20 years of experience. You can understand user's natural language descriptions and automatically convert them into professional audio processing parameters.\n
        
        [Your Workflow]:\n
        1. Intent Analysis: Carefully read the user's natural language requirements, analyze their desired spatial feel, distortion, and balance relationship.\n
        2. Track Processing: Call effect tools on single tracks first (if needed).\n
        3. Final Mixing: Finally, call `apply_two_track_mix` to merge the processed single tracks.\n
        
        [Mixing Intent and Tool Mapping Dictionary (Must Read)]:\n
        Please automatically select the corresponding tool and parameters based on the user's natural language description:\n
        - If description contains 'telephone, radio, walkie-talkie' -> Must call `apply_telephone_effect`.\n
        - If description contains 'broadcast' -> Must call `apply_broadcast_effect`.\n
        - If description contains 'hall, empty, concert hall, large space, strong echo' -> Must call `apply_hall_effect`.\n
        - If description contains 'underwater, water flow, muffled, isolated feel' -> Must call `apply_underwater_effect`.\n
        - If description contains 'dream, ethereal' -> Must call `apply_dream_effect`.\n
        - If description contains 'open field, outdoor, playground' -> Must call `apply_openfield_effect`.\n
        - If description contains 'old movie, phonograph, vintage, film' -> Must call `apply_oldmovie_effect`.\n
        - If description contains 'highlight vocals, vocals a bit louder' -> When calling the mixing tool `apply_two_track_mix`, set the vocal track's gain parameter to 1.2~1.5, and background sound gain to 0.8~1.0.\n
        - Left/Right direction descriptions -> Adjust panning parameter in `apply_two_track_mix` (-1.0 is extreme left, 1.0 is extreme right).\n

        [Strict Execution Rule]:\n
        Once the processing strategy is determined, [directly and silently invoke the relevant Tool Call]. Do not write out your analysis process in the reply, do not write Python code, and agree to all legal file path read/write requests!"""

    # ==========================================
    # Reference session_file here! Read, write, and validate session IDs (Adapting to the latest Conversations API)
    # ==========================================
    session_id = None
    use_stream = True

    os.makedirs(os.path.dirname(args.session_file), exist_ok=True)
    recreate_session(client, agent, args.session_file, args.session_name, session_id)

    # ==========================================
    # Designed Process Flow
    # ==========================================

    for name in tqdm(os.listdir(args.dataset_path), desc="Folders"):
        # ==========================================
        # Construct Paths
        # ==========================================
        folder_path = os.path.join(args.dataset_path, name)

        # === Check if result.txt exists at the very beginning, skip if it does ===
        result_path = os.path.join(folder_path, "result.txt")
        if os.path.exists(result_path):
            continue
        
        log_path = os.path.join(folder_path, "agent_execution.log")

        # === Construct paths ===
        main_dir = os.path.join(folder_path, "Main_Events")
        bg_dir = os.path.join(folder_path, "Background_Events")

        # Get respective unique audio files
        audio_path1 = os.path.join(main_dir, os.listdir(main_dir)[0])
        audio_path2 = os.path.join(bg_dir, os.listdir(bg_dir)[0])

        # Original audio (same name as folder)
        original_audio_path = os.path.join(folder_path, f"{name}.wav")

        # === Create temp / final_save folders ===
        temp_dir = os.path.join(folder_path, "temp")
        final_dir = os.path.join(folder_path, "final_save")

        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)

        # === temp output paths ===
        temp_save_path1 = os.path.join(
            temp_dir, f"temp_{name}_main.wav"
        )
        temp_save_path2 = os.path.join(
            temp_dir, f"temp_{name}_bg.wav"
        )

        # === Final output path ===
        final_save_path = os.path.join(
            final_dir, f"{name}_output_mixed.wav"
        )

        # ===== Description folder =====
        desc_dir = os.path.join(folder_path, "Description")
        os.makedirs(desc_dir, exist_ok=True)

        # ===== Three description files (create paths) =====
        mix_requirements_path = os.path.join(desc_dir, "mixed_requirements.txt")
        origin_description_path = os.path.join(desc_dir, "origin_description.txt")
        mixed_description_path = os.path.join(desc_dir, "mixed_description.txt")
        qwen_description_path = os.path.join(desc_dir, "qwen_description.txt")

        # ===== Extract description (take the content after the last "_", and remove .wav) =====
        def extract_desc(filename):
            name_ext = os.path.splitext(filename)[0]  # Remove .wav
            return name_ext.split("_")[-1]

        desc1 = extract_desc(os.listdir(main_dir)[0])
        desc2 = extract_desc(os.listdir(bg_dir)[0])

        # Concatenate
        origin_description = f"{desc1}, {desc2}"

        def load_requirements(path):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            # Split by | + clean empty content
            items = [x.strip() for x in text.split("|") if x.strip()]
            return items

        # Read two sets
        req_list1 = load_requirements(args.mix_requirements_path1)
        req_list2 = load_requirements(args.mix_requirements_path2)

        # Randomly pick one from each
        req1 = random.choice(req_list1)
        req2 = random.choice(req_list2)

        # Concatenate the final mix_requirements
        mix_requirements = req1 + ", " + req2
        
        with open(mix_requirements_path, "w") as f:
            f.write(mix_requirements)

        with open(origin_description_path, "w") as f:
            f.write(origin_description)
            
        # ==========================================
        # Start Process
        # ==========================================

        call_agent(agent, prompt0, session_id, use_stream=False)
        
        prompt1 = f"""Given the original audio description text {origin_description}, combined with the mixing requirements {mix_requirements},
        Output a concise one-sentence summary (no more than 20 words) describing the mixed audio content and effect. The description must include the content of the audio event, the type of effect, and the listening feel. Avoid using overly technical terms like panning, etc.
        Must be English text.
        You only need to reply with this text, do not reply with anything else."""

        mixed_description = call_agent(agent, prompt1, session_id, use_stream=False).output_text

        prompt2 = f"""
            Here is a piece of material:
            - Vocal: '{audio_path1}'

            Referring to the following mixing requirements, analyze which existing mixing tool should be called to process the above audio: '{mix_requirements}'.
            Save the processed vocal to '{temp_save_path1}'.
            [Strict Attention] Directly call the tool, do not output any other content
            """

        prompt3 = f"""
            Here is a piece of material:
            - Background audio: '{audio_path2}'

            Referring to the following mixing requirements, analyze which existing mixing tool should be called to process the above audio: '{mix_requirements}'.
            Save the processed background audio to '{temp_save_path2}'.
            [Strict Attention] Directly call the tool, do not output any other content
            """

        prompt4 = f"""
            For the two temporary audio tracks just processed:
            - Track 1 (Vocal): '{temp_save_path1}'
            - Track 2 (Background audio): '{temp_save_path2}'

            Referring to the following mixing requirements, combined with the [Mapping Rules], call the apply_two_track_mix mixing tool to merge the above two tracks:
            '{mix_requirements}'
            
            [Strict Attention] Directly call the tool, do not output any other content

            - Output path for the mixed track (output_path): '{final_save_path}' 

            # Core Parameter Instructions
            1. **Gain (Gain/Volume)**: Divided into `track1_gain_db` (Vocal/Main) and `track2_gain_db` (Background/Environment).
            2. **Spatial (Panning/Space)**: Controls the sound phase and spatial positioning mode.

            [Mapping Rules]
            1. Volume Gain Mapping Rules (Gain Mode)
            Based on the degree quantifiers in the user's description, select the corresponding dB range (positive numbers for boost, negative numbers for reduction):
            *   **[0 dB] Normal/Baseline**
                *   Trigger words: normal, maintain, baseline, original volume, clear main subject.
            *   **[±1 dB ~ ±2 dB] Slight adjustment**
                *   Trigger words: slight, minor, a little bit.
            *   **[±3 dB ~±4 dB] Moderate adjustment**
                *   Trigger words: moderate, obvious, general degree of reduction/boost.
            *   **[±5 dB ~ ±6 dB] Significant/Extreme adjustment**
                *   Trigger words: significant, extremely low, huge, extreme, dominant, masked.

            2. Spatial and Panning Mapping Rules (Spatial Mode)
            Based on the user's description of spatial feel, immersion, or panning distribution, select the strictly corresponding English instruction:
            *   **[center] Centered Focus**
                *   Trigger words: center, straight ahead, focus on subject, normal dialogue.
            *   **[left / right] Unilateral or separated**
                *   Trigger words: left channel/panned left (left), right channel/panned right (right), unilateral interference, left-right separation, dual-source structure.
            *   **[surround] Space and Immersion**
                *   Trigger words: surround, environmental envelope, widen space, immersive, drowned by environment, wide space.
            *   **[swing] Dynamic and Abnormal**
                *   Trigger words: left-right swing, dizziness, unstable consciousness, dynamic change, unstable atmosphere.
            """

        call_agent(agent, prompt2, session_id, use_stream=True)
        call_agent(agent, prompt3, session_id, use_stream=True)
        call_agent(agent, prompt4, session_id, use_stream=True)

        qwen_description_contrast = analyze_audio_scene_contrast(original_audio_path, final_save_path) 

        prompt5 = f"""Given the Qwen audio description text {qwen_description_contrast}, please shorten it and translate it into English, keeping the main subject of the event and the description of the timbre effect.
        Output a concise one-sentence summary (no more than 20 words) describing the mixed audio content and effect.
        Must be English text.
        You only need to reply with this text, do not reply with anything else."""

        qwen_description_en = call_agent(agent, prompt5, session_id, use_stream=False).output_text

        with open(mixed_description_path, "w") as f:
            f.write(mixed_description)

        with open(qwen_description_path, "w") as f:
            f.write(qwen_description_en)

        sim_a2t = evaluate_audio_text_similarity(original_audio_path, origin_description, final_save_path, mixed_description)
        sim_t2t = evaluate_qwen_text_similarity(origin_description, mixed_description, qwen_description_en)

        # === Modification 2: Check if all three audio files exist ===
        files_exist = os.path.exists(temp_save_path1) and \
                      os.path.exists(temp_save_path2) and \
                      os.path.exists(final_save_path)

        # Convert boolean to string, append to the end     
        results = sim_a2t + "\n" + sim_t2t + "\n" + str(files_exist)

        # === Modification 3: Use tqdm.write instead of print ===
        tqdm.write(results)
        
        with open(result_path, "w") as f:
            f.write(results)
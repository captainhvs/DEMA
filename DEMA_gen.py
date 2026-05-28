import os
import requests
import re
import traceback
import subprocess
import time
import uuid
import shutil
import datetime
import argparse
import random
import textwrap
import glob

from llama_stack_client import Agent, AgentEventLogger, LlamaStackClient
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client import LlamaStackClient

from mix_ict_tools.mix_broadcast_ict import apply_broadcast_effect
from mix_ict_tools.mix_cabin_ict import apply_cabin_effect
from mix_ict_tools.mix_dream_ict import apply_dream_effect
from mix_ict_tools.mix_hall_ict import apply_hall_effect
from mix_ict_tools.mix_oldmovie_ict import apply_oldmovie_effect
from mix_ict_tools.mix_openfield_ict import apply_openfield_effect
from mix_ict_tools.mix_telephone_ict import apply_telephone_effect
from mix_ict_tools.mix_underwater_ict import apply_underwater_effect
from mix_ict_tools.mix_mix_and_panner_ict import apply_multi_track_mix
from mix_ict_tools.mix_bandwith_estimate import estimate_bandwidth

from mix_ict_tools.qwen_audio_ict import analyze_audio_scene
from mix_ict_tools.qwen_audio_eval_ict import analyze_audio_scene_contrast
from mix_ict_tools.clap_a2t_eval import evaluate_audio_text_similarity
from mix_ict_tools.clap_t2t_eval import evaluate_qwen_text_similarity
from mix_ict_tools.custom_code import execute_custom_code
from agent_utils.manage_vector_store import get_vector_store_uuid_by_name, get_or_create_vector_store,delete_vector_store_by_name,list_vector_store_files,list_all_vector_stores

from typing import List, Dict, Any
from agent_utils.manage_sessions import load_local_sessions, save_local_sessions, delete_target_session,recreate_session
from agent_utils.summary_utils import summarize_execution_case
from agent_utils.summary_utils import summarize_execution_case,aggregate_and_summarize_cases
from agent_utils.call_agent import call_agent, call_agent_with_evolution,run_multitrack_mixing, evolutionary_parameter_optimization_multitrack, extract_scores

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['no_proxy'] = '*'


# ==========================================
# Initialize Configuration Arguments
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Agent Audio Mixing System Configuration (Single Task Arbitrary Tracks)")

    # Input (Modified for single task and arbitrary tracks)
    parser.add_argument("--input_description", type=str, default="",required=True, help="Input natural language description for mixing.")
    parser.add_argument("--origin_description", type=str, nargs='*', default=[], help="List of original descriptions for each track.")
    parser.add_argument("--input_audio_paths", type=str, nargs='+', required=True, help="List of input audio paths (arbitrary number).")
    parser.add_argument("--output_folder", type=str, required=True, help="Result folder to save all outputs and intermediate files.")

    
    # Client and Model Settings
    parser.add_argument("--base_url", type=str, default="http://localhost:8321", help="Base URL for LlamaStack Client connection")
    parser.add_argument("--model_id", type=str, default="ollama/llama3.3:latest", help="The target model ID to invoke")
    
    # Vector Store IDs and Names
    parser.add_argument("--vector_store_id", type=str, default="audio_mixing_knowledge_db", help="Base vector store ID for audio mixing")
    parser.add_argument("--store_name", type=str, default="Audio Mixing Knowledge Base", help="Name of the main knowledge base")
    parser.add_argument("--debug_store_name", type=str, default="Audio Mixing Debug Knowledge Base", help="Name of the debug knowledge base")
    parser.add_argument("--refine_store_name", type=str, default="Audio Mixing Refine Knowledge Base", help="Name of the refine knowledge base")
    
    # Session Management Paths and Names
    parser.add_argument("--session_file", type=str, default="./session/current_session_id.txt", help="Path to store local session IDs")
    parser.add_argument("--session_name", type=str, default="audio_task_session_outdomain", help="Session name for the main agent")
    parser.add_argument("--summary_session_name", type=str, default="audio_task_session_outdomain_summary", help="Session name for the summary agent")
    parser.add_argument("--refine_session_name", type=str, default="audio_task_session_outdomain_refine", help="Session name for the refine agent")
    
    # Dataset and Resource Paths
    parser.add_argument("--abstract_path", type=str, default="./mix_introduction", help="Path to RAG knowledge markdown documents")
    
    # RAG memory directory paths
    parser.add_argument("--rag_folder", type=str, default="./dual_memory", help="Root directory for RAG memory")
    parser.add_argument("--debug_memory_dir", type=str, default="./dual_memory/debug_success", help="Directory for debug success RAG memory")
    parser.add_argument("--refine_memory_dir", type=str, default="./dual_memory/refine_success", help="Directory for refine success RAG memory")
    
    # Success tracking Markdown files
    parser.add_argument("--rag_success_md", type=str, default="./dual_memory/debug_success_cases_temp.md", help="Temp markdown file for tracking debug success cases")
    parser.add_argument("--rag_success_output_folder", type=str, default="./dual_memory/debug_success", help="Output folder for aggregated debug cases")
    parser.add_argument("--refine_success_md", type=str, default="./dual_memory/refine_success_cases_temp.md", help="Temp markdown file for tracking refine success cases")
    
    # Execution hyper-parameters
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size interval for triggering summary and refinement")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    client = LlamaStackClient(base_url=args.base_url, timeout=12000.0)
    print("✅ Connected to LlamaStack Client.")

    vector_store_uuid = get_vector_store_uuid_by_name(client, args.store_name)
    knowledge_search_tool = {
        "type": "file_search",
        "vector_store_ids": [vector_store_uuid]
    }

    # Register tools
    custom_tools_list = [
        apply_multi_track_mix,
        knowledge_search_tool
    ]

    # ==========================================
    # Create Main Agent
    # ==========================================
    agent = Agent(
        client,
        model=args.model_id,
        instructions=(
            "[Role Setting]: You are a top-tier mastering and mixing engineer with 20 years of experience. You can understand users' natural language descriptions and automatically convert them into professional audio processing parameters.\n\n"
        ),
        tools=custom_tools_list,
    )

    session_id = None
    summary_session_id = None
    refine_session_id = None

    os.makedirs(os.path.dirname(args.session_file), exist_ok=True)
    local_sessions = load_local_sessions(args.session_file)
    delete_target_session(client, args.session_name, args.session_file)

    session_id = recreate_session(client, agent, args.session_file, args.session_name, session_id)
    debug_vector_store_uuid = get_or_create_vector_store(client, args.debug_store_name, args.debug_memory_dir)
    refine_vector_store_uuid = get_or_create_vector_store(client, args.refine_store_name, args.refine_memory_dir)

    def build_prompt_from_md_dir(knowledge_dir: str) -> str:
        """Traverse all .md files in the target directory and concatenate them."""
        md_files = glob.glob(os.path.join(knowledge_dir, "*.md"))
        if not md_files:
            return "[The local knowledge base is empty, please rely on your pre-trained knowledge to process the task]"

        collected_sections = []
        for file_path in md_files:
            filename = os.path.basename(file_path)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        section = f"### [Source file: {filename}]\n{content}"
                        collected_sections.append(section)
            except Exception as e:
                print(f"❌ Failed to read {filename}: {e}")

        merged_knowledge = "\n\n------------------------\n\n".join(collected_sections)

        prompt_kb = f"""[Pre-mixing Professional Knowledge and SOP Specifications]
            Before executing an audio processing task or generating code, you must [MANDATORY] read and strictly refer to the following local knowledge base content provided by the user.
            Your parameter selections and module combinations must comply with the following guidelines:

            {merged_knowledge}

            [Execution Requirements]:
            1. When encountering user requirements, directly refer to the above knowledge content to determine parameters.
            2. There is no need to explain the analysis process.
            3. Do not output any code or call tools in this response turn.
            """
        return prompt_kb

    # ==========================================
    # Single Group Execution Path Setup
    # ==========================================
    folder_path = args.output_folder
    os.makedirs(folder_path, exist_ok=True)

    result_path = os.path.join(folder_path, "result.txt")
    if os.path.exists(result_path):
        print(f"Result already exists at {result_path}. Exiting.")
        exit(0)

    # Create sub-directories under output folder
    temp_dir = os.path.join(folder_path, "temp")
    final_dir = os.path.join(folder_path, "final_save")
    desc_dir = os.path.join(folder_path, "Description")

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(desc_dir, exist_ok=True)

    final_save_path = os.path.join(final_dir, "output_mixed.wav")

    mix_requirements_path = os.path.join(desc_dir, "mixed_requirements.txt")
    origin_description_path = os.path.join(desc_dir, "origin_description.txt")
    mixed_description_path = os.path.join(desc_dir, "mixed_description.txt")

    # Generate paths and descriptions for arbitrary N tracks
    num_tracks = len(args.input_audio_paths)
    exec_script_paths = []
    temp_save_paths = []
    descriptions = []

    print(f"🎵 Processing {num_tracks} tracks...")

    for i, audio_path in enumerate(args.input_audio_paths):
        track_name = os.path.splitext(os.path.basename(audio_path))[0]
        desc = track_name.split("_")[-1]
        exec_script_paths.append(os.path.join(folder_path, f"script_track{i+1}.py"))
        temp_save_paths.append(os.path.join(temp_dir, f"temp_track{i+1}.wav"))

    
    mixed_description = args.input_description 



    # Generate paths and descriptions for arbitrary N tracks
    num_tracks = len(args.input_audio_paths)
    exec_script_paths = []
    temp_save_paths = []
    descriptions = []

    print(f"🎵 Processing {num_tracks} tracks...")

    for i, audio_path in enumerate(args.input_audio_paths):
        exec_script_paths.append(os.path.join(folder_path, f"script_track{i+1}.py"))
        temp_save_paths.append(os.path.join(temp_dir, f"temp_track{i+1}.wav"))
        

        if not args.origin_description:
            track_name = os.path.splitext(os.path.basename(audio_path))[0]
            desc = track_name.split("_")[-1]
            descriptions.append(desc)


    if args.origin_description:
        descriptions = args.origin_description
        

        if len(descriptions) != num_tracks:
            print(f"⚠️ Warning: The number of origin_descriptions ({len(descriptions)}) does not match the number of audio tracks ({num_tracks})!")


    origin_description = descriptions 

    # ==========================================
    # Analyze Agent Prompts & Analysis
    # ==========================================
    mix_requirements = []
    
    print("📝 Generating track-specific mix prompts (4-axis syntax)...")
    for i, track_desc in enumerate(origin_description):
        prompt_mix_decompose = f"""Given the overall mixing goal: '{mixed_description}'
        This task focuses on Track {i+1}, which contains: '{track_desc}'.

        Your task is to decompose the overall mixing goal into specific mixing instructions ONLY for this track, using a strict 4-axis syntax.

        The 4-axis syntax is defined as follows:
        1. Scene setting: This axis defines the target acoustic environment or the physical medium of the audio transmission. It dictates the fundamental reverberation characteristics, frequency envelope alterations, and characteristic resonances.
        2. Subjective feeling: This axis captures the perceptual and emotional descriptors attached to the acoustic scene (e.g., abstract, human-centric adjectives requiring subtle acoustic modifications, saturation, or equalization).
        3. Spatial distribution: This axis governs the localization and temporal movement of the audio source within the stereo field (e.g., panning strategy, stereophonic width).
        4. Volume balance: This axis dictates the relative amplitude adjustments and the dynamic hierarchy between the interacting audio tracks (e.g., gain attenuation or emphasis to prevent frequency masking).

        Please output the structured breakdown for this track only.
        It must be in English. You only need to reply with this structured text, do not reply with any conversational filler, introductions, or code blocks.

        Output strictly in the following format:
        Track {i+1} ({track_desc}):
        - Scene setting: [Your extraction/inference]
        - Subjective feeling: [Your extraction/inference]
        - Spatial distribution: [Your extraction/inference]
        - Volume balance: [Your extraction/inference]
        """
        

        track_req = call_agent(agent, prompt_mix_decompose, session_id, use_stream=False).output_text
        mix_requirements.append(track_req)

    with open(mix_requirements_path, "w") as f: f.write(mix_requirements)
    with open(origin_description_path, "w") as f: f.write(origin_description)
    with open(mixed_description_path, "w") as f: f.write(mixed_description)

    prompt2 = build_prompt_from_md_dir(args.abstract_path)
    call_agent(agent, prompt2, session_id, use_stream=False)

    # Setup summary and refine agents
    prompt_rag = """[Task Trigger Condition]...""" # Omitted long text for code clarity
    summary_agent_instructions = f"""[Role Setting]... \n{prompt2}""" 
    refine_agent_instructions = f"""[Role Setting]... \n{prompt2}""" 
    
    summary_agent = Agent(client, model=args.model_id, instructions=summary_agent_instructions, tools=[])
    refine_agent = Agent(client, model=args.model_id, instructions=refine_agent_instructions, tools=[])

    summary_session_id = recreate_session(client, summary_agent, args.session_file, args.summary_session_name, summary_session_id)
    refine_session_id = recreate_session(client, refine_agent, args.session_file, args.refine_session_name, refine_session_id)

    # ==========================================
    # Arbitrary Tracks Processing (Dynamic Loop)
    # 严格遍历生成代码，不传递list给Agent处理路径
    # ==========================================
    for i in range(num_tracks):
        prompt_track = f"""
            [Strict Task Execution: Generate single-track processing code for Track {i+1}]   
            Target effect: '{mix_requirements[i]}'
            ...
            (Your Standard Execution Constraints Code omitted for brevity, keeping identical constraint context)
            ```python
            plugins.append(HighpassFilter(cutoff_frequency_hz=120.0))
            ```
        """
        call_agent_with_evolution(
            agent, client, summary_agent, summary_session_id, debug_vector_store_uuid, args.rag_folder, 
            prompt_track, prompt_rag, session_id, 
            exec_script_paths[i], args.input_audio_paths[i], temp_save_paths[i], 
            use_stream=True, max_retries=5, timeout=6000
        )

    # ==========================================
    # Cumulative Loop Mixing 
    # ==========================================
    print("🎵 Starting Multi-Track Mix...")
    if num_tracks == 1:
        shutil.copy(temp_save_paths[0], final_save_path)
    else:
       
        run_multitrack_mixing(
            agent, 
            session_id, 
            temp_save_paths, 
            final_save_path, 
            mix_requirements
        )

    # ==========================================
    # Qwen-Audio Evaluation 
    # ==========================================
    # 直接使用单输入 analyze_audio_scene 对最后保存的音频进行理解
    if os.path.exists(final_save_path):
        qwen_description = analyze_audio_scene(final_save_path)
    else:
        qwen_description = "No final audio generated."

    prompt5 = f"""Given the Qwen audio description text {qwen_description}, please shorten it and translate it into English, keeping the main subject of the event and the description of the timbre effect.
    Output an overall summary of the mixed audio content and effect description text in one sentence as briefly as possible (no more than 20 words).
    It must be an English text.
    You only need to reply with this text, do not reply with anything else."""


    qwen_description_en = call_agent(agent, prompt5, session_id, use_stream=False).output_text
    

    # ==========================================
    # Similarity Scores & Evolution
    # ==========================================
    eval_audio_path = final_save_path 
    sim_a2t = evaluate_audio_text_similarity(eval_audio_path, origin_description, final_save_path, mixed_description)
    sim_t2t = evaluate_qwen_text_similarity(origin_description, mixed_description, qwen_description_en)

    prompt6 = f"""Given the Qwen audio description text {qwen_description} and the mixing requirements {mixed_description},
      please analyze what differences exist between the mixing result described by the Qwen audio description text and the initial mixing requirements, and output what improvements still need to be made to the mixing parameters."""

    modification_instructions = call_agent(agent, prompt6, session_id, use_stream=False).output_text
    score = extract_scores(sim_a2t, sim_t2t)

    if score:
        # ==========================================
        # Multi-Track Evolutionary Optimization
        # ==========================================
        print("🧬 Evolutionary refinement needed. Starting multitrack optimization...")
        # 传递整个 list 给更新后的多轨优化函数
        evolutionary_parameter_optimization_multitrack(
            client=client,
            agent=agent, 
            refine_agent=refine_agent,
            session_id=session_id, 
            base_scripts=exec_script_paths,      
            track_save_paths=temp_save_paths,   
            mixed_save_path=final_save_path,
            result_path=result_path,
            original_audio_path=final_save_path, # Evaluated using final mixed audio
            Refine_success_md=args.refine_success_md,
            refine_vector_store_uuid=refine_vector_store_uuid,
            origin_description=origin_description, # 完整的原始描述 (拼接后的字符串)
            mixed_description=mixed_description,
            initial_score=score,
            modification_instructions=modification_instructions,
            mix_prompt=mix_requirements,
            tar_dir=folder_path,
            num_iterations=2,      
            variations_per_iter=3, 
        )
    else:
        # 检查所有单轨及最终文件是否存在
        files_exist = all(os.path.exists(path) for path in temp_save_paths) and os.path.exists(final_save_path)
        results = sim_a2t + "\n" + sim_t2t + "\n" + str(files_exist)
        print(results)
        
        with open(result_path, "w") as f:
            f.write(results)

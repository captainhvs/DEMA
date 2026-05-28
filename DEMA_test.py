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
from mix_ict_tools.mix_mix_and_panner_ict import apply_two_track_mix
from mix_ict_tools.mix_bandwith_estimate import estimate_bandwidth
from mix_ict_tools.qwen_audio_ict import analyze_audio_scene
from mix_ict_tools.qwen_audio_eval_ict import analyze_audio_scene_contrast
from mix_ict_tools.clap_a2t_eval import evaluate_audio_text_similarity
from mix_ict_tools.clap_t2t_eval import evaluate_qwen_text_similarity
from mix_ict_tools.custom_code import execute_custom_code
from agent_utils.manage_vector_store import get_vector_store_uuid_by_name, get_or_create_vector_store,delete_vector_store_by_name,list_vector_store_files,list_all_vector_stores


from typing import List, Dict, Any
from tqdm import tqdm
from agent_utils.manage_sessions import load_local_sessions, save_local_sessions, delete_target_session,recreate_session
from agent_utils.summary_utils import summarize_execution_case
from agent_utils.summary_utils import summarize_execution_case,aggregate_and_summarize_cases
from agent_utils.call_agent import call_agent, call_agent_with_evolution,run_mixing,evolutionary_parameter_optimization,extract_scores

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['no_proxy'] = '*'


# ==========================================
# Initialize Configuration Arguments
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Agent Audio Mixing System Configuration")
    
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
    parser.add_argument("--dataset_path", type=str, default="./results_test_50", help="Path to the dataset folders")
    parser.add_argument("--abstract_path", type=str, default="./mix_introduction", help="Path to RAG knowledge markdown documents")
    parser.add_argument("--mix_requirements_path1", type=str, default="./mix_requirements/mix_requirements1_outdomain.txt", help="Path to mix requirements list 1")
    parser.add_argument("--mix_requirements_path2", type=str, default="./mix_requirements/mix_requirements2_outdomain.txt", help="Path to mix requirements list 2")
    
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

args = parse_args()



if __name__ == "__main__":
    # ==========================================
    # Initialize Client and Model
    # ==========================================

    client = LlamaStackClient(base_url=args.base_url, timeout=12000.0)
    print("✅ Connected to LlamaStack Client.")

    model_ids = [model.id for model in client.models.list()]
    print(f"List of currently available model IDs: {model_ids}")

    print(f"Preparing to invoke model: {args.model_id}")

    file_ids = []
    vector_store_uuid = get_vector_store_uuid_by_name(client, args.store_name)
    knowledge_search_tool = {
        "type": "file_search",
        "vector_store_ids": [vector_store_uuid]  # Your knowledge base UUID
    }

    # Register tools (functions)
    custom_tools_list = [
        apply_two_track_mix,
        knowledge_search_tool
    ]
    print("✅ Custom tools list registered.")

    # ==========================================
    # Create Agent
    # ==========================================

    agent = Agent(
        client,
        model=args.model_id,
        instructions=(
            "[Role Setting]: You are a top-tier mastering and mixing engineer with 20 years of experience. You can understand users' natural language descriptions and automatically convert them into professional audio processing parameters.\n\n"
        ),
        tools=custom_tools_list,
    )

    print("✅ Agent created successfully, model, tools, and knowledge base retrieval capabilities loaded.")

    # ==========================================
    # Session file referencing! Read, write, and validate session IDs (Adapting to the latest Conversations API)
    # ==========================================

    session_id = None
    use_stream = True

    summary_session_id = None
    refine_session_id = None

    os.makedirs(os.path.dirname(args.session_file), exist_ok=True)
    recreate_session(client, agent, args.session_file, args.session_name, session_id)



    debug_vector_store_uuid = get_or_create_vector_store(client, args.debug_store_name, args.debug_memory_dir)
    debug_knowledge_search_tool = {  
        "type": "file_search",
        "vector_store_ids": [debug_vector_store_uuid]  # Your knowledge base UUID
    }

    refine_vector_store_uuid = get_or_create_vector_store(client, args.refine_store_name, args.refine_memory_dir)
    refine_knowledge_search_tool = {  
        "type": "file_search",
        "vector_store_ids": [refine_vector_store_uuid]  # Your knowledge base UUID
    }


    def build_prompt_from_md_dir(knowledge_dir: str) -> str:
        """
        Traverse all .md files in the target directory and concatenate them into a Prompt string for the large model.
        """
        print(f"🔍 Scanning directory to load knowledge base: {knowledge_dir}")
        
        # Get all .md files in the directory
        md_files = glob.glob(os.path.join(knowledge_dir, "*.md"))
        
        if not md_files:
            print("⚠️ No .md files found, returning empty knowledge base prompt.")
            return "[The local knowledge base is empty, please rely on your pre-trained knowledge to process the task]"

        collected_sections = []
        
        for file_path in md_files:
            filename = os.path.basename(file_path)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        # 💡 Highly recommended: Add the filename as a subtitle, the large model responds well to this.
                        section = f"### [Source file: {filename}]\n{content}"
                        collected_sections.append(section)
                        print(f"✅ Successfully loaded: {filename}")
            except Exception as e:
                print(f"❌ Failed to read {filename}: {e}")

        # Concatenate all read file contents with line breaks
        merged_knowledge = "\n\n------------------------\n\n".join(collected_sections)

        # Reconstruct your prompt2
        prompt2 = f"""[Pre-mixing Professional Knowledge and SOP Specifications]
            Before executing an audio processing task or generating code, you must [MANDATORY] read and strictly refer to the following local knowledge base content provided by the user.
            Your parameter selections and module combinations must comply with the following guidelines:

            {merged_knowledge}

            [Execution Requirements]:
            1. When encountering user requirements, directly refer to the above knowledge content to determine parameters.
            2. There is no need to explain the analysis process; directly output the corresponding processing actions or invoke preset tools according to the knowledge above.
            3. Do not output any code or call tools in this response turn. Remember, this prompt2 is only meant to provide knowledge base content for your reference when generating code later.
            """
        return prompt2





    # ==========================================
    # Designed Process
    # ==========================================

    summary_count = 0
    refine_count = 0

    for name in tqdm(os.listdir(args.dataset_path), desc="Folders"):
        # ==========================================
        # Construct Paths
        # ==========================================

        session_id = recreate_session(client, agent, args.session_file, args.session_name, session_id)

        folder_path = os.path.join(args.dataset_path, name)

        exec_python_script_path1 = os.path.join(folder_path, "script_track1.py")
        exec_python_script_path2 = os.path.join(folder_path, "script_track2.py")

        # === Check if result.txt exists at the very beginning, skip if it does ===
        result_path = os.path.join(folder_path, "result.txt")

        if os.path.exists(result_path):
            continue
        
        log_path = os.path.join(folder_path, "agent_execution.log")

        # === Construct paths ===
        main_dir = os.path.join(folder_path, "Main_Events")
        bg_dir = os.path.join(folder_path, "Background_Events")

        # Get respective unique audio
        audio_path1 = os.path.join(main_dir, os.listdir(main_dir)[0])
        audio_path2 = os.path.join(bg_dir, os.listdir(bg_dir)[0])

        # Original audio (same name as folder)
        original_audio_path = os.path.join(folder_path, f"{name}.wav")

        # === Create temp / final_save folders ===
        temp_dir = os.path.join(folder_path, "temp")
        final_dir = os.path.join(folder_path, "final_save")

        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)

        # === Temp output paths ===
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

        # ===== Description folders =====
        desc_dir = os.path.join(folder_path, "Description")
        os.makedirs(desc_dir, exist_ok=True)

        # ===== Three description files (create paths) =====
        mix_requirements_path = os.path.join(desc_dir, "mixed_requirements.txt")
        origin_description_path = os.path.join(desc_dir, "origin_description.txt")
        mixed_description_path = os.path.join(desc_dir, "mixed_description.txt")
        qwen_description_path = os.path.join(desc_dir, "qwen_description.txt")

        # ===== Extract description (take content after the last "_", remove .wav) =====
        def extract_desc(filename):
            name_only = os.path.splitext(filename)[0]  # Remove .wav
            return name_only.split("_")[-1]

        desc1 = extract_desc(os.listdir(main_dir)[0])
        desc2 = extract_desc(os.listdir(bg_dir)[0])

        # Concatenate
        origin_description = f"{desc1}, {desc2}"

        def load_requirements(path):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            # Split by | and clean empty contents
            items = [x.strip() for x in text.split("|") if x.strip()]
            return items

        # Read the two sets
        req_list1 = load_requirements(args.mix_requirements_path1)
        req_list2 = load_requirements(args.mix_requirements_path2)

        # Randomly select one from each
        req1 = random.choice(req_list1)
        req2 = random.choice(req_list2)

        # Construct final mix_requirements
        mix_requirements = req1 + ", " + req2

        # Define prompt0   
        prompt0 = """
            "[Role Setting]: You are a top-tier mastering and mixing engineer with 20 years of experience. You can understand users' natural language descriptions and automatically convert them into professional audio processing parameters.\n\n"
                
                "[Your Workflow]:\n"
                "1. Intent Analysis: Carefully read the user's natural language requirements, analyzing the spatial feel, distortion, and balance relationship they desire.\n"
                "2. Track Processing: First, invoke effect tools on single tracks individually (if needed).\n"
                "3. Final Mixing: Finally, invoke `apply_two_track_mix` to merge the processed single tracks.\n\n"
                
                "[Mixing Intent and Tool Mapping Dictionary (Must Read)]:\n"
                "- Please automatically generate Python audio processing code to achieve the target effect based on the user's natural language description.\n"
                "- Descriptions containing 'highlight vocal, vocal a bit louder' -> When invoking `apply_two_track_mix`, set the vocal track's gain to 1.2~1.5, and background to 0.5~0.8.\n"
                "- Left/Right direction descriptions -> Adjust panning parameters (-1.0 is extreme left, 1.0 is extreme right).\n\n"
            
        """

        # Description extraction prompt
        prompt1 = f"""Given the original audio description text {origin_description}, combined with the mixing requirements {mix_requirements},
        output a concise one-sentence summary (no more than 20 words) describing the mixed audio content and effects. The description must include the content of the audio event, the type of effect, and the listening feel. Avoid overly technical terms.
        It must be in English.
        You only need to reply with this text, do not reply with anything else."""
        mixed_description = call_agent(agent, prompt1, session_id, use_stream=False).output_text

        # ABSTRACT RAG prompt
        prompt2 = build_prompt_from_md_dir(args.abstract_path)
        call_agent(agent, prompt2, session_id, use_stream=False)

        # Mix Prompt 1
        prompt3 = f"""
                    [Strict Task Execution: Generate single-track processing code for the vocal/main audio]   
                    Target effect: '{mix_requirements}'

                    [Environment Constraints] (Must read):
                    The system has automatically read the audio, converted it to mono, and saved it. It has predefined an empty list `plugins = []`.
                    Your only task now is: write the code that appends effects to the `plugins` list (i.e., `plugins.append(...)`).

                    [Execution Constraints]:
                    1. You are free to decide which effects to use and their order based on the target effect.
                    2. Absolutely do not import any libraries (e.g., numpy, soundfile, pedalboard, etc.); the system has imported them globally.
                    3. Absolutely do not write any code regarding audio read/write, loading, or rendering (no sf.read/write, no Pedalboard(plugins)), just generate the append code.
                    4. [Output Format Locked]: Your reply must exclusively be a markdown code block containing Python code (starting with ```python and ending with ```). Do not include any explanatory text or nonsense.
                    5. You can only use the following effects (you can choose to use them or not), the parameters are in the parentheses, you must use '=' to assign values to them when using them:
                    HighpassFilter(cutoff_frequency_hz),
                    LowpassFilter(cutoff_frequency_hz), 
                    Compressor(threshold_db, ratio, attack_ms, release_ms), 
                    Distortion(drive_db), 
                    Gain(gain_db), 
                    Delay(delay_seconds, feedback, mix), 
                    Reverb(room_size, damping,wet_level, dry_level)

                    [Code Output Example]:
                    ```python
                    plugins.append(HighpassFilter(cutoff_frequency_hz=120.0))
                    plugins.append(Compressor(threshold_db=-18.0, ratio=4.0, attack_ms=5.0, release_ms=50.0))
                    plugins.append(Reverb(room_size=0.3, wet_level=0.2, dry_level=0.8))
                    plugins.append(Gain(gain_db=3.0))
                    plugins.append(Distortion(drive_db=0.5))
                    [Special Attention]
                    Invoking any tools is strictly prohibited! You are only allowed to output a code block exactly like the example!
                    """
        
        # Mix Prompt 2
        prompt4 = f"""
                    [Strict Task Execution: Generate single-track processing code for the background audio]
                    Target effect: '{mix_requirements}'

                    [Environment Constraints] (Must read):
                    The system has automatically read the audio, converted it to mono, and saved it. It has predefined an empty list `plugins = []`.
                    Your only task now is: write the code that appends effects to the `plugins` list (i.e., `plugins.append(...)`).

                    [Execution Constraints]:
                    1. You are free to decide which effects to use and their order based on the target effect.
                    2. Absolutely do not import any libraries (e.g., numpy, soundfile, pedalboard, etc.); the system has imported them globally.
                    3. Absolutely do not write any code regarding audio read/write, loading, or rendering (no sf.read/write, no Pedalboard(plugins)), just generate the append code.
                    4. [Output Format Locked]: Your reply must exclusively be a markdown code block containing Python code (starting with ```python and ending with ```). Do not include any explanatory text or nonsense.
                    5. You can only use the following effects (you can choose to use them or not), the parameters are in the parentheses, you must use '=' to assign values to them when using them:
                    HighpassFilter(cutoff_frequency_hz),
                    LowpassFilter(cutoff_frequency_hz), 
                    Compressor(threshold_db, ratio, attack_ms, release_ms), 
                    Distortion(drive_db), 
                    Gain(gain_db), 
                    Delay(delay_seconds, feedback, mix), 
                    Reverb(room_size, damping,wet_level, dry_level)

                    [Code Output Example]:
                    ```python
                    plugins.append(HighpassFilter(cutoff_frequency_hz=120.0))
                    plugins.append(Compressor(threshold_db=-18.0, ratio=4.0, attack_ms=5.0, release_ms=50.0))
                    plugins.append(Reverb(room_size=0.3, wet_level=0.2, dry_level=0.8))
                    plugins.append(Gain(gain_db=3.0))
                    plugins.append(Distortion(drive_db=0.5))

                        [Special Attention]
                        Invoking any tools is strictly prohibited! You are only allowed to output a code block exactly like the example!
                        """
            
            # RAG prompt
        
        prompt_rag = """
        [Task Trigger Condition]:
        When the custom code you generated throws an error, please ask which module of audio processing these errors involve? Why does the error occur?
        You can call the RAG tool to query the relevant knowledge base documents regarding the error module.

        [Execution Steps and Requirements]:
        1. Analyze which module the current error belongs to, and make a RAG query based on this.
        2. Query at most 2 times. The output requirement is to analyze the cause of the current error, and discuss how to correct it.
        """

        summary_agent_prompt = textwrap.dedent("""
            [Task: Debug Success Case Aggregation and Summary]
            Below is a series of successful Python audio processing (e.g., Pedalboard, Numpy, etc.) records after debugging.
            Each round of debugging begins with a date and time like [2026-04-23 23:56:41], accompanied by code modifications and error fixes, ultimately leading to success.
            Please read these records carefully and **merge and classify them by error type**.
            For the errors and ultimately successful codes included in each debugging round, please analyze and summarize by comparing the error code and the final successful code. Strictly follow the Markdown format below for your summary report, paying attention to merge similar cases:
            
            ## 1. [Error Type Name] (e.g.: Pedalboard parameter incompatibility error)
            * **Error Code Snippet**: Extract the most representative traceback snippet causing this class of error.
            * **Core Error Cause**: Summarize the root cause leading to this error.
            * **Common Resolution Method**: Extract the repair actions for this type of error (e.g., removed the unsupported 'order' parameter).
            * **Underlying Principle Analysis**: Why did such a modification successfully fix the problem?
            
            ## 2. [Next Error Type Name]
            ...
            
            Here are the original records:
        """).strip()

        summary_agent_instructions = f"""[Role Setting]:
        You are an experienced Senior Python Engineer and Code Review Expert, proficient in writing and troubleshooting various Python scripts, especially familiar with the underlying logic of audio processing frameworks (like Pedalboard, Numpy, etc.).

        [Core Task]:
        You currently act as a "Summary and Review Expert" in a Multi-Agent automated mixing and code generation workflow. Your sole responsibility is to conduct a high-quality, structured, deep analysis and summary of code execution results (including successes, debug troubleshooting, and final failures), so that the system can archive them as reference cases for subsequent fine-tuning (SFT) or RAG retrieval.

        [Strictly Observed Work Principles]:
        1. **Extremely concise, refuse nonsense**: Your output will be written directly into a Markdown file, therefore it is **absolutely forbidden** to output any filler words, transition sentences, or self-introductions such as "Okay", "Here is the summary", "Hope this helps", etc.
        2. **Direct to the pain point, tech-oriented**: When analyzing an error, you must accurately point out the exception type (e.g., TypeError, ValueError) and the root logical flaw that caused the interruption. When analyzing the debug process, you must clearly extract which line or core block of logic was modified to transition from "error" to "success".
        3. **Strictly follow formatting**: The user will clearly specify the modules you need to output in each request (e.g., "Failure error summary", "Debug action analysis", etc.). You must strictly output according to the module names and numbering provided by the user, and are not allowed to merge or add superfluous sections on your own.
        4. **Professional terminology standard**: Use standard programming and audio engineering terminology when describing problems, avoiding vague statements.

        [Prerequisite Knowledge]
        Below is the calling code concerning the mixing modules. You can refer to the following content to assist your analysis and summary:
        {prompt2}
        """

        refine_agent_instructions = f"""[Role Setting]:
        You are a top-tier "Algorithmic Mixing Engineer" and "Knowledge Base Accumulation Expert". Not only are you proficient in the parameter logic of various Python audio processing frameworks (such as Pedalboard, Numpy, etc.), but you are also skilled at acutely extracting universally applicable acoustic principles and code paradigms from massive amounts of tuning test data.

        [Core Task]:
        You are currently acting as an "Experience Consolidation Expert for Parameter Tuning" in the automated mixing workflow. Your sole responsibility is to perform pattern recognition, deduplication, and deep aggregation on the [large, discrete successful tuning/refining records (including Instructions and Track A/B code)] generated after multiple evolutionary iterations by the system. You need to distill these fragmented records into a structured, high-quality "Golden Mixing Strategy Library", so that the system can use them as a high-signal-to-noise ratio RAG retrieval basis or SFT fine-tuning data.

        [Strictly Observed Work Principles]:
        1. **Extremely concise, refuse nonsense**: Your output will directly overwrite the Markdown knowledge base. It is **absolutely forbidden** to output any filler words, transition sentences, or self-introductions such as "Okay", "Compiled for you as follows", "Hope it helps".
        2. **Cluster by acoustic intent (Core)**: Carefully read the "Parameter Instructions (Instruction)" in all records, and merge records with the same or highly similar acoustic goals (e.g.: "Make the vocal more upfront", "Improve mid-frequency clarity") into one major category. Never retain repetitious or wordy cases.
        3. **Extract common tuning paradigms**: Under the merged categories, do not just simply pile up code. You must accurately summarize the **common parameter change laws** 
        (For example: To achieve this goal, usually the EQ xxxHz on Track A is boosted by x dB, and the Compressor threshold on Track B is lowered by x dB).
        4. **Strictly follow formatting**: You must strictly output according to the Markdown structure stipulated below. Do not invent your own heading hierarchies.

        [Required Output Format] (Please repeat this structure for every summarized category):
        ## [Mixing Intent/Processing Scenario] (e.g.: Vocal high-frequency penetration enhancement and low-frequency muddiness elimination)
        * **Trigger Instruction Features**: Summarize the typical optimization instruction features or keywords belonging to this category. This part must copy the relevant parts from the input.
        * **Core Parameter Action Rules**: Clearly distill which specific effect nodes and their numerical ranges (must be quantified) are usually modified on Track A and Track B to realize this intent.
        * **Acoustic Principle Analysis**: Explain using professional mixing engineering theory why modifying these parameter values achieves the aforementioned subjective listening intent.
        * **Golden Reference Code Paradigm**: Combined with the merged cases, output the most concise, pure, and representative local tuning Python code snippet (Must be wrapped inside ```python ... ```).

        [Prerequisite Knowledge]
        The following is the basic calling architecture of the mixing modules used by the system. Please refer to the variables and interface definitions of this framework when summarizing parameter actions:
        {prompt2}
        """

        # 1. Independent basic Prompt exclusively for clustering Refine records by intent
        refine_summary_prompt = textwrap.dedent("""
            [Task: Ultra-concise numerical distillation of successful parameter fine-tuning (Refine) cases]
            Below is a series of parameter fine-tuning (Refine) records that successfully improved the mix quality score, selected by the evolutionary algorithm.
            Each round's record starts with a date and time like [2026-04-28 23:56:41], containing the "Parameter Instruction (Instruction)" that triggered the optimization and the successfully refined Track A/B code.
            
            Please read these massive records carefully and **deeply classify and merge them by "acoustic intent" or "instruction features"** (for instance, merge all records aimed at "brightening vocals" into one category, picking only the highest-scoring ones, and removing redundancies).
            To build the highest density parameter retrieval library, **you must discard all principle analyses and code examples**, extracting ONLY action guidelines and specific numerical changes.
            
            Please strictly follow the Markdown format below to output the golden strategy library. Do not add any foreword, afterword, or superfluous explanatory formatting:
            
            ## 1. [Mixing Intent/Processing Scenario] (e.g.: Vocal high-frequency penetration enhancement and low-frequency muddiness elimination)
            * **Parameter Modification Guidelines**: Summarize the core tuning actions and operational principles to achieve this intent. You must copy the relevant instruction features from the input and summarize the actions (e.g.: Slow down compressor attack time to preserve transients, and use a high-shelf filter to boost high frequencies).
            * **Adjusted Parameter Types and Specific Values**: Clearly and structurally list which specific effect nodes (like effect names in numpy, pedalboard) were modified on Track A and/or Track B, their corresponding parameter types, 
            and the precise quantified numerical changes or set ranges (e.g., Track A Compressor: attack_ms=25.0, ratio=4.0; Track B HighShelfFilter: cutoff_hz=8000, gain_db=3.5).
            
            ## 2. [Next Mixing Intent/Processing Scenario]
            ...
            
            Here are the original tuning records:
        """).strip()

        summary_agent = Agent(
            client,
            model=args.model_id,
            instructions=summary_agent_instructions,
            tools=[], # If this Agent purely does text analysis and summary, this can actually be empty []
        )

        refine_agent = Agent(
            client,
            model=args.model_id,
            instructions=refine_agent_instructions,
            tools=[], # If this Agent purely does text analysis and summary, this can actually be empty []
        )

        summary_session_id = recreate_session(client, summary_agent, args.session_file, args.summary_session_name, summary_session_id)

        refine_session_id = recreate_session(client, refine_agent, args.session_file, args.refine_session_name, refine_session_id)

        call_agent_with_evolution(agent, client, summary_agent, summary_session_id, debug_vector_store_uuid, args.rag_folder, prompt3, prompt_rag, session_id, exec_python_script_path1, audio_path1, temp_save_path1, use_stream=True, max_retries=5, timeout=6000)
        call_agent_with_evolution(agent, client, summary_agent, summary_session_id, debug_vector_store_uuid, args.rag_folder, prompt4, prompt_rag, session_id, exec_python_script_path2, audio_path2, temp_save_path2, use_stream=True, max_retries=5, timeout=6000)

        run_mixing(agent, session_id, temp_save_path1, temp_save_path2, final_save_path, mix_requirements)

        summary_count += 1

        if summary_count % args.batch_size == 0:
            aggregate_and_summarize_cases(summary_agent, summary_agent_prompt, summary_session_id, args.rag_success_md, args.rag_success_output_folder, "debug_success_cases")
            os.remove(args.rag_success_md)
            with open(args.rag_success_md, "w", encoding="utf-8") as f:
                f.write("") 
            get_or_create_vector_store(client, args.debug_store_name, args.debug_memory_dir)

        qwen_description_contrast = analyze_audio_scene_contrast(original_audio_path, final_save_path) 

        prompt5 = f"""Given the Qwen audio description text {qwen_description_contrast}, please shorten it and translate it into English, keeping the main subject of the event and the description of the timbre effect.
        Output an overall summary of the mixed audio content and effect description text in one sentence as briefly as possible (no more than 20 words).
        It must be an English text.
        You only need to reply with this text, do not reply with anything else."""

        qwen_description_contrast_en = call_agent(agent, prompt5, session_id, use_stream=False).output_text
        sim_a2t = evaluate_audio_text_similarity(original_audio_path, origin_description, final_save_path, mixed_description)
        sim_t2t = evaluate_qwen_text_similarity(origin_description, mixed_description, qwen_description_contrast_en)

        prompt6 = f"""Given the Qwen audio description text {qwen_description_contrast} and the mixing requirements {mix_requirements}, please analyze what differences exist between the mixing result described by the Qwen audio description text and the initial mixing requirements, and output what improvements still need to be made to the mixing parameters."""

        modification_instructions = call_agent(agent, prompt6, session_id, use_stream=False).output_text

        score = extract_scores(sim_a2t, sim_t2t)

        if score:

            evolutionary_parameter_optimization(
                client,
                agent, 
                refine_agent,
                session_id, 
                exec_python_script_path1, 
                exec_python_script_path2, 
                temp_save_path1,
                temp_save_path2,
                final_save_path,
                result_path,
                original_audio_path,
                args.refine_success_md,
                refine_vector_store_uuid,
                origin_description,
                mixed_description,
                score,
                modification_instructions,
                mix_requirements,
                folder_path,
                num_iterations=2,      # Total number of evolutionary iterations
                variations_per_iter=3, # Number of variants generated per iteration for the shootout
            )

            refine_count += 1
            if refine_count % args.batch_size == 0:
                aggregate_and_summarize_cases(refine_agent, refine_summary_prompt, refine_session_id, args.refine_success_md, args.refine_memory_dir, "refine_success_cases")
                os.remove(args.refine_success_md)
                with open(args.refine_success_md, "w", encoding="utf-8") as f:
                    f.write("") 
                get_or_create_vector_store(client, args.refine_store_name, args.refine_memory_dir)

        else:
            # === Check if all three audio files exist ===
            files_exist = os.path.exists(temp_save_path1) and \
                        os.path.exists(temp_save_path2) and \
                        os.path.exists(final_save_path)

            # Convert boolean to string, append to the end     
            results = sim_a2t + "\n" + sim_t2t + "\n" + str(files_exist)

            # === Use tqdm.write instead of print ===
            tqdm.write(results)
            
            with open(result_path, "w") as f:
                f.write(results)

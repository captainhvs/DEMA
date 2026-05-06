import random
import os

def load_data_from_txt(file_scene, file_spatial, file_volume):
    """
    从三个 txt 文件中读取并解析四轴数据。
    """
    scenes_data = []
    spatial_data = []
    volume_data = []

    # 1. 解析 Scene 和 Subjective feeling
    if os.path.exists(file_scene):
        with open(file_scene, 'r', encoding='utf-8') as f:
            # 按照 '|' 分割不同场景
            items = f.read().split('|')
            for item in items:
                if '\\' in item:
                    scene, feelings = item.split('\\')
                    scene = scene.strip()
                    # 按照 ',' 分割候选的主观感受
                    feeling_list = [f.strip() for f in feelings.split(',')]
                    scenes_data.append({'scene': scene, 'feelings': feeling_list})

    # 2. 解析 Spatial distribution
    if os.path.exists(file_spatial):
        with open(file_spatial, 'r', encoding='utf-8') as f:
            spatial_data = [item.strip() for item in f.read().split('|') if item.strip()]

    # 3. 解析 Volume balance
    if os.path.exists(file_volume):
        with open(file_volume, 'r', encoding='utf-8') as f:
            volume_data = [item.strip() for item in f.read().split('|') if item.strip()]

    return scenes_data, spatial_data, volume_data


import random

def sample_mix_prompts(scenes_data, spatial_data, volume_data, difficulty="easy", num_samples=1):
    """
    根据设定的难度，从四轴数据中采样生成 Prompt。
    - easy: Subjective feeling + Scene setting
    - middle: easy + Spatial distribution
    - hard: middle + Volume balance
    """
    results = []
    
    for _ in range(num_samples):
        # 1. 所有难度都需要：随机选择一个场景和其中一个主观感受
        scene_info = random.choice(scenes_data)
        scene = scene_info['scene']
        feeling = random.choice(scene_info['feelings'])
        
        # 构建结构化数据字典
        prompt_info = {
            'difficulty': difficulty,
            'scene': scene,
            'feeling': feeling
        }
        
        # 基础文本模板 (easy)
        # 将 feeling 作为形容词直接放在 scene 前面，并用 .lower() 让场景首字母小写
        prompt_text = f"Process the audio into a {feeling} {scene.lower()} scene."

        # 2. Middle 难度：增加空间分布
        if difficulty in ["middle", "hard"]:
            spatial = random.choice(spatial_data)
            prompt_info['spatial'] = spatial
            prompt_text += f" Set the spatial distribution to {spatial.lower()}."

        # 3. Hard 难度：增加音量平衡
        if difficulty == "hard":
            volume = random.choice(volume_data)
            prompt_info['volume'] = volume
            prompt_text += f" Adjust the volume balance to be {volume.lower()}."

        prompt_info['prompt_text'] = prompt_text
        results.append(prompt_info)

    return results

# ==========================================
# 测试与运行示例
# ==========================================
if __name__ == "__main__":


    scenes_path = "/data/liangzhechun/llama-stack/mix_prompt/scene_setting&subject_feeling.txt"
    spatial_path = "/data/liangzhechun/llama-stack/mix_prompt/spatial_distribution.txt"
    volume_path = "/data/liangzhechun/llama-stack/mix_prompt/volumn_balance.txt"

    scenes, spatials, volumes = load_data_from_txt(scenes_path, spatial_path, volume_path)
    


    print("--- 🟢 Easy Sample ---")
    easy_samples = sample_mix_prompts(scenes, spatials, volumes, difficulty="easy", num_samples=2)
    for res in easy_samples:
        print(f"[{res['scene']}]: {res['prompt_text']}")

    print("\n--- 🟡 Middle Sample ---")
    middle_samples = sample_mix_prompts(scenes, spatials, volumes, difficulty="middle", num_samples=2)
    for res in middle_samples:
        print(f"[{res['scene']}]: {res['prompt_text']}")

    print("\n--- 🔴 Hard Sample ---")
    hard_samples = sample_mix_prompts(scenes, spatials, volumes, difficulty="hard", num_samples=2)
    for res in hard_samples:
        print(f"[{res['scene']}]: {res['prompt_text']}")
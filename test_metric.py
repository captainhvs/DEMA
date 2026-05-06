import os
import re

def calculate_metrics(target_directory):
    # 第一层正则：极其宽泛地匹配任意中英文括号及其中间的内容
    pattern_brackets = r"[\(（](.*?)[\)）]"

    total_files = 0          # 所有 result.txt 的总数
    valid_value_files = 0    # 包含至少 5 个括号数值的 txt 数量
    true_count = 0           # 结尾是 True 的数量

    # 5 个值用于计算
    sum_val1, sum_val2, sum_val3, sum_val4, sum_val5 = 0.0, 0.0, 0.0, 0.0, 0.0

    # 遍历目标文件夹
    for root, dirs, files in os.walk(target_directory):
        if 'result.txt' in files:
            file_path = os.path.join(root, 'result.txt')
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 只要文件能打开，就计入总数
                total_files += 1

                # 1. 判断最后一个词是否为 True
                words = content.split()
                if words and words[-1] == 'True':
                    true_count += 1

                # 2. 提取括号内的数字（放宽限制）
                raw_matches = re.findall(pattern_brackets, content, re.DOTALL)
                matches = []
                
                for m in raw_matches:
                    # 第二层正则：从括号提取出的乱七八糟的内容中，精准扣出数字（支持小数和负数）
                    num_match = re.search(r"[-+]?\d*\.?\d+", m)
                    if num_match:
                        matches.append(float(num_match.group()))
                
                # 现在要求成功提取到的数字至少有 5 个时，才加入均值计算
                if len(matches) >= 5:
                    sum_val1 += matches[0]  # 原始音频和原始描述文本匹配度
                    sum_val2 += matches[1]  # 原始音频和混音描述文本匹配度(A1)
                    sum_val3 += matches[2]  # 混音音频和混音描述文本匹配度(M1)
                    sum_val4 += matches[3]  # 原始音频描述与混音客观描述相似度(A2)
                    sum_val5 += matches[4]  # Qwen描述与混音客观描述相似度(M2)
                    valid_value_files += 1
                else:
                    print(f"⚠️ {file_path} 提取数值失败 (找到 {len(matches)} 个有效数字，少于5个)，但已计入总数。")

            except Exception as e:
                print(f"❌ 处理文件出错 {file_path}: {e}")

    # 打印最终结果
    print("-" * 50)
    if total_files > 0:
        true_ratio = (true_count / total_files) * 100
        print(f"📂 共处理了 {total_files} 个 result.txt 文件。")
        print(f"✅ True 的占比:  {true_ratio:.2f}% ({true_count} / {total_files})")
        
        print("-" * 50)
        # 只有在存在有效数值文件时才计算均值
        if valid_value_files > 0:
            avg_val1 = sum_val1 / valid_value_files
            avg_val2 = sum_val2 / valid_value_files # A1
            avg_val3 = sum_val3 / valid_value_files # M1
            avg_val4 = sum_val4 / valid_value_files # A2
            avg_val5 = sum_val5 / valid_value_files # M2
            
            print(f"📈 以下均值基于 {valid_value_files} 个成功提取数值的文件：")
            print(f"📊 原始音频和原始描述文本匹配度: {avg_val1:.4f}")
            print(f"📊 原始音频和混音描述文本匹配度(A1): {avg_val2:.4f}")
            print(f"📊 混音音频和混音描述文本匹配度(M1): {avg_val3:.4f}")
            print(f"📊 原始描述与混音描述相似度(A2): {avg_val4:.4f}") 
            print(f"📊 Qwen描述与混音描述相似度(M2): {avg_val5:.4f}")
            
            print("-" * 50)
            print("🧮 调和平均数计算 (Harmonic Means):")
            # 计算 A1 和 A2 的调和平均数，防止分母为 0
            if (avg_val2 + avg_val4) != 0:
                hm_A1_A2 = (2 * avg_val2 * avg_val4) / (avg_val2 + avg_val4)
                print(f"⭐ A1 & A2 调和平均数: {hm_A1_A2:.4f}")
            else:
                print("⭐ A1 & A2 调和平均数: 无法计算 (A1+A2=0)")

            # 计算 M1 和 M2 的调和平均数，防止分母为 0
            if (avg_val3 + avg_val5) != 0:
                hm_M1_M2 = (2 * avg_val3 * avg_val5) / (avg_val3 + avg_val5)
                print(f"🌟 M1 & M2 调和平均数 (Aggregate Score S): {hm_M1_M2:.4f}")
            else:
                print("🌟 M1 & M2 调和平均数 (Aggregate Score S): 无法计算 (M1+M2=0)")

        else:
            print("📭 所有文件均未找到符合格式的5个数值，无法计算均值。")
    else:
        print("📭 没有找到任何 result.txt 文件。")
    print("-" * 50)



# ==========================================
# 运行配置
# ==========================================
if __name__ == "__main__":
    # 替换为你实际的文件夹路径
    # target_dir = "/data/liangzhechun/llama-stack/results_test_50"
    # target_dir = "/data/liangzhechun/llama-stack/results_test_50_1"
    # target_dir = "/data/liangzhechun/llama-stack/results_test_50_summary_RAG"
    # target_dir = "/data/liangzhechun/llama-stack/results_test_50_outdomain"  
    # target_dir = "/data/liangzhechun/llama-stack/results_test_50_refine"
    target_dir = "/data/liangzhechun/llama-stack/results_test_50_refine_del_bad"
    # target_dir = "/data/liangzhechun/llama-stack/results_single_1005"  
    calculate_metrics(target_dir)
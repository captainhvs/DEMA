import os
import re
import traceback
import subprocess
from tqdm import tqdm
import random
import sys
from contextlib import redirect_stdout
import io








def load_local_sessions(filepath):
    """读取本地 txt，返回字典格式: {'session_name': 'session_id'}"""
    sessions = {}
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and "=" in line:
                    name, sid = line.split("=", 1)
                    sessions[name.strip()] = sid.strip()
    return sessions

def save_local_sessions(filepath, sessions):
    """将字典重新覆盖写入本地 txt"""
    with open(filepath, "w", encoding="utf-8") as f:
        for name, sid in sessions.items():
            f.write(f"{name}={sid}\n")



def delete_target_session(client, target_session_name, filepath):
    """终极删除：同时清理 Llama Stack 服务端 Conversation 和本地 txt 记录"""
    sessions = load_local_sessions(filepath)
    if target_session_name in sessions:
        target_sid = sessions[target_session_name]
        
        # 1. 尝试从服务端删除 (使用全新的 Conversations API)
        try:
            # 【核心改动】：使用 client.conversations.delete
            client.conversations.delete(conversation_id=target_sid)
            print(f"🗑️ [服务端清理] 已成功销毁会话: {target_session_name}")
        except Exception as e:
            print(f"⚠️ [服务端清理] 找不到或已过期，跳过服务端删除: {e}")
            
        # 2. 从本地字典移除并重新保存
        del sessions[target_session_name]
        save_local_sessions(filepath, sessions)
        print(f"🗑️ [本地清理] 已从本地记录中移除: {target_session_name}")
    else:
        print(f"⚠️ [本地清理] 未找到名为 '{target_session_name}' 的记录。")




def load_session(client,agent,session_file, session_name, session_id):
    """删除指定会话后，立即创建一个同名新会话并保存 ID"""
    # 1. 删除旧会话
    local_sessions = load_local_sessions(session_file)

    
    # 2. 检查我们需要的 session_name 是否在本地记录里
    if session_name in local_sessions:
        saved_session_id = local_sessions[session_name]
        
        try:
            # 【核心改动】：使用全新的 Conversations API 进行校验
            # 注意：参数名变成了 conversation_id，且不需要传 agent_id 了
            client.conversations.retrieve(conversation_id=saved_session_id)
            
            session_id = saved_session_id
            print(f"\n🔄 成功从服务端恢复历史会话 [{session_name}], Session ID: {session_id}")
            return session_id
        except Exception as e:
            print(f"\n⚠️ 会话 [{session_name}] 在服务端校验失败 (可能已被清理)。准备新建...")
            del local_sessions[session_name]


    # 3. 如果本地没有记录，或者服务端验证失败判定为死会话，则在服务端新建
    if not session_id:
        # 直接调用 agent 实例方法创建（它底层会自动去调 client.conversations.create）
        session_id = agent.create_session(session_name=session_name)
        
        # 将新创建的 [name: id] 存入字典，并写回本地 txt
        local_sessions[session_name] = session_id
        save_local_sessions(session_file, local_sessions)
        
        print(f"\n✅ 已在服务端成功开启全新连续会话 [{session_name}], 最新 Session ID: {session_id}")
        return session_id





def recreate_session(client,agent,session_file, session_name,session_id):
    """删除指定会话后，立即创建一个同名新会话并保存 ID"""

    delete_target_session(client, session_name, session_file)
    load_session(client,agent,session_file, session_name, session_id)
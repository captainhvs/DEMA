from llama_stack_client import LlamaStackClient
import os
import glob



# 屏蔽系统代理
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['no_proxy'] = '*'




def list_all_vector_stores(client, base_url):
    """
    列出服务器上所有的向量库名称和 ID
    """
    print(f"\n--- 正在从 {base_url} 获取所有向量库列表 ---")
    print(f"{'向量库名称':<30} | {'向量库 ID (UUID)':<40}")
    print("-" * 75)
    
    try:
        all_stores = client.vector_stores.list()
        if not all_stores:
            print("目前没有任何向量库。")
            return
            
        for store in all_stores:
            print(f"{str(store.name):<30} | {str(store.id):<40}")
        print("-" * 75)
        print(f"总计: {len(all_stores)} 个库")
        
    except Exception as e:
        print(f"查看失败: {e}")





# --- 函数：查看库内文件内容 ---
def list_vector_store_files(client,base_url, target_name):
    """
    查看特定名称向量库中包含的具体文件名
    """
    
    print(f"\n--- 正在检索知识库 '{target_name}' 的内部文件 ---")
    try:
        # 1. 先根据名称找 ID
        all_stores = client.vector_stores.list()
        store = next((s for s in all_stores if s.name == target_name), None)
        
        if not store:
            print(f"❌ 错误: 未找到名为 '{target_name}' 的库，请先创建它。")
            return

        # 2. 调用接口列出该 ID 关联的文件
        # 这里的接口会返回文件 ID 列表
        files_in_store = client.vector_stores.files.list(vector_store_id=store.id)
        
        if not files_in_store:
            print("💡 该知识库目前是空的（没有关联任何文件）。")
            return

        print(f"库 ID: {store.id} 内包含以下文件:")
        print("-" * 50)
        for f in files_in_store:
            # 根据文件 ID 获取文件详情（主要是为了拿到文件名）
            f_details = client.files.retrieve(file_id=f.id)
            print(f"📄 文件名: {f_details.filename:<20} | 文件 ID: {f.id}")
        print("-" * 50)

    except Exception as e:
        print(f"查看库内容失败: {e}")



def delete_vector_store_by_name(client, target_name):
    """
    根据特定名称查找并删除向量库
    """
    print(f"\n--- 准备删除名称为 '{target_name}' 的向量库 ---")
    try:
        # 1. 获取最新列表进行匹配
        all_stores = client.vector_stores.list()
        matches = [s for s in all_stores if s.name == target_name]
        
        if not matches:
            print(f"未发现匹配名称为 '{target_name}' 的库，无需操作。")
            return

        print(f"发现 {len(matches)} 个匹配项，正在执行删除...")
        for store in matches:
            # 执行删除
            client.vector_stores.delete(vector_store_id=store.id)
            print(f"✅ 已成功移除库 - 名称: {store.name} | ID: {store.id}")

    except Exception as e:
        print(f"删除操作失败: {e}")



def get_or_create_vector_store(client, store_name, knowledge_dir):
    """
    获取或创建向量库。
    如果文件夹内没有文件，将创建一个空仓库而不报错。
    """
    actual_id = None
    existing_filenames = set()

    # --- 1. 预检查：尝试找到现有库 ---
    try:
        all_stores = client.vector_stores.list()
        existing_store = next((s for s in all_stores if s.name == store_name), None)
        
        if existing_store:
            actual_id = existing_store.id
            print(f"找到现有同名库: {store_name} (ID: {actual_id})")
            
            # 获取已有文件列表用于去重
            files_in_store = client.vector_stores.files.list(vector_store_id=actual_id)
            for f in files_in_store:
                f_id = getattr(f, 'id', None) or getattr(f, 'file_id', None)
                if f_id:
                    try:
                        f_details = client.files.retrieve(file_id=f_id)
                        existing_filenames.add(f_details.filename)
                    except: continue
    except Exception as e:
        print(f"同步预检提示: {e}")

    # --- 2. 扫描本地文件夹 ---
    new_file_ids = []
    if os.path.exists(knowledge_dir):
        print(f"--- 正在扫描本地目录: {knowledge_dir} ---")
        py_files = glob.glob(os.path.join(knowledge_dir, "*.py"))
        md_files = glob.glob(os.path.join(knowledge_dir, "*.md"))
        all_target_files = py_files + md_files
        
        for file_path in all_target_files:
            fname = os.path.basename(file_path)
            if fname in existing_filenames:
                print(f"⏭️  跳过已存在的文件: {fname}")
                continue
                
            file_ext = os.path.splitext(fname)[1].lower()
            mime_type = "text/markdown" if file_ext == '.md' else "text/x-python"
                
            with open(file_path, "rb") as f:
                print(f"📤 正在上传新文件: {fname}...")
                original_content = f.read().decode('utf-8')
                modified_content = f"### [来源文件: {fname}]\n\n{original_content}"
                
                file_obj = client.files.create(
                    file=(fname, modified_content.encode('utf-8'), mime_type),
                    purpose="assistants",
                )
                new_file_ids.append(file_obj.id)
    else:
        print(f"⚠️ 警告: 知识库目录 {knowledge_dir} 不存在，将尝试创建空仓库。")

    # --- 3. 执行同步或创建逻辑 ---
    try:
        if actual_id:
            # 库已存在：如果有新文件则关联
            if new_file_ids:
                for f_id in new_file_ids:
                    try:
                        client.vector_stores.files.create(vector_store_id=actual_id, file_id=f_id)
                    except: pass
                print(f"✅ 已成功向现有库添加 {len(new_file_ids)} 个新文件。")
            else:
                print("🆗 现有库已是最新状态。")
        else:
            # 🌟 修改点：库不存在时，直接创建，不再检查 new_file_ids 是否为空
            if not new_file_ids:
                print(f"📂 本地文件夹为空或不存在，正在创建空向量库 '{store_name}'...")
            else:
                print(f"🚀 正在创建新向量库 '{store_name}' 并关联 {len(new_file_ids)} 个文件...")

            new_store = client.vector_stores.create(
                name=store_name,
                file_ids=new_file_ids, # 即使是 [] 也会被接受
                extra_body={
                    "provider_id": "faiss", 
                    "embedding_model": "sentence-transformers/nomic-ai/nomic-embed-text-v1.5"
                }
            )
            actual_id = new_store.id
            print(f"✨ 向量库创建成功，ID: {actual_id}")
            
        return actual_id

    except Exception as e:
        print(f"❌ 同步/创建向量库失败: {e}")
        raise e 


def get_vector_store_uuid_by_name(client, target_name):
    """
    根据向量库名称查找并返回其 UUID (id)。
    如果未找到，则返回 None。
    """
    try:
        # 获取所有向量库列表
        all_stores = client.vector_stores.list()
        
        # 寻找第一个名称匹配的库
        store = next((s for s in all_stores if s.name == target_name), None)
        
        if store:
            print(f"✅ 找到知识库 '{target_name}'，UUID 为: {store.id}")
            return store.id
        else:
            print(f"❌ 未找到名为 '{target_name}' 的知识库。")
            return None
            
    except Exception as e:
        print(f"查询 UUID 失败: {e}")
        return None



def clear_vector_store_contents_by_name(client, target_name):
    """
    根据特定名称查找向量库，并清空其中所有的文件内容（不删除向量库本身）。
    """
    print(f"\n--- 准备清空知识库 '{target_name}' 中的所有文件 ---")
    try:
        # 1. 先根据名称找 ID
        all_stores = client.vector_stores.list()
        store = next((s for s in all_stores if s.name == target_name), None)
        
        if not store:
            print(f"❌ 未找到名为 '{target_name}' 的知识库，无法执行清空操作。")
            return

        print(f"✅ 找到知识库 '{target_name}' (ID: {store.id})，正在获取文件列表...")

        # 2. 调用接口列出该 ID 关联的所有文件
        files_in_store = client.vector_stores.files.list(vector_store_id=store.id)
        
        if not files_in_store:
            print("💡 该知识库目前已经是空的（没有关联任何文件），无需清理。")
            return

        # 3. 遍历删除文件
        # print(f"共发现 {len(files_in_store)} 个文件，开始逐一删除...")
        print("-" * 50)
        
        success_count = 0
        for f in files_in_store:
            # 兼容性获取文件 ID
            f_id = getattr(f, 'id', None) or getattr(f, 'file_id', None)
            if not f_id:
                continue
                
            try:
                # 步骤 A: 从向量库中移除该文件的关联
                client.vector_stores.files.delete(vector_store_id=store.id, file_id=f_id)
                
                # 步骤 B: (可选但推荐) 从 Llama Stack 的文件管理系统中彻底物理删除该文件，防止占用空间
                try:
                    client.files.delete(file_id=f_id)
                except Exception as file_del_e:
                    # 如果该文件被其他向量库复用，物理删除可能会报错，这里做个忽略处理
                    pass 
                
                print(f"🗑️ 已成功删除文件 ID: {f_id}")
                success_count += 1
            except Exception as e:
                print(f"❌ 删除文件 ID {f_id} 时发生错误: {e}")
                
        print("-" * 50)
        print(f"✅ 清理完成！成功移除了 {success_count} 个文件，知识库 '{target_name}' 现已清空。")

    except Exception as e:
        print(f"清空库内容失败: {e}")



if __name__ == "__main__":

    # --- 使用配置 ---
    BASE_URL = "http://localhost:8321"
    TARGET_NAME = "Audio Mixing Knowledge Base"
    TARGET_NAME = "Audio Mixing Refine Knowledge Base"
    TARGET_NAME = "Audio Mixing Debug Knowledge Base"
    knowledge_dir = "/data/liangzhechun/llama-stack/memory_RAG/debug_success"

    client = LlamaStackClient(base_url=BASE_URL)


    # # # 1. 查看所有知识库
    # list_all_vector_stores(client, BASE_URL)


    # # 删除指定知识库
    # clear_vector_store_contents_by_name(client, TARGET_NAME)
    # delete_vector_store_by_name(client, TARGET_NAME)


    # 3.查看知识库里到底有哪些内容
    # list_vector_store_files(client,BASE_URL, TARGET_NAME)


    # 4. 获取或创建知识库，并同步本地文件夹中新增的脚本
    actual_store_id = get_or_create_vector_store(client, TARGET_NAME, knowledge_dir)




    # # 5. 根据名称获取向量库 UUID（验证函数）
    # get_vector_store_uuid_by_name(client, TARGET_NAME)


    #  llama-stack-client models register /data/liangzhechun/llama-stack/ckpt/nomic-embed-text-v1.5 --provider-id sentence-transformers --model-type embedding
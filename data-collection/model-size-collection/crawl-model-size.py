import threading, requests, json, time, yaml, random, os
from queue import Queue, Empty
from tqdm import tqdm

# --- 自定义超时异常 ---
class ModelExtractionTimeout(Exception):
    """单个模型采集时间过长"""
    pass

class HuggingFaceAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://huggingface.co'
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def get_tree_contents(self, model_id, path=''):
        """使用验证成功的 /tree/main 端点"""
        sub_path = f"/{path}" if path else ""
        url = f"{self.base_url}/api/models/{model_id}/tree/main{sub_path}"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=60)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # 遭遇限流，执行指数退避
                    wait_time = 20 * (attempt + 1) + random.uniform(0, 5)
                    time.sleep(wait_time)
                    continue
                return {"error": f"Status {response.status_code}"}
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return {"error": str(e)}
        return {"error": "Max retries exceeded"}

    def crawl_recursive(self, model_id, entries, deadline):
        """
        深度递归扫描：
        保留了全局超时检查，但去除了 max_depth 限制以支持全量研究。
        """
        files_info = {}
        
        # 检查是否超过该模型的总限时
        if time.time() > deadline:
            raise ModelExtractionTimeout("单个模型采集耗时超过 30 分钟上限")

        for entry in entries:
            path_in_repo = entry.get('path', '')
            
            if entry.get('type') == 'directory':
                # 在递归请求前加入极小延迟，防止单线程瞬间爆发请求引发 429
                time.sleep(0.2) 
                
                sub_contents = self.get_tree_contents(model_id, path_in_repo)
                if isinstance(sub_contents, list):
                    # 递归进入，不限深度
                    files_info.update(self.crawl_recursive(model_id, sub_contents, deadline))
                else:
                    # 记录子目录访问错误（如 403 Forbidden）
                    files_info[path_in_repo] = sub_contents
            else:
                # 记录文件及其大小
                files_info[path_in_repo] = entry.get('size', 0)
        return files_info

def worker(api_key, model_queue, processed_ids, in_flight, processed_lock, write_lock, pbar, jsonl_path, interval, thread_id):
    hf_api = HuggingFaceAPI(api_key)
    # 设定单个模型的最大等待时间（秒）
    PER_MODEL_TIMEOUT = 2400 

    while True:
        try:
            model_id = model_queue.get_nowait()
        except Empty:
            break

        # --- Fix 1: 提前检查去重，避免 continue 误触发 finally 块 ---
        with processed_lock:
            if model_id in processed_ids or model_id in in_flight:
                # 如果使用了 model_queue.join()，这里需要 task_done()
                continue
            in_flight.add(model_id)

        actual_request_made = False
        try:
            deadline = time.time() + PER_MODEL_TIMEOUT
            
            # 1. 尝试获取根目录内容
            top_level = hf_api.get_tree_contents(model_id)
            actual_request_made = True

            if isinstance(top_level, list):
                # 2. 递归开始
                full_tree = hf_api.crawl_recursive(model_id, top_level, deadline)
            else:
                # 处理 404/401 等初始错误
                full_tree = top_level

        except ModelExtractionTimeout as e:
            full_tree = {"error": f"Timeout: {str(e)}"}
            pbar.write(f"⏰ [超时跳过] {model_id}")
        except Exception as e:
            full_tree = {"error": f"SystemCrash: {str(e)}"}
            pbar.write(f"⚠️ [系统异常] {model_id}: {str(e)}")
        finally:
            # --- Fix 1：确保数据记录后，再从 in_flight 移除并更新进度 ---
            with write_lock:
                with open(jsonl_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({model_id: full_tree}, ensure_ascii=False) + "\n")
            
            with processed_lock:
                processed_ids.add(model_id)
                in_flight.discard(model_id)
            
            pbar.update(1)
            if actual_request_made:
                time.sleep(interval)

def load_processed_ids(jsonl_path):
    """
    Fix 3: 智能分类扫描进度。
    只跳过不可恢复的错误，允许重试临时性错误。
    """
    processed = set()
    if os.path.exists(jsonl_path):
        print("正在同步已有进度...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    for mid, content in d.items():
                        if isinstance(content, dict) and "error" in content:
                            err = str(content["error"])
                            # 不可重试的硬错误列表
                            hard_errors = ["Status 404", "Status 401", "Status 403", "Timeout"]
                            if any(he in err for he in hard_errors):
                                processed.add(mid)
                            # 429 限流或 Connection Error 不在上述列表，下次运行会自动重爬
                        else:
                            # 成功获取的数据直接跳过
                            processed.add(mid)
                except: continue
    print(f"✅ 断点续传：已跳过 {len(processed)} 个已完成/不可恢复的任务。")
    return processed

# def merge_jsonl_to_final_json(jsonl_path, final_json_path):
#     """Fix 4: 补全合并逻辑"""
#     print(f"正在合并为最终大 JSON 文件...")
#     seen = set()
#     with open(final_json_path, 'w', encoding='utf-8') as out:
#         out.write("{\n")
#         first = True
#         with open(jsonl_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 try:
#                     obj = json.loads(line)
#                     for mid, details in obj.items():
#                         if mid in seen: continue
#                         if not first: out.write(",\n")
#                         out.write(f"  {json.dumps(mid, ensure_ascii=False)}: {json.dumps(details, ensure_ascii=False)}")
#                         seen.add(mid)
#                         first = False
#                 except: continue
#         out.write("\n}")
#     print(f"🎊 合并完成！最终累计 {len(seen)} 条模型树数据。")

def merge_jsonl_to_final_json(jsonl_path, final_json_path):
    print(f"正在合并为最终大 JSON 文件（确保最新补课数据生效）...")
    
    all_data = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # 每一行都是一个字典，如 {"ID": "Content"}
                obj = json.loads(line)
                # update 会自动处理：如果 ID 重复，新值会覆盖旧值
                # 这样你最后那 2808 个成功的结果就会顶掉之前的错误记录
                all_data.update(obj)
            except: continue
    
    with open(final_json_path, 'w', encoding='utf-8') as out:
        # 使用 indent=2 让大 JSON 易于阅读
        json.dump(all_data, out, ensure_ascii=False)
    
    print(f"🎊 合并完成！最终唯一且“转正”后的模型总数：{len(all_data)}")   

def main():
    # --- 基础路径设置 ---
    filtered_dir = 'data-collection/model-filtering/output'
    details_dir = 'data-collection/model-size-collection'
    output_dir = f'{details_dir}/output'  # output/ 已在 .gitignore 里
    input_file = f'{filtered_dir}/2026-11-03_filtered_models_8199.json'
    jsonl_path = f'{output_dir}/model_size_tree.jsonl'
    final_json_path = f'{output_dir}/model_size_tree_final.json'

    # 设定爬取范围（None = 到列表末尾）
    start_index, end_index = 0, None
    # ------------------

    os.makedirs(output_dir, exist_ok=True)
    with open(f'{details_dir}/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    processed_ids = load_processed_ids(jsonl_path)
    in_flight = set()
    processed_lock = threading.Lock()
    write_lock = threading.Lock()
    model_queue = Queue()

    with open(input_file, 'r') as f:
        model_list = json.load(f)[start_index:end_index]
    
    for m in model_list:
        mid = m.get('modelId')
        if mid and mid not in processed_ids:
            model_queue.put(mid)

    pbar = tqdm(total=model_queue.qsize(), desc="🌳 Tree 采集进度", unit="model")

    threads = []
    for i, key in enumerate(config['huggingface_keys']):
        # 由于 Tree 采集涉及大量递归网络请求，建议间隔不低于 1.5 秒
        t = threading.Thread(target=worker, args=(key, model_queue, processed_ids, in_flight, 
                                                 processed_lock, write_lock, pbar, jsonl_path, 1.5, i+1))
        t.start()
        threads.append(t)
        time.sleep(0.5)

    for t in threads:
        t.join()
    
    pbar.close()
    
    # 全部跑完后进行一次大合并
    if os.path.exists(jsonl_path):
        merge_jsonl_to_final_json(jsonl_path, final_json_path)

if __name__ == '__main__':
    main()
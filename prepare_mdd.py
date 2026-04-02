import json
import os
import glob

# 1. 路径配置
mdd_path = '/mnt/workspace/MDD-5k/**/*.json'
lf_data_dir = '/mnt/workspace/LLaMA-Factory/data'
info_path = os.path.join(lf_data_dir, 'dataset_info.json')

# 2. 搜索并读取文件
json_files = sorted(glob.glob(mdd_path, recursive=True))

sharegpt_data = []

for file in json_files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            content = json.load(f)
            # 兼容 root 是列表或字典的情况
            items = content if isinstance(content, list) else [content]
            
            for item in items:
                if 'conversation' not in item: continue
                convs = []
                
                for turn in item['conversation']:
                    p_val = turn.get('patient') or turn.get('Patient')
                    d_val = turn.get('doctor') or turn.get('Doctor')
                    
                    # 核心逻辑：确保 human 和 gpt 严格交替，防止报错
                    if p_val:
                        if not convs or convs[-1]['from'] == 'gpt':
                            convs.append({'from': 'human', 'value': p_val})
                        else: # 如果上一句也是人说的，合并在一起
                            convs[-1]['value'] += "\n" + p_val
                            
                    if d_val:
                        if not convs or convs[-1]['from'] == 'human':
                            convs.append({'from': 'gpt', 'value': d_val})
                        else: # 如果上一句也是医生说的，合并在一起
                            convs[-1]['value'] += "\n" + d_val
                
                # 只有对话有来有回（至少2条）才算有效
                if len(convs) >= 2:
                    sharegpt_data.append({'conversations': convs})
                    
    except Exception:
        continue

# 3. 截取前 30 条精华数据
final_data = sharegpt_data[:30]

# 4. 保存转换后的数据
os.makedirs(lf_data_dir, exist_ok=True)
with open(os.path.join(lf_data_dir, 'mdd_train.json'), 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)


# 5. 注册到 LLaMA-Factory (dataset_info.json)
if os.path.exists(info_path):
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
else:
    info = {}

info['mdd_5k'] = {
    'file_name': 'mdd_train.json',
    'formatting': 'sharegpt',
    'columns': { 'messages': 'conversations' },
    'tags': {
        'role_tag': 'from', 'content_tag': 'value',
        'user_tag': 'human', 'assistant_tag': 'gpt'
    }
}

with open(info_path, 'w', encoding='utf-8') as f:
    json.dump(info, f, ensure_ascii=False, indent=2)

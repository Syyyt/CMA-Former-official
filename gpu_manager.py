import os
import sys
import subprocess

def get_gpu_status():
    """
    使用 nvidia-smi 获取 GPU 状态
    返回列表: [{'index': 0, 'memory': 100, 'utilization': 10}, ...]
    """
    try:
        # 查询 index, memory.used, utilization.gpu
        cmd = "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd.split(), encoding='utf-8')
        
        gpu_list = []
        for line in output.strip().split('\n'):
            if not line: continue
            idx, mem, util = line.split(',')
            gpu_list.append({
                'index': int(idx),
                'memory': int(mem.strip()),        # 显存使用量 (MB)
                'utilization': int(util.strip())   # 核心利用率 (%)
            })
        return gpu_list
    except Exception as e:
        print(f"[GPU Manager] Warning: 无法获取 GPU 状态 ({e})，默认使用 GPU 0")
        return []

def auto_select_gpu(memory_threshold=1000, verbose=True):
    """
    自动设置 CUDA_VISIBLE_DEVICES
    :param memory_threshold: 显存阈值，虽然我们选最小的，但可以用来打印警告
    :param verbose: 是否打印选择信息
    """
    gpus = get_gpu_status()
    
    if not gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        return

    # 排序策略：
    # 1. 优先按显存使用量排序 (从小到大)
    # 2. 如果显存一样，按核心利用率排序 (从小到大)
    best_gpu = sorted(gpus, key=lambda x: (x['memory'], x['utilization']))[0]
    
    # 设置环境变量
    target_idx = str(best_gpu['index'])
    os.environ["CUDA_VISIBLE_DEVICES"] = target_idx
    
    if verbose:
        print("="*40)
        print(f"[GPU Manager] 检测到 {len(gpus)} 张 GPU")
        for gpu in gpus:
            status = "Selected" if str(gpu['index']) == target_idx else "        "
            print(f"  GPU {gpu['index']}: 显存 {gpu['memory']} MB | 利用率 {gpu['utilization']}% | {status}")
        print(f"[GPU Manager] ===> 自动切换至 GPU {target_idx}")
        print("="*40)

# 为了方便直接调用，也可以在这里执行，但建议在主程序显式调用函数
if __name__ == "__main__":
    auto_select_gpu()

import multiprocessing
import subprocess
import random
import time
import re
from tqdm import tqdm

def parse_main_cpp():
    TxAntNum = None
    QAM_bitLength = None
    modulation_mapping = {
        'QAM16': 4,
        'QAM64': 6,
        'QAM256': 8,
        'QAM1024': 10,
        'QAM4096': 12,
    }
    try:
        with open('main.cpp', 'r') as file:
            content = file.read()
            # 提取TxAntNum
            tx_match = re.search(r'static constexpr size_t TxAntNum\s*=\s*(\d+);', content)
            if tx_match:
                TxAntNum = int(tx_match.group(1))
            else:
                raise ValueError("Cannot find TxAntNum in main.cpp")

            # 提取调制方式
            qam_match = re.search(r'using QAM\s*=\s*(\w+)<', content)
            if qam_match:
                qam_type = qam_match.group(1)
                if qam_type in modulation_mapping:
                    QAM_bitLength = modulation_mapping[qam_type]
                else:
                    raise ValueError(f"Unsupported QAM type: {qam_type}")
            else:
                raise ValueError("Cannot find QAM type in main.cpp")
    except FileNotFoundError:
        raise FileNotFoundError("main.cpp not found in the current directory")
    return TxAntNum, QAM_bitLength

def run_simulation(sample, snr, playout, c, r, seed):
    # 构建命令行参数
    cmd = ['./build/demo', str(sample), str(snr), str(playout), str(c), str(r), str(seed)]
    # 运行C++程序
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Process failed with return code {result.returncode}")
        print(f"stderr: {result.stderr}")
        return None
    # 解析输出结果
    output = result.stdout
    err_bits = 0
    err_frames = 0
    for line in output.splitlines():
        if 'Total Err Bits:' in line:
            err_bits = int(line.strip().split(':')[-1])
        elif 'Total Err Frames:' in line:
            err_frames = int(line.strip().split(':')[-1])
    return err_bits, err_frames

def main():
    # 动态读取main.cpp中的参数
    TxAntNum, QAM_bitLength = parse_main_cpp()
    if TxAntNum is None or QAM_bitLength is None:
        print("Failed to parse TxAntNum or QAM_bitLength from main.cpp")
        return

    # 设置目标错误帧数

    total_err_bits = 0
    total_err_frames = 0
    total_samples = 0  # 总帧数

    # 参数设置
    target_err_frames = 1000
    sample = 100
    snr = 9
    playout = 10
    c = 4
    r = 0.1

    # 获取CPU核心数
    num_processes = multiprocessing.cpu_count()

    # 创建进程池
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 初始化进度条
        pbar = tqdm(total=target_err_frames, desc='Processing', unit='err_frames')
        results = []
        while total_err_frames < target_err_frames:
            # 提交新的任务
            while len(results) < num_processes:
                seed = random.randint(0, 1000000)
                result = pool.apply_async(run_simulation, args=(sample, snr, playout, c, r, seed))
                results.append(result)
            # 检查任务是否完成
            time.sleep(0.1)
            for i in range(len(results)-1, -1, -1):
                result = results[i]
                if result.ready():
                    res = result.get()
                    if res is not None:
                        err_bits, err_frames = res
                        total_err_bits += err_bits
                        total_err_frames += err_frames
                        total_samples += sample
                        pbar.update(err_frames)
                        # 更新进度条的后缀信息
                        total_bits = total_samples * TxAntNum * QAM_bitLength
                        ber = total_err_bits / total_bits if total_bits > 0 else 0
                        fer = total_err_frames / total_samples if total_samples > 0 else 0
                        pbar.set_postfix({'Err Bits': total_err_bits, 'BER': ber, 'FER': fer})
                    results.pop(i)
            if total_err_frames >= target_err_frames:
                break
        pbar.close()
    print(f"Total Err Bits: {total_err_bits}")
    print(f"Total Err Frames: {total_err_frames}")
    total_bits = total_samples * TxAntNum * QAM_bitLength
    ber = total_err_bits / total_bits
    fer = total_err_frames / total_samples
    print(f"Total Bits: {total_bits}")
    print(f"Total Samples (Frames): {total_samples}")
    print(f"BER: {ber}")
    print(f"FER: {fer}")

if __name__ == '__main__':
    main()

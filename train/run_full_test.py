import torch
import subprocess
import tempfile
import time
from pathlib import Path

# 【修改点 1】：换成我们刚刚写好的极速加载器
from bnn_dataset import FastBNNDataset

def test_full_dataset():
    print("正在加载 CIFAR-10 完整测试集 (共2000张飞机与汽车)...")
    
    # 使用极速读取！
    test_dataset = FastBNNDataset(train=False)
    total_samples = len(test_dataset)
    
    # 【修改点 2】：调整相对路径，指向上一层目录的 score_cli
    score_cli_path = "../score_cli"
    
    print(f"\n🚀 开始 C 引擎全面测评，共 {total_samples} 个样本...")
    print("因为需要跨进程调用 C 命令行 2000 次，这大约需要十几秒钟，请稍候...\n")
    
    correct_count = 0
    start_time = time.time()
    
    for i in range(total_samples):
        img_tensor, true_label = test_dataset[i]
        img_bytes = (img_tensor * 255).to(torch.uint8).numpy().tobytes()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(img_bytes)
            tmp_bin_path = f.name
            
        try:
            cli_output = subprocess.check_output(
                [score_cli_path, tmp_bin_path],
                stderr=subprocess.STDOUT,
                text=True
            ).strip()
            
            score = float(cli_output)
            predicted_label = 1 if score > 0 else 0
            
            if predicted_label == true_label:
                correct_count += 1
                
        finally:
            Path(tmp_bin_path).unlink(missing_ok=True)
            
        # 每处理 200 个样本，汇报一次进度
        if (i + 1) % 200 == 0:
            current_acc = (correct_count / (i + 1)) * 100
            print(f"已处理: {i + 1:4d} / {total_samples} | 当前准确率: {current_acc:.2f}%")
            
    end_time = time.time()
    
    print("-" * 50)
    print(f"🎉 完整实战大考结束！")
    print(f"总样本数: {total_samples}")
    print(f"正确预测: {correct_count}")
    print(f"最终大考准确率: {(correct_count/total_samples)*100:.2f}%")
    print(f"评测总耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    score_cli_path = "../score_cli"
    if not Path(score_cli_path).exists():
        print(f"❌ 错误：找不到 C 引擎可执行文件 ({score_cli_path})！")
        print("请确保你已经在根目录下执行了 make 命令。")
    else:
        test_full_dataset()
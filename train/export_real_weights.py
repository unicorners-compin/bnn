import torch
from pathlib import Path

# 导入我们之前写的模型结构
from bnn_model import BNNClassifier

def export_to_c_header(pth_path="bnn_trained_promax.pth", output_header="src/bnn_weights.h"):
    print(f"正在加载训练好的模型: {pth_path}...")
    
    # 1. 实例化模型并加载训练好的权重
    model = BNNClassifier()
    model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    
    # 获取底层权重数据 (形状为 [1, 8704])，并展平为一维
    weights_float = model.weight.data.view(-1)
    
    # 2. 二值化：把浮点数重新变成 0 和 1
    # 权重大于等于 0 视为 1，小于 0 视为 0 (与底层的 XNOR 逻辑对应)
    bits = (weights_float >= 0).int().tolist()
    
    # 3. 压缩打包：把 8704 个 bit 打包成 136 个 64位的无符号整数 (uint64_t)
    words = []
    for i in range(136):
        # 每次截取 64 个 bit
        word_bits = bits[i * 64 : (i + 1) * 64]
        
        # 组装成一个 64 位整数
        val = 0
        for j in range(64):
            val |= (word_bits[j] << j)
        words.append(val)
        
    # 4. 写入 C 语言头文件
    header_path = Path(output_header)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "#ifndef BNN_WEIGHTS_H",
        "#define BNN_WEIGHTS_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define BNN_MODEL_WORDS {len(words)}u",
        "#define BNN_MODEL_BIAS 0",
        "",
        f"static const uint64_t BNN_WEIGHTS[{len(words)}] = {{"
    ]
    
    # 将整数格式化为十六进制写入
    for i, w in enumerate(words):
        comma = "," if i + 1 < len(words) else ""
        lines.append(f"    0x{w:016X}ULL{comma}")
        
    lines.extend([
        "};",
        "",
        "#endif",
        ""
    ])
    
    header_path.write_text("\n".join(lines), encoding="ascii")
    print(f"🎉 成功！真实权重已导出至: {header_path}")

if __name__ == "__main__":
    # 如果你的 C 代码 src 目录在别的地方，可以在这里修改路径
    export_to_c_header(output_header="bnn_weights.h")
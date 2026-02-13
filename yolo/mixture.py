import os
import random

# -------------------------- 配置项：请替换为你的实际文件路径 --------------------------
# 已生成的COCO TXT路径
coco_train_path = r"C:\Users\Administrator\Desktop\yolov8-pytorch-master\Datasets\Labels\coco_train.txt"
coco_val_path = r"C:\Users\Administrator\Desktop\yolov8-pytorch-master\Datasets\Labels\coco_val.txt"
# 已生成的自定义火焰/烟雾TXT路径
custom_train_path = r"C:\Users\Administrator\Desktop\yolov8-pytorch-master\Datasets\Labels\custom_train.txt"
custom_val_path = r"C:\Users\Administrator\Desktop\yolov8-pytorch-master\Datasets\Labels\custom_val.txt"
# 最终合并+打乱后的TXT保存路径
final_train_out = r"C:\Users\Administrator\Desktop\yolov8-pytorch-master\Datasets\Labels\train.txt"
final_val_out = r"C:\Users\Administrator\Desktop\yolov8-pytorch-master\Datasets\Labels\val.txt"
# 随机种子（固定后，每次打乱顺序一致，便于复现）
RANDOM_SEED = 0

# -------------------------- 工具函数：读取TXT并过滤空行 --------------------------
def read_txt(file_path):
    """读取TXT文件，返回非空行的列表"""
    if not os.path.exists(file_path):
        raise ValueError(f"❌ 文件不存在：{file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]  # 过滤空行和纯空格行
    print(f"✅ 读取{file_path}：共{len(lines)}行有效内容")
    return lines


# -------------------------- 主逻辑：合并+打乱 --------------------------
def main():
    # 1. 读取所有训练集内容（COCO + 自定义）
    coco_train_lines = read_txt(coco_train_path)
    custom_train_lines = read_txt(custom_train_path)
    final_train_lines = coco_train_lines + custom_train_lines  # 合并训练集

    # 2. 读取所有验证集内容（COCO + 自定义）
    coco_val_lines = read_txt(coco_val_path)
    custom_val_lines = read_txt(custom_val_path)
    final_val_lines = coco_val_lines + custom_val_lines  # 合并验证集

    # 3. 打乱顺序（固定种子，确保每次运行结果一致）
    random.seed(RANDOM_SEED)
    random.shuffle(final_train_lines)
    random.shuffle(final_val_lines)
    print(f"✅ 训练集已打乱：共{len(final_train_lines)}行")
    print(f"✅ 验证集已打乱：共{len(final_val_lines)}行")

    # 4. 写入最终TXT文件
    with open(final_train_out, "w", encoding="utf-8") as f:
        f.write("\n".join(final_train_lines))
    print(f"✅ 最终训练集已保存：{final_train_out}")

    with open(final_val_out, "w", encoding="utf-8") as f:
        f.write("\n".join(final_val_lines))
    print(f"✅ 最终验证集已保存：{final_val_out}")


if __name__ == "__main__":
    main()
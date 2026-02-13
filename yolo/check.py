import os

# 替换为你的标注文件路径
annotation_files = ["C:\\Users\\Administrator\\Desktop\\yolov8-pytorch-master\\Datasets\\Labels\\train.txt",
                    "C:\\Users\\Administrator\\Desktop\\yolov8-pytorch-master\\Datasets\\Labels\\val.txt"]
num_classes = 82  # 你的总类别数，索引0-81

for file in annotation_files:
    print(f"检查文件：{file}")
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            img_path = parts[0]
            boxes = parts[1:]
            for box_idx, box in enumerate(boxes):
                try:
                    x1, y1, x2, y2, cls_id = box.split(",")
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    cls_id = int(cls_id)

                    # 检查类别ID越界
                    if cls_id >= num_classes or cls_id < 0:
                        print(f"❌ 行{line_idx + 1} 框{box_idx + 1}：类别ID={cls_id} 超出范围（0-{num_classes - 1}）")
                    # 检查标注框非法
                    if x1 >= x2 or y1 >= y2:
                        print(
                            f"❌ 行{line_idx + 1} 框{box_idx + 1}：标注框坐标非法（x1={x1} ≥ x2={x2} 或 y1={y1} ≥ y2={y2}）")
                except Exception as e:
                    print(f"❌ 行{line_idx + 1} 框{box_idx + 1}：格式错误 - {e}")
        print("检查完成\n")
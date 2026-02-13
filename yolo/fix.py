import os

annotation_files = ["C:\\Users\\Administrator\\Desktop\\yolov8-pytorch-master\\Datasets\\Labels\\train.txt",
                    "C:\\Users\\Administrator\\Desktop\\yolov8-pytorch-master\\Datasets\\Labels\\val.txt"]
num_classes = 82

for file in annotation_files:
    print(f"修复文件：{file}")
    fixed_lines = []
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            img_path = parts[0]
            boxes = parts[1:]
            valid_boxes = []
            for box in boxes:
                try:
                    x1, y1, x2, y2, cls_id = box.split(",")
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    cls_id = int(cls_id)
                    # 过滤非法标注
                    if cls_id < 0 or cls_id >= num_classes:
                        continue
                    if x1 >= x2 or y1 >= y2:
                        continue
                    valid_boxes.append(box)
                except:
                    continue
            # 保留有有效标注的行
            if valid_boxes:
                fixed_line = f"{img_path} {' '.join(valid_boxes)}"
                fixed_lines.append(fixed_line)
    # 覆盖原文件
    with open(file, "w", encoding="utf-8") as f:
        f.write("\n".join(fixed_lines))
    print(f"修复完成，保留 {len(fixed_lines)} 行有效数据\n")
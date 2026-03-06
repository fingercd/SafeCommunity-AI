import os

# ====================== 配置项（根据你的实际情况修改） ======================
# 训练集txt文件路径（替换成你自己的路径）
TRAIN_ANNOTATION_PATH = r"C:\Users\Administrator\Desktop\yolov8-pytorch-master\Datasets\Labels\train.txt"
# 输入图片尺寸（对应你的input_shape=640）
IMG_SIZE = 640
# 数据集总类别数（82类）
NUM_CLASSES = 82
# 检查的行数（设为-1检查全部，设为100检查前100行）
CHECK_LINES = -1

# ====================== 核心检查逻辑 ======================
def check_train_annotation():
    # 统计变量
    total_lines = 0          # 总检查行数
    error_lines = 0          # 有错误的行数
    error_boxes = 0          # 有错误的框数量
    missing_img = 0          # 图片路径不存在的数量

    # 读取标注文件
    if not os.path.exists(TRAIN_ANNOTATION_PATH):
        print(f"❌ 错误：标注文件不存在 → {TRAIN_ANNOTATION_PATH}")
        return

    with open(TRAIN_ANNOTATION_PATH, encoding='utf-8') as f:
        train_lines = f.readlines()

    # 限制检查行数
    if CHECK_LINES > 0:
        train_lines = train_lines[:CHECK_LINES]

    print(f"===== 开始检查标注文件：{TRAIN_ANNOTATION_PATH} =====")
    print(f"检查行数：{len(train_lines)} | 图片尺寸：{IMG_SIZE} | 类别数：{NUM_CLASSES}\n")

    # 逐行检查
    for line_idx, line in enumerate(train_lines):
        total_lines += 1
        line = line.strip()
        if not line:  # 跳过空行
            continue

        try:
            # 分割图片路径和标注框
            parts = line.split()
            img_path = parts[0]
            boxes = parts[1:] if len(parts) > 1 else []

            # 检查1：图片路径是否存在
            if not os.path.exists(img_path):
                missing_img += 1
                print(f"【行{line_idx}】❌ 图片不存在 → {img_path}")
                error_lines += 1

            # 检查2：标注框格式
            for box_idx, box in enumerate(boxes):
                box_parts = box.split(',')
                # 框必须包含x1,y1,x2,y2,cls 5个值
                if len(box_parts) != 5:
                    error_boxes += 1
                    print(f"【行{line_idx} 框{box_idx}】❌ 格式错误（需5个值）→ {box}")
                    continue

                # 提取坐标和类别
                x1, y1, x2, y2, cls = box_parts
                # 检查坐标是否为数字
                try:
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    cls = int(cls)
                except ValueError:
                    error_boxes += 1
                    print(f"【行{line_idx} 框{box_idx}】❌ 数值错误 → {box}")
                    continue

                # 检查3：坐标是否在合理范围（0 ~ IMG_SIZE）
                if not (0 <= x1 <= IMG_SIZE and 0 <= y1 <= IMG_SIZE and 0 <= x2 <= IMG_SIZE and 0 <= y2 <= IMG_SIZE):
                    error_boxes += 1
                    print(f"【行{line_idx} 框{box_idx}】❌ 坐标越界 → 坐标：({x1},{y1},{x2},{y2}) | 范围：0~{IMG_SIZE}")

                # 检查4：类别ID是否合法
                if not (0 <= cls < NUM_CLASSES):
                    error_boxes += 1
                    print(f"【行{line_idx} 框{box_idx}】❌ 类别ID越界 → ID：{cls} | 范围：0~{NUM_CLASSES-1}")

        except Exception as e:
            error_lines += 1
            print(f"【行{line_idx}】❌ 未知错误 → {str(e)[:100]}")

    # ====================== 输出检查总结 ======================
    print("\n" + "="*50)
    print(f"✅ 检查完成 | 总行数：{total_lines}")
    print(f"❌ 错误行数：{error_lines} | 错误框数：{error_boxes} | 缺失图片：{missing_img}")
    if error_lines == 0 and error_boxes == 0 and missing_img == 0:
        print(f"🎉 标注文件无异常！")
    else:
        print(f"⚠️  发现异常，请修正后再训练！")

if __name__ == "__main__":
    check_train_annotation()
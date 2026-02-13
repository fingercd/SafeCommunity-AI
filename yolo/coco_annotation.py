import json
import os
import random
from collections import defaultdict

# -------------------------- 路径配置 --------------------------
# 使用正斜杠统一路径，避免转义问题
train_img_path = r"C:/Users/Administrator/Desktop/yolov8-pytorch-master/Datasets/JPEGImages/train"
val_img_path = r"C:/Users/Administrator/Desktop/yolov8-pytorch-master/Datasets/JPEGImages/val"

train_json_path = r"C:/Users/Administrator/Desktop/yolov8-pytorch-master/Datasets/Annotations/coco/train_json/instances_train2017.json"
val_json_path = r"C:/Users/Administrator/Desktop/yolov8-pytorch-master/Datasets/Annotations/coco/val_json/instances_val2017.json"

train_out_path = r"C:/Users/Administrator/Desktop/yolov8-pytorch-master/Datasets/Labels/coco_train.txt"
val_out_path = r"C:/Users/Administrator/Desktop/yolov8-pytorch-master/Datasets/Labels/coco_val.txt"
test_out_path = r"C:/Users/Administrator/Desktop/yolov8-pytorch-master/Datasets/Labels/coco_test.txt"
test_images_out = r"C:/Users/Administrator/Desktop/yolov8-pytorch-master/Datasets/Labels/coco_test_images.txt"

# 数据集划分比例（8:2:1）
# 训练集:验证集:测试集 = 8:2:1
train_ratio = 8 / 11  # 8/11 ≈ 0.727
val_ratio = 2 / 11  # 2/11 ≈ 0.182
test_ratio = 1 / 11  # 1/11 ≈ 0.091

# 确保输出目录存在
os.makedirs(os.path.dirname(train_out_path), exist_ok=True)
os.makedirs(os.path.dirname(val_out_path), exist_ok=True)
os.makedirs(os.path.dirname(test_out_path), exist_ok=True)


# -------------------------- COCO类别ID映射 --------------------------
def coco_cat2id(cat):
    if cat >= 1 and cat <= 11:
        cat -= 1
    elif cat >= 13 and cat <= 25:
        cat -= 2
    elif cat >= 27 and cat <= 28:
        cat -= 3
    elif cat >= 31 and cat <= 44:
        cat -= 5
    elif cat >= 46 and cat <= 65:
        cat -= 6
    elif cat == 67:
        cat -= 7
    elif cat == 70:
        cat -= 9
    elif cat >= 72 and cat <= 82:
        cat -= 10
    elif cat >= 84 and cat <= 90:
        cat -= 11
    return cat


# -------------------------- 加载并合并COCO数据 --------------------------
def load_and_merge_coco_data(train_json_path, val_json_path):
    """
    加载并合并训练集和验证集的COCO数据
    """
    print("加载训练集JSON...")
    with open(train_json_path, encoding="utf-8") as f:
        train_data = json.load(f)

    print("加载验证集JSON...")
    with open(val_json_path, encoding="utf-8") as f:
        val_data = json.load(f)

    # 合并数据
    merged_data = {
        "images": train_data["images"] + val_data["images"],
        "annotations": train_data["annotations"] + val_data["annotations"],
        "categories": train_data["categories"]  # 假设类别相同
    }

    print(f"合并后数据统计:")
    print(f"  图片数量: {len(merged_data['images'])}")
    print(f"  标注数量: {len(merged_data['annotations'])}")
    print(f"  类别数量: {len(merged_data['categories'])}")

    return merged_data


# -------------------------- 转换函数 --------------------------
def convert_coco2txt_with_split(img_base_path, json_data, train_out_path, val_out_path, test_out_path, test_images_out,
                                max_boxes_per_image=40):
    """
    转换COCO标注为自定义TXT格式，并按8:2:1比例划分
    """
    if not json_data:
        raise ValueError("JSON数据为空！")

    name_box_id = defaultdict(list)
    test_name_box_id = defaultdict(list)  # 测试集只保存图片路径

    # 创建图片ID到文件名的映射
    id_to_filename = {}
    for img in json_data["images"]:
        id_to_filename[img["id"]] = img["file_name"]

    # 先收集所有标注
    all_annotations = []
    for ant in json_data["annotations"]:
        img_id = ant["image_id"]
        if img_id in id_to_filename:
            all_annotations.append(ant)

    # 按图片ID分组
    annotations_by_image = defaultdict(list)
    for ant in all_annotations:
        annotations_by_image[ant["image_id"]].append(ant)

    # 获取所有图片ID并打乱
    all_image_ids = list(annotations_by_image.keys())
    random.seed(0)  # 固定随机种子
    random.shuffle(all_image_ids)

    # 按照8:2:1比例划分
    total_images = len(all_image_ids)
    train_num = int(total_images * train_ratio)
    val_num = int(total_images * val_ratio)
    test_num = total_images - train_num - val_num

    train_ids = all_image_ids[:train_num]
    val_ids = all_image_ids[train_num:train_num + val_num]
    test_ids = all_image_ids[train_num + val_num:]

    print(f"\n数据集划分:")
    print(f"  训练集图片: {len(train_ids)} ({train_num / total_images * 100:.1f}%)")
    print(f"  验证集图片: {len(val_ids)} ({val_num / total_images * 100:.1f}%)")
    print(f"  测试集图片: {len(test_ids)} ({test_num / total_images * 100:.1f}%)")

    # 统计变量
    filtered_count = 0
    processed_count = 0
    skipped_too_many_boxes = 0

    # 处理训练集和验证集
    def process_set(image_ids, is_test=False):
        nonlocal filtered_count, processed_count, skipped_too_many_boxes

        result = defaultdict(list)
        test_result = defaultdict(list)

        for img_id in image_ids:
            real_filename = id_to_filename[img_id]

            # 检查文件名前三位是否为"000"（如果不是测试集）
            filename_without_ext = os.path.splitext(real_filename)[0]

            if not is_test and (len(filename_without_ext) < 3 or filename_without_ext[:3] != "000"):
                filtered_count += 1
                continue

            # 确定图片路径（根据文件名判断在train还是val文件夹）
            img_file_in_train = os.path.join(img_base_path, "train", real_filename).replace("\\", "/")
            img_file_in_val = os.path.join(img_base_path, "val", real_filename).replace("\\", "/")

            if os.path.exists(img_file_in_train):
                img_file = img_file_in_train
            elif os.path.exists(img_file_in_val):
                img_file = img_file_in_val
            else:
                filtered_count += 1
                continue

            # 如果是测试集，只保存图片路径
            if is_test:
                test_result[img_file] = []  # 空列表表示没有标注
                continue

            # 收集该图片的所有标注
            boxes = []
            for ant in annotations_by_image[img_id]:
                # 转换坐标格式
                cat_id = coco_cat2id(ant["category_id"])
                x1, y1, w, h = ant["bbox"]
                x2, y2 = x1 + w, y1 + h

                # 格式化为字符串
                box_str = f"{int(x1)},{int(y1)},{int(x2)},{int(y2)},{cat_id}"
                boxes.append(box_str)
                processed_count += 1

            # 检查标注数量是否超过限制
            if len(boxes) > max_boxes_per_image:
                skipped_too_many_boxes += 1
                continue

            if boxes:  # 只保存有标注的图片
                result[img_file] = boxes

        return result, test_result

    # 处理三个集合
    print("\n处理训练集...")
    train_data, _ = process_set(train_ids, is_test=False)

    print("处理验证集...")
    val_data, _ = process_set(val_ids, is_test=False)

    print("处理测试集...")
    _, test_data = process_set(test_ids, is_test=True)

    # 写入训练集文件
    print(f"\n写入训练集: {train_out_path}")
    with open(train_out_path, "w", encoding="utf-8", newline='\n') as f:
        for img_file, boxes in train_data.items():
            line = f"{img_file} {' '.join(boxes)}"
            f.write(line + "\n")

    # 写入验证集文件
    print(f"写入验证集: {val_out_path}")
    with open(val_out_path, "w", encoding="utf-8", newline='\n') as f:
        for img_file, boxes in val_data.items():
            line = f"{img_file} {' '.join(boxes)}"
            f.write(line + "\n")

    # 写入测试集文件（仅图片路径）
    print(f"写入测试集: {test_out_path}")
    with open(test_out_path, "w", encoding="utf-8", newline='\n') as f:
        for img_file in test_data.keys():
            f.write(f"{img_file}\n")

    # 写入测试集图片路径文件
    print(f"写入测试集图片路径: {test_images_out}")
    with open(test_images_out, "w", encoding="utf-8", newline='\n') as f:
        for img_file in test_data.keys():
            f.write(f"{img_file}\n")

    # 生成合并的训练+验证集文件
    combined_trainval_path = r"C:/Users/Administrator/Desktop/yolov8-pytorch-master/Datasets/Labels/coco_trainval.txt"
    print(f"生成合并训练+验证集: {combined_trainval_path}")
    with open(combined_trainval_path, "w", encoding="utf-8", newline='\n') as f:
        # 写入训练集
        with open(train_out_path, "r", encoding="utf-8") as f_train:
            f.write(f_train.read())
        # 写入验证集
        with open(val_out_path, "r", encoding="utf-8") as f_val:
            f.write(f_val.read())

    # 输出统计信息
    print("\n" + "=" * 50)
    print("COCO数据集转换完成！")
    print(f"训练集: {len(train_data)} 张图片")
    print(f"验证集: {len(val_data)} 张图片")
    print(f"测试集: {len(test_data)} 张图片")
    print(f"总计: {len(train_data) + len(val_data) + len(test_data)} 张图片")
    print(f"过滤非'000'开头图片: {filtered_count} 张")
    print(f"跳过标注过多(>{max_boxes_per_image})的图片: {skipped_too_many_boxes} 张")
    print(f"训练时使用：训练集 + 验证集 = {len(train_data) + len(val_data)} 张图片")
    print(f"测试集图片路径保存在: {test_images_out}")


# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    print("开始加载和合并COCO数据集...")
    try:
        # 加载并合并训练集和验证集数据
        merged_data = load_and_merge_coco_data(train_json_path, val_json_path)

        # 转换为TXT格式并划分数据集
        convert_coco2txt_with_split(
            img_base_path=r"C:/Users/Administrator/Desktop/yolov8-pytorch-master/Datasets/JPEGImages",
            json_data=merged_data,
            train_out_path=train_out_path,
            val_out_path=val_out_path,
            test_out_path=test_out_path,
            test_images_out=test_images_out,
            max_boxes_per_image=40
        )
    except Exception as e:
        print(f"❌ COCO数据集转换失败：{e}")
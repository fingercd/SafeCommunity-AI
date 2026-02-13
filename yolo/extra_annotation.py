import os
import random
import xml.etree.ElementTree as ET

# -------------------------- 你的配置参数（无需修改） --------------------------
# 自定义XML根目录（所有XML都在这个目录下，未拆分train/val）
xml_dir = r"C:\Users\Administrator\Desktop\yolov8-pytorch-master\Datasets\Annotations\custom"
# 自定义图片根目录（下有train/val子文件夹）
img_dir = r"C:\Users\Administrator\Desktop\yolov8-pytorch-master\Datasets\JPEGImages"
# 类别文件（COCO80类+fire+smoke，需确保最后两行是fire、smoke）
classes_path = r"C:\Users\Administrator\Desktop\yolov8-pytorch-master\Class/coco_classes.txt"
# 生成的自定义TXT保存路径
train_out = r"C:\Users\Administrator\Desktop\yolov8-pytorch-master\Datasets\Labels\custom_train.txt"
val_out = r"C:\Users\Administrator\Desktop\yolov8-pytorch-master\Datasets\Labels\custom_val.txt"
test_out = r"C:\Users\Administrator\Desktop\yolov8-pytorch-master\Datasets\Labels\custom_test.txt"  # 新增测试集输出路径
test_images_out = r"C:\Users\Administrator\Desktop\yolov8-pytorch-master\Datasets\Labels\test_images.txt"  # 新增：仅测试集图片路径

# 数据集划分比例（8:2:1）
# 训练集:验证集:测试集 = 8:2:1
train_ratio = 7 / 10  # 8/11 ≈ 0.727
val_ratio = 2 / 10  # 2/11 ≈ 0.182
test_ratio = 1 / 10  # 1/11 ≈ 0.091

# ---------- 新增：标注数量限制 ----------
# 每张图片最大允许的标注框数量，超过则跳过该图片
MAX_BOXES_PER_IMAGE = 40

# -------------------------- 固定配置（无需修改） --------------------------
# 支持的图片后缀（按优先级匹配）
SUPPORTED_IMG_SUFFIX = [".jpg", ".jpeg", ".png", ".bmp"]
# 图片的子文件夹（按优先级匹配）
IMG_SUBDIRS = ["train", "val"]


# -------------------------- 加载类别映射 --------------------------
def get_classes(classes_path):
    with open(classes_path, encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes, len(classes)


classes, num_classes = get_classes(classes_path)
class2id = {cls: idx for idx, cls in enumerate(classes)}
# 验证fire和smoke是否在类别文件中
assert "fire" in class2id, f"类别文件{classes_path}中未找到'fire'类！"
assert "smoke" in class2id, f"类别文件{classes_path}中未找到'smoke'类！"
print(f"✅ 加载类别数：{num_classes}，fire ID={class2id['fire']}，smoke ID={class2id['smoke']}")
print(f"✅ 标注数量限制：每张图片最多 {MAX_BOXES_PER_IMAGE} 个标注框")
print(f"✅ 数据集划分比例：训练集{train_ratio * 100:.1f}% | 验证集{val_ratio * 100:.1f}% | 测试集{test_ratio * 100:.1f}%")


# -------------------------- 按XML文件名匹配图片路径 --------------------------
def get_img_path_by_xml_name(xml_basename):
    """
    根据XML文件名（无后缀），在img_dir的train/val子文件夹中匹配图片
    xml_basename: XML文件名（如 fire_001）
    return: 匹配到的图片绝对路径，未找到则返回None
    """
    img_basename = xml_basename
    # 遍历图片子文件夹（train→val）
    for subdir in IMG_SUBDIRS:
        # 遍历支持的图片后缀
        for suffix in SUPPORTED_IMG_SUFFIX:
            img_path = os.path.join(img_dir, subdir, img_basename + suffix)
            if os.path.exists(img_path):
                return img_path
    return None


# -------------------------- 转换单张XML为标注行 --------------------------
def convert_xml_to_line(xml_file, include_boxes=True):
    """
    转换单个XML为TXT的一行
    xml_file: XML文件绝对路径
    include_boxes: 是否包含标注框（True用于训练/验证集，False用于测试集）
    return: 标注行字符串，失败则返回None
    """
    # 提取XML文件名（无路径、无后缀）
    xml_basename = os.path.basename(xml_file)
    xml_name_no_ext = os.path.splitext(xml_basename)[0]

    # 匹配图片路径
    img_path = get_img_path_by_xml_name(xml_name_no_ext)
    if not img_path:
        print(f"⚠️ 未找到XML[{xml_file}]对应的图片，跳过")
        return None

    # 如果不需要标注框（测试集），直接返回图片路径
    if not include_boxes:
        return img_path

    # 解析XML标注框
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception as e:
        print(f"⚠️ 解析XML[{xml_file}]失败：{e}，跳过")
        return None

    box_str = ""
    box_count = 0  # ---------- 新增：统计有效标注框数量 ----------

    for obj in root.iter("object"):
        # 跳过困难样本
        difficult = obj.find("difficult").text if obj.find("difficult") is not None else "0"
        if int(difficult) == 1:
            continue

        # 匹配类别ID
        cls_name = obj.find("name").text.strip()
        if cls_name not in class2id:
            print(f"⚠️ XML[{xml_file}]中的类别[{cls_name}]不在类别文件中，跳过该标注")
            continue
        cls_id = class2id[cls_name]

        # 解析坐标（兼容浮点数坐标，比如标注工具输出的100.5）
        bndbox = obj.find("bndbox")
        x1 = int(float(bndbox.find("xmin").text))
        y1 = int(float(bndbox.find("ymin").text))
        x2 = int(float(bndbox.find("xmax").text))
        y2 = int(float(bndbox.find("ymax").text))

        # 过滤无效框（x1 >= x2 或 y1 >= y2）
        if x1 >= x2 or y1 >= y2:
            print(f"⚠️ XML[{xml_file}]中的坐标无效（x1={x1},y1={y1},x2={x2},y2={y2}），跳过该标注")
            continue

        # ---------- 新增：标注数量限制检查 ----------
        box_count += 1
        if box_count > MAX_BOXES_PER_IMAGE:
            print(f"⚠️ XML[{xml_file}]标注数量超过{MAX_BOXES_PER_IMAGE}个，跳过该图片")
            return None

        # 格式：" x1,y1,x2,y2,cls_id"（前面加空格，避免和图片路径粘连）
        box_str += f" {x1},{y1},{x2},{y2},{cls_id}"

    # 过滤空标注
    if box_str.strip() == "":
        print(f"⚠️ XML[{xml_file}]无有效标注框，跳过")
        return None

    # 拼接最终行（图片路径 + 标注框）
    return f"{img_path}{box_str}"


# -------------------------- 主逻辑：划分数据集并生成TXT --------------------------
def main():
    # 1. 获取所有XML文件
    xml_list = [os.path.join(xml_dir, f) for f in os.listdir(xml_dir) if f.lower().endswith(".xml")]
    if not xml_list:
        raise ValueError(f"❌ XML目录[{xml_dir}]下未找到任何.xml文件！")
    total_xml = len(xml_list)
    print(f"✅ 共找到{total_xml}个XML文件")

    # 2. 随机打乱并按照8:2:1比例划分
    random.seed(0)  # 固定随机种子，确保划分结果可复现
    random.shuffle(xml_list)

    # 计算各集合的数量
    train_num = int(total_xml * train_ratio)
    val_num = int(total_xml * val_ratio)
    test_num = total_xml - train_num - val_num  # 确保总数不变

    # 划分数据集
    train_list = xml_list[:train_num]
    val_list = xml_list[train_num:train_num + val_num]
    test_list = xml_list[train_num + val_num:]

    print(f"✅ 划分结果：训练集{len(train_list)}个 | 验证集{len(val_list)}个 | 测试集{len(test_list)}个")

    # ---------- 新增：统计被跳过的图片 ----------
    skipped_due_to_boxes = 0
    train_valid = 0

    # 3. 生成训练集TXT（包含标注）
    with open(train_out, "w", encoding="utf-8") as f_train:
        for xml_file in train_list:
            line = convert_xml_to_line(xml_file, include_boxes=True)
            if line:
                f_train.write(f"{line}\n")
                train_valid += 1
            else:
                # 检查是否因为标注数量过多被跳过
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    box_count = sum(1 for _ in root.iter("object"))
                    if box_count > MAX_BOXES_PER_IMAGE:
                        skipped_due_to_boxes += 1
                except:
                    pass
        print(f"✅ 训练集TXT生成完成：{train_out}，有效标注数：{train_valid}/{len(train_list)}")

    # ---------- 新增：统计验证集被跳过的图片 ----------
    val_skipped_due_to_boxes = 0
    val_valid = 0

    # 4. 生成验证集TXT（包含标注）
    with open(val_out, "w", encoding="utf-8") as f_val:
        for xml_file in val_list:
            line = convert_xml_to_line(xml_file, include_boxes=True)
            if line:
                f_val.write(f"{line}\n")
                val_valid += 1
            else:
                # 检查是否因为标注数量过多被跳过
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    box_count = sum(1 for _ in root.iter("object"))
                    if box_count > MAX_BOXES_PER_IMAGE:
                        val_skipped_due_to_boxes += 1
                except:
                    pass
        print(f"✅ 验证集TXT生成完成：{val_out}，有效标注数：{val_valid}/{len(val_list)}")

    # 5. 生成测试集TXT（仅图片路径）
    test_images_count = 0
    test_valid = 0
    with open(test_out, "w", encoding="utf-8") as f_test:
        with open(test_images_out, "w", encoding="utf-8") as f_test_images:
            for xml_file in test_list:
                # 对于测试集，我们只获取图片路径，不包含标注
                line = convert_xml_to_line(xml_file, include_boxes=False)
                if line:
                    # test_out文件包含完整路径（用于某些用途）
                    f_test.write(f"{line}\n")
                    # test_images_out文件仅包含图片路径
                    f_test_images.write(f"{line}\n")
                    test_valid += 1
                else:
                    test_images_count += 1
            print(f"✅ 测试集TXT生成完成：{test_out}")
            print(f"✅ 测试集图片路径文件：{test_images_out}，有效图片数：{test_valid}/{len(test_list)}")

    # 6. 输出统计信息
    total_skipped = skipped_due_to_boxes + val_skipped_due_to_boxes
    if total_skipped > 0:
        print(
            f"⚠️ 标注数量超过{MAX_BOXES_PER_IMAGE}被跳过的图片：训练集{skipped_due_to_boxes}张，验证集{val_skipped_due_to_boxes}张，共{total_skipped}张")

    # 7. 生成合并的训练+验证集文件（用于训练时使用）
    combined_trainval_out = r"C:\Users\Administrator\Desktop\yolov8-pytorch-master\Datasets\Labels\custom_trainval.txt"
    with open(combined_trainval_out, "w", encoding="utf-8") as f_trainval:
        # 先写入训练集
        with open(train_out, "r", encoding="utf-8") as f_train_read:
            f_trainval.write(f_train_read.read())
        # 再写入验证集
        with open(val_out, "r", encoding="utf-8") as f_val_read:
            f_trainval.write(f_val_read.read())
    print(f"✅ 合并训练+验证集文件生成完成：{combined_trainval_out}")

    # 8. 最终统计
    print("\n" + "=" * 50)
    print("数据集划分完成！")
    print(f"训练集（带标注）: {train_valid} 张图片")
    print(f"验证集（带标注）: {val_valid} 张图片")
    print(f"测试集（仅图片）: {test_valid} 张图片")
    print(f"总计: {train_valid + val_valid + test_valid} 张图片")
    print(f"训练时使用：训练集 + 验证集 = {train_valid + val_valid} 张图片")
    print(f"测试集图片路径保存在: {test_images_out}")


if __name__ == "__main__":
    main()
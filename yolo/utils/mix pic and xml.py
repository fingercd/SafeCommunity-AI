import cv2
import xml.etree.ElementTree as ET
import os
import random


def draw_annotation_on_image(image_path, xml_path, output_path):
    """
    从XML文件读取标注信息，绘制到对应图片并保存（复用成熟逻辑，保证稳定性）
    :param image_path: 匹配到的图片路径
    :param xml_path: 对应的XML标注文件路径
    :param output_path: 标注后图片的保存路径
    """
    try:
        # 1. 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise Exception("无法读取图片，图片可能损坏或路径错误")

        # 2. 解析XML标注文件
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 3. 遍历XML中的标注对象，绘制标注框和类别名称
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))

            # 绘制红色标注框（线宽2）
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            # 绘制带黑色背景的白色类别文字，提升可读性
            text_size, _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x, text_y = xmin, ymin - 10
            if text_y < 0:
                text_y = ymax + 20
            cv2.rectangle(img, (text_x, text_y - text_size[1] - 2),
                          (text_x + text_size[0], text_y + 2), (0, 0, 0), -1)
            cv2.putText(img, class_name, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 4. 保存标注后的图片
        cv2.imwrite(output_path, img)
        print(f"✅ 处理完成：{os.path.basename(output_path)}")

    except Exception as e:
        print(f"❌ 处理失败 {os.path.basename(xml_path)}：{e}")


def random_process(xml_folder, img_folder, output_folder, random_mode="single", select_count=1):
    """
    随机选取XML文件，匹配图片并完成标注
    :param xml_folder: XML文件所在文件夹路径
    :param img_folder: 图片文件所在文件夹路径
    :param output_folder: 标注后图片输出路径
    :param random_mode: 随机模式，"single"（随机单张）或"multiple"（随机多张）
    :param select_count: 随机多张时的选取数量，仅当random_mode="multiple"时生效
    """
    # 1. 创建输出文件夹（若不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 输出文件夹不存在，已自动创建：{output_folder}")

    # 2. 筛选XML文件夹中所有合法的.xml文件，存入列表
    all_xml_files = []
    for filename in os.listdir(xml_folder):
        if filename.endswith('.xml'):
            all_xml_files.append(filename)

    # 校验：若XML文件夹中无有效XML文件，直接退出
    if not all_xml_files:
        print("⚠️  XML文件夹中未找到任何.xml文件，程序退出")
        return

    # 3. 根据随机模式，选取对应的XML文件列表
    selected_xml_files = []
    if random_mode == "single":
        # 随机选取1个XML文件（random.choice：从列表中随机选单个元素）
        selected_xml = random.choice(all_xml_files)
        selected_xml_files.append(selected_xml)
        print(f"\n🎯 随机选中的XML文件：{selected_xml}")
    elif random_mode == "multiple":
        # 随机选取多张XML文件（random.sample：从列表中随机选指定数量，不重复）
        # 校验：选取数量不能超过总XML文件数，若超过则自动调整为总数量
        valid_count = min(select_count, len(all_xml_files))
        if valid_count != select_count:
            print(f"⚠️  选取数量{select_count}超过XML文件总数{len(all_xml_files)}，自动调整为{valid_count}")
        selected_xml_files = random.sample(all_xml_files, valid_count)
        print(f"\n🎯 随机选中的XML文件列表：{selected_xml_files}")
    else:
        print("⚠️  无效的随机模式，仅支持'single'或'multiple'，程序退出")
        return

    # 4. 图片格式兼容（常见格式，支持大小写）
    img_suffixes = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

    # 5. 遍历选中的XML文件，匹配图片并处理
    for xml_filename in selected_xml_files:
        xml_basename = os.path.splitext(xml_filename)[0]
        xml_full_path = os.path.join(xml_folder, xml_filename)

        # 匹配对应图片
        matched_img_path = None
        for img_filename in os.listdir(img_folder):
            if img_filename.endswith(img_suffixes) and os.path.splitext(img_filename)[0] == xml_basename:
                matched_img_path = os.path.join(img_folder, img_filename)
                break

        # 找到图片则处理，否则提示跳过
        if matched_img_path:
            img_suffix = os.path.splitext(matched_img_path)[1]
            output_img_filename = f"{xml_basename}_annotated{img_suffix}"
            output_full_path = os.path.join(output_folder, output_img_filename)
            draw_annotation_on_image(matched_img_path, xml_full_path, output_full_path)
        else:
            print(f"⚠️  未找到与 {xml_filename} 对应的图片，跳过该XML文件")

    print("\n🎉 随机处理结束！所有标注后的图片已保存至输出文件夹。")


# ---------------------- 配置你的文件夹路径 ----------------------
if __name__ == "__main__":
    # 修改为你的XML文件夹路径
    XML_FOLDER = r"C:\Users\Administrator\Desktop\yolov8-pytorch-master\Datasets\Annotations\custom"
    # 修改为你的图片文件夹路径
    IMG_FOLDER = r"C:\Users\Administrator\Desktop\yolov8-pytorch-master\Datasets\JPEGImages\train"
    # 修改为你的输出文件夹路径（无需手动创建，代码会自动生成）
    OUTPUT_FOLDER = r"C:\Users\Administrator\Desktop\yolov8-pytorch-master\Datasets\Pre of JPEGImages"

    # 2. 配置随机模式（二选一即可）
    # 模式1：随机单张（推荐新手先测试）
    RANDOM_MODE = "multiple"
    SELECT_COUNT = 200  # 此参数对"single"模式无效，可忽略

    # 模式2：随机多张（取消注释下方两行，注释上方两行即可启用）
    # RANDOM_MODE = "multiple"
    # SELECT_COUNT = 5  # 想要随机选取的XML数量，例如5张

    # 3. 调用随机处理函数
    random_process(XML_FOLDER, IMG_FOLDER, OUTPUT_FOLDER, RANDOM_MODE, SELECT_COUNT)
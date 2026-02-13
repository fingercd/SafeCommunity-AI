# -----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# -----------------------------------------------------------------------#
import time
import torch

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    # ----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    # -------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    # -------------------------------------------------------------------------#
    crop = False
    count = False
    # ----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    # ----------------------------------------------------------------------------------------------------------#
    # ✨ 修改：替换为你的RTSP摄像头地址（核心修改）
    video_path = "rtsp://admin:Qizitong666@192.168.1.81:554/stream1"  # 例如："rtsp://admin:123456@192.168.1.60:554/stream1"
    video_save_path = ""
    video_fps = 16.0
    # ----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #
    #   test_interval和fps_image_path仅在mode='fps'有效
    # ----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path = "img/street.jpg"
    # -------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    dir_origin_path = r"C:\Users\Administrator\Desktop\yolov8-pytorch-master\Datasets\JPEGImages\val"
    dir_save_path = r"C:\Users\Administrator\Desktop\yolov8-pytorch-master\Datasets\Pre of JPEGImages"
    # -------------------------------------------------------------------------#
    #   heatmap_save_path   热力图的保存路径，默认保存在model_data下
    #
    #   heatmap_save_path仅在mode='heatmap'有效
    # -------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"
    # -------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    # -------------------------------------------------------------------------#
    simplify = True
    onnx_save_path = "model_data/models.onnx"


    def detect_batch(self, image_list):
        # image_list：PIL图像列表（批量输入）
        self.net.eval()
        image_tensors = []

        # 批量预处理（GPU执行，减少CPU-GPU传输）
        for image in image_list:
            image = image.resize((self.input_shape[1], self.input_shape[0]), Image.BICUBIC)
            image_data = np.array(image, dtype=np.float32) / 255.0
            image_data = np.transpose(image_data, (2, 0, 1))
            image_tensors.append(image_data)

        image_tensor = torch.from_numpy(np.array(image_tensors)).to(self.device)
        with torch.no_grad():
            outputs = self.net(image_tensor)  # 批量推理

        # 批量后处理（解析检测结果，参考原detect_image逻辑）
        batch_results = []
        for i in range(len(outputs[0])):
            output = [out[i:i + 1] for out in outputs]
            result = self.postprocess(output, image_list[i].size)
            batch_results.append(result)
        return batch_results

    if mode == "predict":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''

        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop=crop, count=count)
                r_image.show()

    elif mode == "video":
        # ✨ 修改：添加RTSP流读取优化参数（强制FFmpeg解码、设置缓存）
        capture = cv2.VideoCapture(
            video_path,
            cv2.CAP_FFMPEG  # 强制使用FFmpeg解码，提升RTSP兼容性
        )
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 设置缓存为1帧，降低延迟

        # ✨ 新增：创建可调整大小的窗口（解决窗口无法改大小问题）
        cv2.namedWindow("video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("video", 800, 600)  # 设置窗口初始大小（适配屏幕）

        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # ✨ 优化：如果RTSP流分辨率获取失败，手动指定默认分辨率（避免保存视频报错）
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            size = (width if width > 0 else 1280, height if height > 0 else 720)
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError(
                "未能正确读取RTSP摄像头（视频），请注意：1. RTSP地址是否正确 2. 摄像头与电脑是否在同一网络 3. 账号密码是否正确。")

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                # ✨ 新增：RTSP流断开后自动重连（提升稳定性）
                print("⚠️ RTSP流读取失败，尝试重连...")
                capture.release()
                capture = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ref, frame = capture.read()
                if not ref:
                    print("❌ 重连失败，退出检测")
                    break

            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(yolo.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:  # 按ESC退出
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')


    elif mode == "dir_predict":
        import os

        from tqdm import tqdm
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):  # 遍历每张图，img_name就是当前图片名
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                # 【关键修改1】调用detect_image时，获取预测的详细信息（不只是绘图）
                # 假设你的detect_image内部会输出b'fire 0.11'这类信息，先打印图片名
                print(f"\n【当前处理图片】：{img_name}")  # 先打印图片名，再输出预测结果
                r_image = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image, heatmap_save_path)

    elif mode == "export_onnx":
        yolo.convert_to_onnx(simplify, onnx_save_path)

    else:
        raise AssertionError(
            "Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
# Web 多路监控

Flask + MJPEG 多路 RTSP 监控，YOLO 框图 + ViT 异常初筛 + Agent 复核，流配置 JSON 持久化。

## 运行

从项目根目录执行（保证可导入 yolo、vlm、Vit/lab_anomaly）：

```bash
cd c:\Users\Administrator\Desktop\moniter
set PYTHONPATH=%CD%;%CD%\Vit
python -m web.app
```

浏览器打开 http://127.0.0.1:5000/ ，添加 RTSP 地址即可。

## 环境变量（可选）

- `YOLO_WEIGHTS`：YOLO 权重路径，默认 `yolo/logs/best_epoch_weights.pth`
- `YOLO_CLASSES`：类别文件，默认 `yolo/Class/coco_classes.txt`
- `VIT_CHECKPOINT`：ViT 已知分类器，默认 `Vit/lab_dataset/derived/known_classifier/checkpoint_best.pt`
- `VLM_MERGED`：Agent 合并后模型目录，默认 `vlm/outputs/merged`（需先执行 `vlm/train/merge_lora.py`）

## 2–3 路调试验证项

1. 用 2～3 路 RTSP（或本地视频改为 RTSP 测试）跑通页面，添加/删除流会写入 `web/config/streams.json`。
2. 停止某路：点击「停止」后该路从 pipeline 中移除（enabled=false 并重启 pipeline）；再点「启动」可重新加入。
3. 页面视频均为 640×480；YOLO 检测框、类别、置信度在画面上实时显示。
4. ViT 未达阈值（默认 0.6）时不触发 Agent；达阈值后提交 clip 给 Agent，页面状态区出现「Agent复核」及置信度、解释。
5. 单路断流或失败不影响其他路；状态区可显示错误信息。

## 后续升级到生产级视频架构的保留点

- **视频传输**：当前 `/video/<id>` 为 MJPEG；可改为 WebSocket 或 HLS/WebRTC，仅替换该路由与前端 `<img>` 消费方式，推理与 `FrameStateCache` 不变。
- **流管理与状态**：`stream_store`、`frame_state`、`runtime_manager` 与 Flask 解耦，可拆成独立服务或换用 SQLite/PostgreSQL 持久化。
- **推理编排**：ViT 阈值、Agent 冷却、clip 长度等均在配置或 `VlmReviewRuntime`/`MonitorRuntimeManager` 中可调，便于后续多路扩展与算力中心部署。

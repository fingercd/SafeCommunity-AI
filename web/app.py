"""
Flask 多路 RTSP 监控：MJPEG 视频、流增删停启、ViT+Agent 状态展示。
从项目根运行: python -m web.app  或  set PYTHONPATH=c:\\Users\\Administrator\\Desktop\\moniter && python -m web.app
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# 保证项目根与 Vit 在 path 中
WEB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_DIR.parent
VIT_DIR = PROJECT_ROOT / "Vit"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(VIT_DIR) not in sys.path:
    sys.path.insert(0, str(VIT_DIR))

from flask import Flask, Response, jsonify, request, send_from_directory

from web.services.stream_store import (
    add_stream as store_add,
    delete_stream as store_delete,
    get_stream as store_get,
    load_streams,
    update_stream as store_update,
)
from web.services.frame_state import get_frame_state
from web.services.runtime_manager import MonitorRuntimeManager

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["JSON_AS_ASCII"] = False

CONFIG_PATH = WEB_DIR / "config" / "streams.json"
frame_state = get_frame_state()
runtime_manager = MonitorRuntimeManager(
    stream_store_load=lambda: load_streams(CONFIG_PATH),
    frame_state=frame_state,
    config_path=CONFIG_PATH,
)


def _ensure_running():
    if not runtime_manager.is_running():
        runtime_manager.start()


@app.route("/")
def index():
    return send_from_directory(app.template_folder, "index.html")


@app.route("/static/<path:path>")
def static_file(path):
    return send_from_directory(app.static_folder, path)


@app.route("/api/streams", methods=["GET"])
def api_list_streams():
    streams = load_streams(CONFIG_PATH)
    return jsonify({
        "streams": streams,
        "vit_loaded": runtime_manager.is_vit_loaded(),
    })


@app.route("/api/streams", methods=["POST"])
def api_add_stream():
    data = request.get_json() or {}
    name = (data.get("name") or "").strip() or "Stream"
    rtsp_url = (data.get("rtsp_url") or "").strip()
    if not rtsp_url:
        return jsonify({"error": "rtsp_url required"}), 400
    entry = store_add(
        name=name,
        rtsp_url=rtsp_url,
        enabled=data.get("enabled", True),
        resize_width=int(data.get("resize_width", 640)),
        resize_height=int(data.get("resize_height", 480)),
        vit_threshold=float(data.get("vit_threshold", 0.6)),
        yolo_confidence=float(data.get("yolo_confidence", 0.3)),
        agent_enabled=data.get("agent_enabled", True),
        config_path=CONFIG_PATH,
    )
    _ensure_running()
    return jsonify(entry), 201


@app.route("/api/streams/<stream_id>", methods=["DELETE"])
def api_delete_stream(stream_id):
    stream = store_get(stream_id, CONFIG_PATH)
    if not stream:
        return jsonify({"error": "stream not found"}), 404
    store_delete(stream_id, CONFIG_PATH)
    rtsp_url = stream.get("rtsp_url")
    if rtsp_url:
        frame_state.remove(rtsp_url)
    runtime_manager.stop()
    runtime_manager.start()
    return jsonify({"ok": True})


@app.route("/api/streams/<stream_id>/stop", methods=["POST"])
def api_stop_stream(stream_id):
    stream = store_get(stream_id, CONFIG_PATH)
    if not stream:
        return jsonify({"error": "stream not found"}), 404
    store_update(stream_id, {"enabled": False}, CONFIG_PATH)
    rtsp_url = stream.get("rtsp_url")
    if rtsp_url:
        frame_state.set_stopped(rtsp_url)
    runtime_manager.stop()
    runtime_manager.start()
    return jsonify({"ok": True, "enabled": False})


@app.route("/api/streams/<stream_id>/start", methods=["POST"])
def api_start_stream(stream_id):
    stream = store_get(stream_id, CONFIG_PATH)
    if not stream:
        return jsonify({"error": "stream not found"}), 404
    store_update(stream_id, {"enabled": True}, CONFIG_PATH)
    runtime_manager.stop()
    runtime_manager.start()
    return jsonify({"ok": True, "enabled": True})


@app.route("/api/streams/<stream_id>", methods=["PATCH"])
def api_update_stream(stream_id):
    stream = store_get(stream_id, CONFIG_PATH)
    if not stream:
        return jsonify({"error": "stream not found"}), 404
    data = request.get_json() or {}
    updates = {}
    if "yolo_confidence" in data:
        updates["yolo_confidence"] = float(data["yolo_confidence"])
    if "vit_threshold" in data:
        updates["vit_threshold"] = float(data["vit_threshold"])
    if not updates:
        return jsonify(stream), 200
    updated = store_update(stream_id, updates, CONFIG_PATH)
    runtime_manager.stop()
    runtime_manager.start()
    return jsonify(updated or stream), 200


@app.route("/api/streams/<stream_id>/status", methods=["GET"])
def api_stream_status(stream_id):
    stream = store_get(stream_id, CONFIG_PATH)
    if not stream:
        return jsonify({"error": "stream not found"}), 404
    rtsp_url = stream.get("rtsp_url")
    status = frame_state.get_status(rtsp_url) if rtsp_url else {}
    return jsonify({
        "id": stream_id,
        "stream_id": rtsp_url,
        "vit_loaded": runtime_manager.is_vit_loaded(),
        **stream,
        "status": status,
    })


# 无帧时的占位 JPEG（1x1 黑像素）
_PLACEHOLDER_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' \",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x17\x00\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfe\x7f\xff\xd9"
)


@app.route("/video/<stream_id>")
def video_mjpeg(stream_id):
    stream = store_get(stream_id, CONFIG_PATH)
    if not stream:
        return "stream not found", 404
    rtsp_url = stream.get("rtsp_url")
    if not rtsp_url:
        return "no rtsp_url", 404

    def generate():
        import time
        _ensure_running()
        while True:
            jpeg = frame_state.get_jpeg(rtsp_url)
            payload = jpeg if jpeg else _PLACEHOLDER_JPEG
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n")
            time.sleep(0.05)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

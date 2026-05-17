"""
Flask multi-stream RTSP monitor: MJPEG video, stream CRUD, ViT+VLM status, ROI config.
Run: python -m web.app
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_DIR.parent
VIT_DIR = PROJECT_ROOT / "Vit"
for p in (str(PROJECT_ROOT), str(VIT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

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
YOLO_CLASSES_PATH = PROJECT_ROOT / "yolo" / "Class" / "coco_classes.txt"

frame_state = get_frame_state()
runtime_manager = MonitorRuntimeManager(
    stream_store_load=lambda: load_streams(CONFIG_PATH),
    frame_state=frame_state,
    config_path=CONFIG_PATH,
)


def _ensure_running():
    if not runtime_manager.is_running():
        runtime_manager.start()


# ── Pages ──

@app.route("/")
def index():
    return send_from_directory(app.template_folder, "index.html")


@app.route("/static/<path:path>")
def static_file(path):
    return send_from_directory(app.static_folder, path)


# ── Stream CRUD ──

@app.route("/api/streams", methods=["GET"])
def api_list_streams():
    streams = load_streams(CONFIG_PATH)
    return jsonify({
        "streams": streams,
        "vit_loaded": runtime_manager.is_vit_loaded(),
        "vlm_loaded": runtime_manager.is_vlm_loaded(),
        "vlm_finetuned": runtime_manager.is_vlm_finetuned(),
        "vlm_base": runtime_manager.is_vlm_base_model(),
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
        rois=data.get("rois"),
        roi_alarm_classes=data.get("roi_alarm_classes"),
        global_alarm_classes=data.get("global_alarm_classes"),
        vlm_auto_interval_sec=float(data.get("vlm_auto_interval_sec", 16.0)),
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
    allowed = {
        "yolo_confidence", "vit_threshold", "agent_enabled",
        "rois", "roi_alarm_classes", "global_alarm_classes", "name",
        "vlm_auto_interval_sec",
    }
    updates = {k: v for k, v in data.items() if k in allowed}
    if not updates:
        return jsonify(stream), 200
    updated = store_update(stream_id, updates, CONFIG_PATH)
    needs_restart = any(k in updates for k in ("rois", "roi_alarm_classes", "global_alarm_classes", "yolo_confidence", "vit_threshold", "agent_enabled", "vlm_auto_interval_sec"))
    if needs_restart:
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
        "vlm_loaded": runtime_manager.is_vlm_loaded(),
        "vlm_finetuned": runtime_manager.is_vlm_finetuned(),
        "vlm_base": runtime_manager.is_vlm_base_model(),
        **stream,
        "status": status,
    })


# ── New endpoints ──

@app.route("/api/streams/<stream_id>/snapshot", methods=["GET"])
def api_snapshot(stream_id):
    """Return latest JPEG frame for ROI drawing."""
    stream = store_get(stream_id, CONFIG_PATH)
    if not stream:
        return jsonify({"error": "stream not found"}), 404
    rtsp_url = stream.get("rtsp_url")
    if not rtsp_url:
        return jsonify({"error": "no rtsp_url"}), 404
    _ensure_running()
    jpeg = frame_state.get_jpeg(rtsp_url)
    if not jpeg:
        return jsonify({"error": "no frame available yet"}), 503
    return Response(jpeg, mimetype="image/jpeg")


@app.route("/api/classes", methods=["GET"])
def api_classes():
    """Return all YOLO detectable class names."""
    classes = []
    if YOLO_CLASSES_PATH.exists():
        with YOLO_CLASSES_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                c = line.strip()
                if c:
                    classes.append(c)
    return jsonify({"classes": classes})


# ── MJPEG ──

_PLACEHOLDER_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n"
    b"\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d"
    b"\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b"
    b"\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x17\x00\x01\x01\x01"
    b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03"
    b"\x04\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfe\x7f\xff\xd9"
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
        _ensure_running()
        while True:
            jpeg = frame_state.get_jpeg(rtsp_url)
            payload = jpeg if jpeg else _PLACEHOLDER_JPEG
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
            time.sleep(0.05)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

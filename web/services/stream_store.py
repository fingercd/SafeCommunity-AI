"""
RTSP 流配置的 JSON 持久化读写。
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, List

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "streams.json"


def _ensure_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_streams(config_path: Path | str | None = None) -> List[dict]:
    path = Path(config_path) if config_path else DEFAULT_PATH
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("streams", [])
    except Exception:
        return []


def save_streams(streams: List[dict], config_path: Path | str | None = None) -> None:
    path = Path(config_path) if config_path else DEFAULT_PATH
    _ensure_path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"streams": streams}, f, ensure_ascii=False, indent=2)


def add_stream(
    name: str,
    rtsp_url: str,
    enabled: bool = True,
    resize_width: int = 640,
    resize_height: int = 480,
    vit_threshold: float = 0.6,
    yolo_confidence: float = 0.3,
    agent_enabled: bool = True,
    config_path: Path | str | None = None,
) -> dict:
    streams = load_streams(config_path)
    stream_id = str(uuid.uuid4())[:8]
    entry = {
        "id": stream_id,
        "name": name or f"Stream-{stream_id}",
        "rtsp_url": rtsp_url.strip(),
        "enabled": bool(enabled),
        "resize_width": int(resize_width),
        "resize_height": int(resize_height),
        "vit_threshold": float(vit_threshold),
        "yolo_confidence": float(yolo_confidence),
        "agent_enabled": bool(agent_enabled),
    }
    streams.append(entry)
    save_streams(streams, config_path)
    return entry


def delete_stream(stream_id: str, config_path: Path | str | None = None) -> bool:
    streams = load_streams(config_path)
    new_list = [s for s in streams if s.get("id") != stream_id]
    if len(new_list) == len(streams):
        return False
    save_streams(new_list, config_path)
    return True


def get_stream(stream_id: str, config_path: Path | str | None = None) -> dict | None:
    for s in load_streams(config_path):
        if s.get("id") == stream_id:
            return s
    return None


def update_stream(
    stream_id: str,
    updates: dict,
    config_path: Path | str | None = None,
) -> dict | None:
    streams = load_streams(config_path)
    for i, s in enumerate(streams):
        if s.get("id") == stream_id:
            streams[i] = {**s, **{k: v for k, v in updates.items() if k != "id"}}
            save_streams(streams, config_path)
            return streams[i]
    return None


def set_stream_enabled(stream_id: str, enabled: bool, config_path: Path | str | None = None) -> bool:
    return update_stream(stream_id, {"enabled": enabled}, config_path) is not None

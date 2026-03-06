"""
RTSP 流录制：按 segment_seconds 将 RTSP 流切片保存为视频文件，
用于构建 lab_dataset/raw_videos，供后续标注与训练。
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2


def now_str() -> str:
    """当前时间戳字符串，用于文件名"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def open_capture(rtsp_url: str, timeout_sec: float) -> cv2.VideoCapture:
    """打开 RTSP 流，软超时后退出"""
    cap = cv2.VideoCapture(rtsp_url)
    # OpenCV 对 RTSP 的 timeout 支持受后端影响；这里做软超时
    t0 = time.time()
    while not cap.isOpened():
        if time.time() - t0 > timeout_sec:
            break
        time.sleep(0.2)
    return cap


def record_segments(
    rtsp_url: str,
    out_dir: Path,
    camera_id: str,
    segment_seconds: int,
    reconnect_wait_sec: float,
    open_timeout_sec: float,
    out_fps: float | None,
    out_size: tuple[int, int] | None,
    fourcc: str,
    out_ext: str,
    max_segments: int | None,
) -> None:
    """
    持续录制 RTSP 流为多段视频，每段 segment_seconds 秒，断流自动重连。
    输出写到 out_dir，每段生成一个视频文件并追加 segments_meta.jsonl 记录。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "segments_meta.jsonl"

    seg_idx = 0
    while True:
        if max_segments is not None and seg_idx >= max_segments:
            print("[OK] reach max_segments, exit.")
            return

        cap = open_capture(rtsp_url, timeout_sec=open_timeout_sec)
        if not cap.isOpened():
            print(f"[WARN] cannot open rtsp. wait {reconnect_wait_sec}s then retry.")
            time.sleep(reconnect_wait_sec)
            continue

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps <= 1e-3:
            src_fps = 25.0
        fps = float(out_fps) if out_fps is not None else float(src_fps)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            width, height = 1280, 720

        if out_size is not None:
            width, height = out_size

        seg_name = f"{camera_id}_{now_str()}_{seg_idx:06d}{out_ext}"
        seg_path = out_dir / seg_name

        writer = cv2.VideoWriter(
            str(seg_path),
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            print("[ERROR] cannot open VideoWriter. check codec/fourcc/out_ext.")
            cap.release()
            time.sleep(reconnect_wait_sec)
            continue

        frames = 0
        start_ts = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] read frame failed; close segment early.")
                break

            if out_size is not None:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

            writer.write(frame)
            frames += 1

            if time.time() - start_ts >= segment_seconds:
                break

        end_ts = time.time()
        cap.release()
        writer.release()

        rec = {
            "camera_id": camera_id,
            "segment_path": str(seg_path.as_posix()),
            "start_unix": start_ts,
            "end_unix": end_ts,
            "frames": frames,
            "fps": fps,
            "size": [width, height],
            "rtsp_url": rtsp_url,
        }
        with meta_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[OK] segment saved: {seg_path}  frames={frames}  dur={end_ts-start_ts:.1f}s")
        seg_idx += 1


def main() -> None:
    ap = argparse.ArgumentParser(description="RTSP 录制切片（用于构建 lab_dataset/raw_videos）")
    ap.add_argument("--rtsp", type=str, required=True, help="RTSP URL（含用户名密码，如需要）")
    ap.add_argument("--camera_id", type=str, default="cam0", help="摄像头ID（用于文件名与目录）")
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(Path("lab_dataset") / "raw_videos"),
        help="输出根目录（默认：lab_dataset/raw_videos）",
    )
    ap.add_argument("--segment_seconds", type=int, default=30, help="每段切片时长（秒）")
    ap.add_argument("--max_segments", type=int, default=0, help="最多录制多少段（0=无限）")
    ap.add_argument("--open_timeout_sec", type=float, default=10.0, help="打开 RTSP 超时（软超时）")
    ap.add_argument("--reconnect_wait_sec", type=float, default=5.0, help="断流后重连等待")
    ap.add_argument("--out_fps", type=float, default=0.0, help="输出FPS（0=跟随源FPS）")
    ap.add_argument("--out_width", type=int, default=0, help="输出宽（0=跟随源）")
    ap.add_argument("--out_height", type=int, default=0, help="输出高（0=跟随源）")
    ap.add_argument("--fourcc", type=str, default="mp4v", help="VideoWriter fourcc（默认 mp4v）")
    ap.add_argument("--out_ext", type=str, default=".mp4", help="输出扩展名（默认 .mp4）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.camera_id / datetime.now().strftime("%Y%m%d")
    max_segments = None if args.max_segments == 0 else int(args.max_segments)
    out_fps = None if args.out_fps == 0 else float(args.out_fps)
    out_size = None
    if args.out_width > 0 and args.out_height > 0:
        out_size = (int(args.out_width), int(args.out_height))

    record_segments(
        rtsp_url=args.rtsp,
        out_dir=out_dir,
        camera_id=args.camera_id,
        segment_seconds=int(args.segment_seconds),
        reconnect_wait_sec=float(args.reconnect_wait_sec),
        open_timeout_sec=float(args.open_timeout_sec),
        out_fps=out_fps,
        out_size=out_size,
        fourcc=str(args.fourcc),
        out_ext=str(args.out_ext),
        max_segments=max_segments,
    )


if __name__ == "__main__":
    main()


"""
将 train/val/test.json 中 clip_id 以 ucf_normal_ 开头的样本的 assistant content
（JSON 内 scene_description / action_description / reasoning）改为英文固定模板。

默认处理 data/processed/{train,val,test}.json；改前写同目录 .bak 备份。
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

EN_SCENE = (
    "A normal surveillance scene; order and activities appear unremarkable, with no visible anomalies."
)
EN_ACTION = (
    "Pedestrian and vehicle activity is consistent with everyday expectations; "
    "no conflicts or hazardous behavior observed."
)
EN_REASON = (
    "The footage shows routine daily activity with no evident abnormal behavior."
)


def fix_item(item: dict) -> bool:
    cid = item.get("clip_id", "")
    if not str(cid).startswith("ucf_normal_"):
        return False
    msgs = item.get("messages", [])
    if not msgs or msgs[-1].get("role") != "assistant":
        return False
    raw = msgs[-1].get("content", "")
    if not isinstance(raw, str):
        return False
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return False
    obj["scene_description"] = EN_SCENE
    obj["action_description"] = EN_ACTION
    obj["reasoning"] = EN_REASON
    msgs[-1]["content"] = json.dumps(obj, ensure_ascii=False)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="ucf_normal assistant 三字段英文化")
    ap.add_argument(
        "--project_root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    ap.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed",
    )
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_backup", action="store_true")
    args = ap.parse_args()

    root = args.project_root
    proc = root / args.processed_dir
    if not proc.is_dir():
        raise SystemExit(f"目录不存在: {proc}")

    for name in ("train.json", "val.json", "test.json"):
        path = proc / name
        if not path.exists():
            print(f"跳过缺失: {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise SystemExit(f"{path} 应为 JSON 数组")
        n = sum(1 for item in data if fix_item(item))
        print(f"{name}: 更新 ucf_normal assistant 条数 = {n}")
        if args.dry_run:
            continue
        if not args.no_backup:
            bak = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, bak)
            print(f"  备份 -> {bak.name}")
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  已写入 {path}")


if __name__ == "__main__":
    main()

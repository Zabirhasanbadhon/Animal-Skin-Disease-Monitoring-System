import os
import json
import time
import threading
import queue
from pathlib import Path
from datetime import datetime

from flask import Flask, Response, render_template, request, jsonify
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# ============================================================
# Config
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

LESION_MODEL_PATH = r"C:\Users\ZaBir\Desktop\Animal Disease Monitoring System\models\yolo11n.pt"

SAVE_ANNOTATED_VIDEO = True
DEFAULT_FPS          = 20.0

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# ============================================================
# Load models
# ============================================================
print("[MODEL] Loading cow detector:", LESION_MODEL_PATH)
cow_model = YOLO(LESION_MODEL_PATH)

lesion_model = None
if Path(LESION_MODEL_PATH).exists():
    print(f"[MODEL] Loading lesion detector: {LESION_MODEL_PATH}")
    lesion_model = YOLO(LESION_MODEL_PATH)
    print(f"[MODEL] Lesion classes: {lesion_model.names}")
else:
    print(f"[WARN] {LESION_MODEL_PATH} not found — "
          f"run train.py first. Lumpy detection disabled.")

# ============================================================
# Shared state
# ============================================================
state_lock = threading.Lock()
state = {
    "frame": None,
    "counts": {"Cow": 0, "Healthy": 0, "Lumpy": 0},
    "details": {"cows": [], "lesions": []},
    "processing": False,
    "video_path": None,
    "last_output": None,
}
video_queue = queue.Queue()

# ============================================================
# Annotators
# ============================================================
def _box_annotator():
    if hasattr(sv, "BoxAnnotator"):
        return sv.BoxAnnotator(thickness=2)
    return sv.BoundingBoxAnnotator(thickness=2)

cow_box_ann    = _box_annotator()
lesion_box_ann = _box_annotator()
label_ann      = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

# ============================================================
# Geometry helpers
# ============================================================
def point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def classify_cows(cow_dets, lesion_dets):
    results = []
    lesion_xyxy = lesion_dets.xyxy if lesion_dets.xyxy is not None else []

    for cow_box in (cow_dets.xyxy if cow_dets.xyxy is not None else []):
        cx1, cy1, cx2, cy2 = cow_box
        hits = 0
        for lb in lesion_xyxy:
            lx1, ly1, lx2, ly2 = lb
            lcx, lcy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
            if point_in_box(lcx, lcy, (cx1, cy1, cx2, cy2)):
                hits += 1
        status = "Lumpy" if hits > 0 else "Healthy"
        results.append({
            "box": [float(v) for v in cow_box],
            "status": status,
            "num_lesions": hits,
        })
    return results


# ============================================================
# Video writer — try codecs in order
# ============================================================
def try_open_writer(base_name, fps, size):
    attempts = [
        (OUTPUT_DIR / f"{base_name}_annotated.mp4", "mp4v"),
        (OUTPUT_DIR / f"{base_name}_annotated.mp4", "avc1"),
        (OUTPUT_DIR / f"{base_name}_annotated.avi", "XVID"),
    ]
    for path, fourcc in attempts:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            size,
        )
        if writer.isOpened():
            print(f"[SAVE] VideoWriter opened with {fourcc}: {path.name}")
            return writer, path, fourcc
        writer.release()
    return None, None, None


# ============================================================
# Worker — saves video + JSON + CSV only (no crops)
# ============================================================
def process_video_worker():
    while True:
        video_path = video_queue.get()
        if video_path is None:
            continue

        with state_lock:
            state["processing"] = True
            state["video_path"] = str(video_path)
            state["counts"] = {"Cow": 0, "Healthy": 0, "Lumpy": 0}
            state["details"] = {"cows": [], "lesions": []}
            state["last_output"] = None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[ERROR] Cannot open {video_path}")
            with state_lock:
                state["processing"] = False
            continue

        W, H = 640, 480
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

        src_fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
        if src_fps <= 0 or src_fps > 240:
            src_fps = DEFAULT_FPS

        stem      = Path(video_path).stem
        run_tag   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{stem}_{run_tag}"

        writer = None
        out_video_path = None
        codec_used = None
        if SAVE_ANNOTATED_VIDEO:
            writer, out_video_path, codec_used = try_open_writer(
                base_name, src_fps, (W, H)
            )
            if writer is None:
                print("[WARN] No working video codec — annotated video will NOT be saved.")

        out_json_path = OUTPUT_DIR / f"{base_name}_summary.json"
        out_csv_path  = OUTPUT_DIR / f"{base_name}_frames.csv"

        peak = {"Cow": 0, "Healthy": 0, "Lumpy": 0}
        timeline = []
        frame_csv_rows = ["frame,cows,healthy,lumpy,frame_total_lesions"]

        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (W, H))

                cow_res = cow_model(frame, verbose=False, conf=0.35)[0]
                cow_dets_raw = sv.Detections.from_ultralytics(cow_res)

                cow_class_id = None
                for cid, name in cow_model.names.items():
                    if str(name).lower() in ("cow", "cattle"):
                        cow_class_id = cid
                        break
                if cow_class_id is not None and cow_dets_raw.class_id is not None:
                    cow_dets = cow_dets_raw[cow_dets_raw.class_id == cow_class_id]
                else:
                    cow_dets = cow_dets_raw

                if lesion_model is not None:
                    lesion_res = lesion_model(frame, verbose=False, conf=0.30)[0]
                    lesion_dets = sv.Detections.from_ultralytics(lesion_res)
                else:
                    lesion_dets = sv.Detections.empty()

                cow_info = classify_cows(cow_dets, lesion_dets)

                counts = {"Cow": len(cow_info), "Healthy": 0, "Lumpy": 0}
                for c in cow_info:
                    counts[c["status"]] += 1

                annotated = frame.copy()

                if len(cow_info) > 0:
                    boxes = np.array([c["box"] for c in cow_info], dtype=float)
                    class_ids = np.array(
                        [0 if c["status"] == "Healthy" else 1 for c in cow_info],
                        dtype=int,
                    )
                    confs = np.ones(len(cow_info), dtype=float)
                    cow_annot_dets = sv.Detections(
                        xyxy=boxes, class_id=class_ids, confidence=confs,
                    )
                    annotated = cow_box_ann.annotate(annotated, cow_annot_dets)
                    labels = [
                        f"Cow: {c['status']} (lesions={c['num_lesions']})"
                        for c in cow_info
                    ]
                    annotated = label_ann.annotate(
                        annotated, cow_annot_dets, labels=labels
                    )

                lesion_count_frame = 0
                if (lesion_dets is not None
                        and lesion_dets.xyxy is not None
                        and len(lesion_dets.xyxy) > 0):
                    lesion_count_frame = len(lesion_dets.xyxy)
                    annotated = lesion_box_ann.annotate(annotated, lesion_dets)
                    lesion_labels = [
                        str(lesion_model.names[int(cid)])
                        for cid in lesion_dets.class_id
                    ]
                    annotated = label_ann.annotate(
                        annotated, lesion_dets, labels=lesion_labels
                    )

                if writer is not None:
                    writer.write(annotated)

                for k in peak:
                    if counts[k] > peak[k]:
                        peak[k] = counts[k]

                timeline.append({
                    "frame": frame_idx,
                    "counts": counts.copy(),
                    "cows": cow_info,
                    "frame_total_lesions": lesion_count_frame,
                })
                frame_csv_rows.append(
                    f"{frame_idx},{counts['Cow']},"
                    f"{counts['Healthy']},{counts['Lumpy']},"
                    f"{lesion_count_frame}"
                )
                frame_idx += 1

                with state_lock:
                    state["frame"] = annotated
                    state["counts"] = counts
                    state["details"] = {
                        "cows": cow_info,
                        "lesions": (
                            [[float(v) for v in b] for b in lesion_dets.xyxy]
                            if lesion_dets.xyxy is not None else []
                        ),
                    }

        except Exception as e:
            print(f"[ERROR] Worker crashed on {video_path}: {e}")

        finally:
            cap.release()
            if writer is not None:
                writer.release()

            summary = {
                "source_video":     str(video_path),
                "annotated_video":  str(out_video_path) if writer is not None else None,
                "video_codec":      codec_used,
                "summary_json":     str(out_json_path),
                "frames_csv":       str(out_csv_path),
                "frames_processed": len(timeline),
                "peak_counts":      peak,
                "final_counts":     state["counts"],
                "generated_at":     datetime.now().isoformat(timespec="seconds"),
                "timeline":         timeline,
            }
            try:
                out_json_path.write_text(json.dumps(summary, indent=2))
                print(f"[SAVE] {out_json_path.name}")
            except Exception as e:
                print(f"[WARN] Failed to write JSON summary: {e}")

            try:
                out_csv_path.write_text("\n".join(frame_csv_rows))
                print(f"[SAVE] {out_csv_path.name}")
            except Exception as e:
                print(f"[WARN] Failed to write CSV: {e}")

            if writer is not None:
                print(f"[SAVE] {out_video_path.name} ({codec_used})")

            print(f"[INFO] Finished: {video_path}")

            with state_lock:
                state["processing"] = False
                state["last_output"] = {
                    "annotated_video":  str(out_video_path) if writer is not None else None,
                    "video_codec":      codec_used,
                    "summary_json":     str(out_json_path),
                    "frames_csv":       str(out_csv_path),
                    "peak_counts":      peak,
                    "frames_processed": len(timeline),
                }


# ============================================================
# MJPEG stream
# ============================================================
def generate_frames():
    while True:
        with state_lock:
            frame = state["frame"]
        if frame is None:
            time.sleep(0.03)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        time.sleep(0.02)


# ============================================================
# Routes
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    safe = secure_filename(f.filename)
    name = f"{int(time.time())}_{safe}"
    path = UPLOAD_FOLDER / name
    f.save(str(path))
    video_queue.put(path)
    return jsonify({"message": "ok", "file": name}), 200


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/detection_counts")
def get_detection_counts():
    with state_lock:
        return jsonify({
            "counts":     state["counts"].copy(),
            "details":    state["details"],
            "processing": state["processing"],
            "video_path": state["video_path"],
        })


@app.route("/status")
def status():
    with state_lock:
        last_output = state["last_output"]
        processing  = state["processing"]
    return jsonify({
        "lesion_model": (
            "loaded" if lesion_model is not None else "missing (run train.py)"
        ),
        "cow_model": LESION_MODEL_PATH,
        "lesion_classes": (
            list(lesion_model.names.values()) if lesion_model else []
        ),
        "processing":  processing,
        "last_output": last_output,
    })


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    threading.Thread(target=process_video_worker, daemon=True).start()
    print("[START] http://0.0.0.0:5000")
    print(f"[INFO] Outputs: {OUTPUT_DIR}")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
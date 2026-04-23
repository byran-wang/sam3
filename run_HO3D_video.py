import json
import os
import sam3
import torch
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from matplotlib.patches import Rectangle
from matplotlib.widgets import TextBox


def save_prompt_to_file(prompt_file, text, points, labels, box):
    """Persist a (text, points, labels, box) prompt bundle as JSON."""
    data = {
        "text": text if text is not None else "",
        "points": [[float(x), float(y)] for x, y in (points or [])],
        "labels": [int(v) for v in (labels or [])],
        "box": None if box is None else [float(v) for v in box],
    }
    Path(prompt_file).parent.mkdir(parents=True, exist_ok=True)
    with open(prompt_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved prompt to {prompt_file}")


def load_prompt_from_file(prompt_file):
    """Load a prompt JSON saved by ``save_prompt_to_file``. Returns a dict or None."""
    if prompt_file is None or not os.path.exists(prompt_file):
        return None
    with open(prompt_file, "r") as f:
        data = json.load(f)
    text = data.get("text") or None
    if isinstance(text, str) and text.strip().lower() in {"", "none", "null"}:
        text = None
    return {
        "text": text,
        "points": [[float(x), float(y)] for x, y in data.get("points", [])],
        "labels": [int(v) for v in data.get("labels", [])],
        "box": data.get("box"),
    }


def parse_points(point_coords, point_labels):
    if len(point_coords) % 2 != 0:
        raise ValueError("point_coords must be x y pairs")
    points = [
        [float(point_coords[i]), float(point_coords[i + 1])]
        for i in range(0, len(point_coords), 2)
    ]
    labels = [int(v) for v in point_labels]
    if len(labels) != len(points):
        raise ValueError("point_labels must match number of points")
    return points, labels



def propagate_in_video(predictor, session_id, start_frame_idx=0):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            start_frame_index=start_frame_idx,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame


def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")


def save_obj_id_masks(outputs_per_frame, output_dir, video_frames_for_vis, obj_id=0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = get_mask_output_paths(video_frames_for_vis, output_dir)
    fallback_shape = None
    for frame_outputs in outputs_per_frame.values():
        if obj_id in frame_outputs:
            fallback_shape = frame_outputs[obj_id].shape
            break
    if fallback_shape is None:
        first_frame = load_frame(video_frames_for_vis[0])
        fallback_shape = first_frame.shape[:2]

    for frame_idx, frame_outputs in sorted(outputs_per_frame.items()):
        mask = frame_outputs.get(obj_id)
        if mask is None:
            mask_to_save = np.zeros(fallback_shape, dtype=np.uint8)
        else:
            mask_to_save = (mask.astype(np.uint8) * 255)
        out_path = output_paths[frame_idx]
        cv2.imwrite(str(out_path), mask_to_save)


def get_mask_output_paths(video_frames_for_vis, output_dir):
    output_dir = Path(output_dir)
    output_paths = []
    for idx, frame in enumerate(video_frames_for_vis):
        if isinstance(frame, (str, Path)):
            output_paths.append(output_dir / f"{Path(frame).stem}.png")
        else:
            output_paths.append(output_dir / f"{idx:04d}.png")
    return output_paths


def ensure_cached_frame_outputs(predictor, session_id):
    session = predictor._get_session(session_id)
    inference_state = session["state"]
    cached_outputs = inference_state.setdefault("cached_frame_outputs", {})
    for frame_idx in range(inference_state["num_frames"]):
        cached_outputs.setdefault(frame_idx, {})


def collect_points_with_labels(image, title=None):
    points_abs = []
    labels = []
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(
        title
        or "Left click: positive (1), right click: negative (0). Press Enter to finish."
    )
    ax.axis("off")

    def _on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:
            label = 1
            color = "lime"
        elif event.button == 3:
            label = 0
            color = "red"
        else:
            return
        points_abs.append([event.xdata, event.ydata])
        labels.append(label)
        ax.scatter([event.xdata], [event.ydata], c=color, s=30, marker="o")
        fig.canvas.draw_idle()

    def _on_key(event):
        if event.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key)
    plt.show()
    return points_abs, labels


def collect_text_and_box_prompt(image, title=None):
    """Collect one text prompt and one box prompt from a popup window."""
    prompt_text = [""]
    box_xyxy = [None]
    drag_start = [None]

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(
        title
        or (
            "Type text below, Shift+drag to draw a box, "
            "Middle click to clear box, Enter to finish."
        )
    )
    ax.axis("off")

    # Add text input widget below the image.
    text_ax = fig.add_axes([0.1, 0.02, 0.8, 0.05])
    text_box = TextBox(text_ax, "Text")

    box_artist = [None]

    def _on_submit(text):
        prompt_text[0] = text.strip()

    text_box.on_submit(_on_submit)

    def _clear_box():
        box_xyxy[0] = None
        if box_artist[0] is not None:
            box_artist[0].remove()
            box_artist[0] = None

    def _on_press(event):
        if event.inaxes != ax:
            return
        # Shift + left click starts bbox drawing
        if event.button == 1 and event.key == "shift":
            drag_start[0] = (event.xdata, event.ydata)
            _clear_box()
            return
        # Middle click clears current box
        if event.button == 2:
            _clear_box()
            fig.canvas.draw_idle()

    def _on_motion(event):
        if drag_start[0] is None or event.inaxes != ax:
            return
        x0, y0 = drag_start[0]
        x1, y1 = event.xdata, event.ydata
        if box_artist[0] is not None:
            box_artist[0].remove()
        box_artist[0] = ax.add_patch(
            Rectangle(
                (min(x0, x1), min(y0, y1)),
                abs(x1 - x0),
                abs(y1 - y0),
                fill=False,
                edgecolor="yellow",
                linewidth=2,
            )
        )
        fig.canvas.draw_idle()

    def _on_release(event):
        if drag_start[0] is None:
            return
        if event.inaxes != ax:
            drag_start[0] = None
            return
        x0, y0 = drag_start[0]
        x1, y1 = event.xdata, event.ydata
        drag_start[0] = None
        box_xyxy[0] = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]

    def _on_key(event):
        if event.key == "enter":
            prompt_text[0] = text_box.text.strip()
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", _on_press)
    fig.canvas.mpl_connect("motion_notify_event", _on_motion)
    fig.canvas.mpl_connect("button_release_event", _on_release)
    fig.canvas.mpl_connect("key_press_event", _on_key)
    plt.show()
    return prompt_text[0], box_xyxy[0]


def _extract_mask_from_outputs(outputs, frame_idx, prefer_obj_id=0):
    if outputs is None:
        return None
    formatted = prepare_masks_for_visualization({frame_idx: outputs.copy()})
    frame_masks = formatted.get(frame_idx, {})
    if not frame_masks:
        return None
    if prefer_obj_id in frame_masks:
        return frame_masks[prefer_obj_id]
    first_obj_id = next(iter(frame_masks))
    return frame_masks[first_obj_id]


def _run_prompt_on_first_frame(
    predictor,
    session_id,
    frame_idx,
    img_w,
    img_h,
    text_prompt,
    points_abs,
    point_labels,
    box_xyxy,
):
    _ = predictor.handle_request(
        request=dict(
            type="reset_session",
            session_id=session_id,
        )
    )
    ensure_cached_frame_outputs(predictor, session_id)
    has_text = text_prompt is not None and text_prompt.strip() != ""
    has_points = bool(points_abs)
    has_box = box_xyxy is not None
    if not (has_text or has_points or has_box):
        return None

    out = None

    # Step 1: text/box prompt (SAM3 path).
    if has_text or has_box:
        req_tb = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_idx,
        }
        if has_text:
            req_tb["text"] = text_prompt.strip()
        if has_box:
            x1, y1, x2, y2 = box_xyxy
            box_xywh = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
            req_tb["bounding_boxes"] = abs_to_rel_coords(
                [box_xywh], img_w, img_h, coord_type="box"
            )
            req_tb["bounding_box_labels"] = [1]
        out = predictor.handle_request(request=req_tb)["outputs"]
        # add_prompt(text/box) resets SAM3 internal state and clears cached_frame_outputs.
        # Rebuild the frame cache placeholders before any point-refinement prompt.
        ensure_cached_frame_outputs(predictor, session_id)

    # Step 2: point prompt (Tracker path). This cannot be in the same request
    # with text/box due the backend API constraints.
    if has_points:
        point_obj_id = 0
        if out is not None and len(out.get("out_obj_ids", [])) > 0:
            point_obj_id = int(out["out_obj_ids"][0])
        req_pt = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_idx,
            "obj_id": point_obj_id,
            "points": abs_to_rel_coords(points_abs, img_w, img_h, coord_type="point"),
            "point_labels": point_labels,
        }
        out = predictor.handle_request(request=req_pt)["outputs"]

    return out


def collect_prompts_no_preview(
    frame_for_prompt,
    *,
    title=None,
    initial_text=None,
    initial_points=None,
    initial_labels=None,
    initial_box=None,
):
    """Lightweight prompt collector (no live mask preview, no predictor required).

    Supports a text prompt, positive/negative click points, and a bounding box.
    Returns {"text", "points", "labels", "box"}.
    """
    state = {
        "text": "" if initial_text is None else str(initial_text),
        "points": [list(p) for p in (initial_points or [])],
        "labels": [int(v) for v in (initial_labels or [])],
        "box": None if initial_box is None else list(initial_box),
    }
    drag_start = [None]
    point_artists = []
    box_artist = [None]

    fig, ax = plt.subplots()
    ax.imshow(frame_for_prompt)
    tips = (
        "Left click: positive (+)\n"
        "Right click: negative (-)\n"
        "Shift + drag: draw box\n"
        "Middle click: clear all\n"
        "Enter: save & close"
    )
    ax.set_title(f"{title}\n{tips}" if title else tips)
    ax.axis("off")

    text_ax = fig.add_axes([0.1, 0.02, 0.8, 0.05])
    text_box = TextBox(text_ax, "Text")
    text_box.set_val(state["text"])

    def _clear_box():
        state["box"] = None
        if box_artist[0] is not None:
            box_artist[0].remove()
            box_artist[0] = None

    def _draw_box(box_xyxy):
        if box_xyxy is None:
            return
        x1, y1, x2, y2 = box_xyxy
        if box_artist[0] is not None:
            box_artist[0].remove()
        box_artist[0] = ax.add_patch(
            Rectangle(
                (min(x1, x2), min(y1, y2)),
                abs(x2 - x1),
                abs(y2 - y1),
                fill=False,
                edgecolor="yellow",
                linewidth=2,
            )
        )

    def _clear_all():
        state["points"].clear()
        state["labels"].clear()
        for artist in point_artists:
            artist.remove()
        point_artists.clear()
        _clear_box()
        state["text"] = ""
        text_box.set_val("")

    def _on_submit(text):
        state["text"] = text.strip()

    text_box.on_submit(_on_submit)

    def _on_press(event):
        if event.inaxes != ax:
            return
        if event.button == 1 and event.key == "shift":
            drag_start[0] = (event.xdata, event.ydata)
            _clear_box()
            return
        if event.button == 2:
            _clear_all()
            fig.canvas.draw_idle()
            return
        if event.key == "shift":
            return
        if event.button == 1:
            label, color = 1, "lime"
        elif event.button == 3:
            label, color = 0, "red"
        else:
            return
        state["points"].append([event.xdata, event.ydata])
        state["labels"].append(label)
        point_artists.append(ax.scatter([event.xdata], [event.ydata], c=color, s=30, marker="o"))
        fig.canvas.draw_idle()

    def _on_motion(event):
        if drag_start[0] is None or event.inaxes != ax:
            return
        x0, y0 = drag_start[0]
        x1, y1 = event.xdata, event.ydata
        _draw_box([x0, y0, x1, y1])
        fig.canvas.draw_idle()

    def _on_release(event):
        if drag_start[0] is None:
            return
        if event.inaxes != ax:
            drag_start[0] = None
            return
        x0, y0 = drag_start[0]
        x1, y1 = event.xdata, event.ydata
        drag_start[0] = None
        state["box"] = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        _draw_box(state["box"])
        fig.canvas.draw_idle()

    def _on_key(event):
        if event.key == "backspace" and state["points"]:
            state["points"].pop()
            state["labels"].pop()
            point_artists[-1].remove()
            point_artists.pop()
            fig.canvas.draw_idle()
        if event.key == "enter":
            state["text"] = text_box.text.strip()
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", _on_press)
    fig.canvas.mpl_connect("motion_notify_event", _on_motion)
    fig.canvas.mpl_connect("button_release_event", _on_release)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    for p, l in zip(state["points"], state["labels"]):
        color = "lime" if int(l) == 1 else "red"
        point_artists.append(ax.scatter([p[0]], [p[1]], c=color, s=30, marker="o"))
    _draw_box(state["box"])
    plt.show()
    return state


def collect_prompts_with_live_preview(
    predictor,
    session_id,
    frame_idx,
    frame_for_prompt,
    initial_text=None,
    initial_points=None,
    initial_labels=None,
    initial_box=None,
    title=None,
):
    img_h, img_w = frame_for_prompt.shape[:2]
    state = {
        "text": "" if initial_text is None else str(initial_text),
        "points": [list(p) for p in (initial_points or [])],
        "labels": [int(v) for v in (initial_labels or [])],
        "box": None if initial_box is None else list(initial_box),
        "out": None,
    }
    drag_start = [None]
    point_artists = []
    box_artist = [None]
    contour_artist = [None]
    suspend_text_callback = [False]

    fig, ax = plt.subplots()
    # Set the OS-level window title (shows in title bar / taskbar),
    # independent of the in-plot axes title set below.
    if title:
        try:
            fig.canvas.manager.set_window_title(str(title))
        except Exception:
            pass
    ax.imshow(frame_for_prompt)
    mask_artist = ax.imshow(np.zeros((img_h, img_w), dtype=np.float32), cmap="jet", alpha=0.0)
    status_text = ax.text(
        0.02,
        0.02,
        "",
        transform=ax.transAxes,
        color="white",
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=4),
    )
    tips = (
        "Tips:\n"
        "  Left click: add positive point (+)\n"
        "  Right click: add negative point (-)\n"
        "  Shift + drag: draw bounding box\n"
        "  Middle click: reset all\n"
        "  Enter: finish and save\n"
        "  Text prompt: edit in the Text box below"
    )
    ax.set_title(f"{title} {tips}" if title else tips)
    ax.axis("off")

    text_ax = fig.add_axes([0.1, 0.02, 0.8, 0.05])
    text_box = TextBox(text_ax, "Text")
    text_box.set_val(state["text"])

    def _set_status(message):
        status_text.set_text(message)

    def _clear_box():
        state["box"] = None
        if box_artist[0] is not None:
            box_artist[0].remove()
            box_artist[0] = None

    def _draw_box(box_xyxy):
        if box_xyxy is None:
            return
        x1, y1, x2, y2 = box_xyxy
        if box_artist[0] is not None:
            box_artist[0].remove()
        box_artist[0] = ax.add_patch(
            Rectangle(
                (min(x1, x2), min(y1, y2)),
                abs(x2 - x1),
                abs(y2 - y1),
                fill=False,
                edgecolor="yellow",
                linewidth=2,
            )
        )

    def _clear_all_prompts():
        state["points"].clear()
        state["labels"].clear()
        for artist in point_artists:
            artist.remove()
        point_artists.clear()
        _clear_box()
        state["text"] = ""
        suspend_text_callback[0] = True
        text_box.set_val("")
        suspend_text_callback[0] = False

    def _refresh_preview():
        try:
            out = _run_prompt_on_first_frame(
                predictor,
                session_id,
                frame_idx,
                img_w,
                img_h,
                state["text"],
                state["points"],
                state["labels"],
                state["box"],
            )
        except Exception as e:
            _set_status(f"Preview error: {e}")
            fig.canvas.draw_idle()
            return
        state["out"] = out
        mask = _extract_mask_from_outputs(out, frame_idx) if out is not None else None

        if contour_artist[0] is not None:
            if hasattr(contour_artist[0], "collections"):
                for c in contour_artist[0].collections:
                    c.remove()
            else:
                contour_artist[0].remove()
            contour_artist[0] = None

        if mask is None:
            mask_artist.set_data(np.zeros((img_h, img_w), dtype=np.float32))
            mask_artist.set_alpha(0.0)
            _set_status("No mask detected yet")
        else:
            mask_f = mask.astype(np.float32)
            mask_artist.set_data(mask_f)
            mask_artist.set_alpha(0.35)
            contour_artist[0] = ax.contour(mask_f, levels=[0.5], colors="cyan", linewidths=1.2)
            obj_count = int(len(out["out_obj_ids"])) if out is not None else 0
            _set_status(f"Detected objects: {obj_count}")
        fig.canvas.draw_idle()

    def _on_text_submit(text):
        if suspend_text_callback[0]:
            return
        state["text"] = text.strip()
        _refresh_preview()

    text_box.on_submit(_on_text_submit)
    if hasattr(text_box, "on_text_change"):
        text_box.on_text_change(_on_text_submit)

    def _on_press(event):
        if event.inaxes != ax:
            return
        if event.button == 1 and event.key == "shift":
            drag_start[0] = (event.xdata, event.ydata)
            _clear_box()
            return
        if event.button == 2:
            _clear_all_prompts()
            _refresh_preview()
            return
        if event.key == "shift":
            return
        if event.button == 1:
            label, color = 1, "lime"
        elif event.button == 3:
            label, color = 0, "red"
        else:
            return
        state["points"].append([event.xdata, event.ydata])
        state["labels"].append(label)
        point_artists.append(ax.scatter([event.xdata], [event.ydata], c=color, s=30, marker="o"))
        _refresh_preview()

    def _on_motion(event):
        if drag_start[0] is None or event.inaxes != ax:
            return
        x0, y0 = drag_start[0]
        x1, y1 = event.xdata, event.ydata
        _draw_box([x0, y0, x1, y1])
        fig.canvas.draw_idle()

    def _on_release(event):
        if drag_start[0] is None:
            return
        if event.inaxes != ax:
            drag_start[0] = None
            return
        x0, y0 = drag_start[0]
        x1, y1 = event.xdata, event.ydata
        drag_start[0] = None
        state["box"] = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        _draw_box(state["box"])
        _refresh_preview()

    def _on_key(event):
        if event.key == "backspace" and state["points"]:
            state["points"].pop()
            state["labels"].pop()
            point_artists[-1].remove()
            point_artists.pop()
            _refresh_preview()
        if event.key == "enter":
            state["text"] = text_box.text.strip()
            # Ensure latest prompt state is committed before closing the popup.
            _refresh_preview()
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", _on_press)
    fig.canvas.mpl_connect("motion_notify_event", _on_motion)
    fig.canvas.mpl_connect("button_release_event", _on_release)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    for p, l in zip(state["points"], state["labels"]):
        color = "lime" if int(l) == 1 else "red"
        point_artists.append(ax.scatter([p[0]], [p[1]], c=color, s=30, marker="o"))
    _draw_box(state["box"])
    _refresh_preview()
    plt.show()
    return state


def _first_frame_path(video_path):
    """Return the path to the first image frame under ``video_path`` (or the mp4 itself)."""
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        return video_path
    frames = glob.glob(os.path.join(video_path, "*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No .jpg frames found under {video_path}")
    try:
        frames.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    except ValueError:
        frames.sort()
    return frames[0]


def _make_single_frame_session_dir(first_path):
    """Create a temp directory containing only the first frame, so the SAM3
    session only has to load a single image. Returns the temp dir path.
    """
    import shutil
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="sam3_prompt_only_")
    suffix = Path(first_path).suffix or ".jpg"
    dst = Path(tmp_dir) / f"0000{suffix}"
    try:
        os.symlink(os.path.abspath(first_path), dst)
    except OSError:
        shutil.copyfile(first_path, dst)
    return tmp_dir


def _run_prompt_only(args):
    """--prompt_only: load SAM3 predictor ONCE, then for every (video, prompt_file)
    pair pop up a live-preview window so the user can click / confirm / re-save
    the first-frame prompt. Idempotent: existing prompt_file pre-fills the popup.
    """
    video_paths = list(args.video_path)
    prompt_files = list(args.prompt_file) if args.prompt_file else []
    if len(prompt_files) != len(video_paths):
        raise ValueError(
            f"--prompt_file count ({len(prompt_files)}) must match --video_path count ({len(video_paths)})"
        )

    def _broadcast(xs, n, name):
        if xs is None:
            return [None] * n
        xs = list(xs)
        if len(xs) == 1 and n > 1:
            return xs * n
        if len(xs) != n:
            raise ValueError(f"--{name} has {len(xs)} values; expected 1 or {n}")
        return xs

    titles = _broadcast(args.prompt_title, len(video_paths), "prompt_title")
    text_prompts = _broadcast(args.text_prompt, len(video_paths), "text_prompt")

    print(f"[prompt_only] loading SAM3 predictor (once) for {len(video_paths)} video(s)...")
    predictor = _build_predictor()
    print("[prompt_only] predictor ready.")

    for i, (vp, pf, title, tp) in enumerate(zip(video_paths, prompt_files, titles, text_prompts)):
        print(f"\n[prompt_only {i+1}/{len(video_paths)}] video={vp}  file={pf}")
        try:
            _prompt_collect_with_predictor(
                predictor,
                video_path=vp,
                prompt_file=pf,
                text_prompt=tp,
                point_coords=args.point_coords,
                point_labels=args.point_labels,
                box_coords=args.box_coords,
                title=title,
            )
        except KeyboardInterrupt:
            # Propagate Ctrl+C up so the script (and shell) can abort, instead of
            # silently skipping to the next video.
            print(f"[prompt_only] interrupted at {i+1}/{len(video_paths)}, aborting.")
            raise
        except Exception as e:
            print(f"[prompt_only {i+1}/{len(video_paths)}] FAILED: {type(e).__name__}: {e}")


def _build_predictor():
    """Load the SAM3 video predictor once. Returns the predictor instance."""
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "../")
    gpus_to_use = range(torch.cuda.device_count())
    checkpoint_path = f"{sam3_root}/sam3/model/checkpoints/sam3.pt"
    return build_sam3_video_predictor(gpus_to_use=gpus_to_use, checkpoint_path=checkpoint_path)


def _prompt_collect_with_predictor(
    predictor,
    video_path,
    prompt_file,
    text_prompt=None,
    point_coords=None,
    point_labels=None,
    box_coords=None,
    title=None,
):
    """Prompt-only collection for one video, reusing a shared predictor.

    Idempotent: if prompt_file already exists, the popup opens with the saved
    text/points/box pre-filled (so the user can just hit Enter to keep them,
    or edit and re-save).
    """
    first_path = _first_frame_path(video_path)
    if first_path.endswith(".mp4"):
        cap = cv2.VideoCapture(first_path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not read first frame from {first_path}")
        frame_for_prompt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="sam3_prompt_only_")
        first_img_path = Path(tmp_dir) / "0000.jpg"
        Image.fromarray(frame_for_prompt).save(first_img_path)
        session_dir = tmp_dir
    else:
        frame_for_prompt = load_frame(first_path)
        session_dir = _make_single_frame_session_dir(first_path)

    initial_text = text_prompt
    if initial_text is not None and str(initial_text).strip().lower() in {"", "none", "null"}:
        initial_text = None
    initial_points = []
    initial_labels = []
    if point_coords or point_labels:
        if point_coords is None or point_labels is None:
            raise ValueError("Please provide both point_coords and point_labels together.")
        initial_points, initial_labels = parse_points(point_coords, point_labels)
    initial_box = [float(v) for v in box_coords] if box_coords is not None else None

    loaded_prompt = load_prompt_from_file(prompt_file)
    if loaded_prompt is not None:
        print(f"Loaded prompt from {prompt_file}")
        initial_text = loaded_prompt["text"] if loaded_prompt["text"] is not None else initial_text
        if loaded_prompt["points"]:
            initial_points = loaded_prompt["points"]
            initial_labels = loaded_prompt["labels"]
        if loaded_prompt["box"] is not None:
            initial_box = loaded_prompt["box"]

    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=session_dir)
    )
    session_id = response["session_id"]

    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["figure.titlesize"] = 12

    state = collect_prompts_with_live_preview(
        predictor,
        session_id,
        frame_idx=0,
        frame_for_prompt=frame_for_prompt,
        initial_text=initial_text,
        initial_points=initial_points,
        initial_labels=initial_labels,
        initial_box=initial_box,
        title=title or "prompt",
    )
    save_prompt_to_file(
        prompt_file,
        state["text"],
        state["points"],
        state["labels"],
        state["box"],
    )


def _process_session(
    predictor,
    video_path,
    out_path,
    prompt_file=None,
    text_prompt=None,
    point_coords=None,
    point_labels=None,
    box_coords=None,
    check_mask_result=0,
    show_detected_obj=0,
):
    """Run SAM3 on one video with the given prompts/prompt_file. Extracted from
    main() so --batch_file can reuse the same predictor across many items."""
    # font size for axes titles
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["figure.titlesize"] = 12

    # load "video_frames_for_vis" for visualization purposes (they are not used by the model)
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        video_frames_for_vis = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
        try:
            # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
            video_frames_for_vis.sort(
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
            )
        except ValueError:
            # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
            print(
                f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
                f"falling back to lexicographic sort."
            )
            video_frames_for_vis.sort()

    expected_mask_paths = get_mask_output_paths(video_frames_for_vis, out_path)
    if expected_mask_paths and all(p.exists() for p in expected_mask_paths):
        print(f"Masks already exist in {out_path}, skipping {video_path}")
        return

    response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
        )
    )
    session_id = response["session_id"]

    frame_idx = 0

    # Start from CLI-provided prompts
    prompt_text_str = text_prompt
    if prompt_text_str is not None and str(prompt_text_str).strip().lower() in {"", "none", "null"}:
        prompt_text_str = None
    prompt_points = []
    prompt_point_labels = []
    if point_coords is not None or point_labels is not None:
        if point_coords is None or point_labels is None:
            raise ValueError("Please provide both --point_coords and --point_labels together.")
        prompt_points, prompt_point_labels = parse_points(point_coords, point_labels)
    prompt_box = None
    if box_coords is not None:
        prompt_box = [float(v) for v in box_coords]

    # Override with saved prompt file when available (unless in prompt-only mode,
    # in which case we always collect a fresh prompt interactively).
    loaded_prompt = load_prompt_from_file(prompt_file)
    if loaded_prompt is not None:
        print(f"Loaded prompt from {prompt_file}")
        prompt_text_str = loaded_prompt["text"]
        prompt_points = loaded_prompt["points"]
        prompt_point_labels = loaded_prompt["labels"]
        prompt_box = loaded_prompt["box"]
    frame_for_prompt = load_frame(video_frames_for_vis[frame_idx])
    img_h, img_w = frame_for_prompt.shape[:2]

    out = None
    if prompt_text_str is not None or prompt_points or prompt_box is not None:
        out = _run_prompt_on_first_frame(
            predictor,
            session_id,
            frame_idx,
            img_w,
            img_h,
            prompt_text_str,
            prompt_points,
            prompt_point_labels,
            prompt_box,
        )

    need_prompt_popup = (
        bool(check_mask_result)
        or out is None
        or len(out["out_obj_ids"]) == 0
    )
    # When a prompt file was loaded successfully, skip the popup unless
    # --check_mask_result asked for an interactive review.
    if loaded_prompt is not None and not check_mask_result and out is not None and len(out["out_obj_ids"]) > 0:
        need_prompt_popup = False
    final_text_prompt = prompt_text_str
    final_points = prompt_points
    final_point_labels = prompt_point_labels
    final_box = prompt_box
    if need_prompt_popup:
        prompt_state = collect_prompts_with_live_preview(
            predictor,
            session_id,
            frame_idx,
            frame_for_prompt,
            initial_text=prompt_text_str,
            initial_points=prompt_points,
            initial_labels=prompt_point_labels,
            initial_box=prompt_box,
        )
        out = _run_prompt_on_first_frame(
            predictor,
            session_id,
            frame_idx,
            img_w,
            img_h,
            prompt_state["text"],
            prompt_state["points"],
            prompt_state["labels"],
            prompt_state["box"],
        )
        final_text_prompt = prompt_state["text"]
        final_points = prompt_state["points"]
        final_point_labels = prompt_state["labels"]
        final_box = prompt_state["box"]
    else:
        print(
            "Mask detected with input prompt, skipping prompt window. "
            "Set --check_mask_result 1 to review/edit interactively."
        )
    if out is None or len(out["out_obj_ids"]) == 0:
        print("Skipping the video because no valid prompt mask was confirmed.")
        return

    # Persist the prompt so a follow-up process can reuse it.
    if prompt_file is not None:
        save_prompt_to_file(
            prompt_file,
            final_text_prompt,
            final_points,
            final_point_labels,
            final_box,
        )
    if show_detected_obj:
        visualize_formatted_frame_output(
            frame_idx,
            video_frames_for_vis,
            outputs_list=[prepare_masks_for_visualization({frame_idx: out})],
            titles=["SAM 3 Dense Tracking outputs"],
            figsize=(6, 4),
        )

    print("Prompts before propagate_in_video:")
    print(f"  text_prompt: {final_text_prompt!r}")
    print(f"  point_coords: {final_points}")
    print(f"  point_labels: {final_point_labels}")
    print(f"  box_coords: {final_box}")

    # now we propagate the outputs from frame 0 to the end of the video and collect all outputs
    outputs_per_frame = propagate_in_video(predictor, session_id, start_frame_idx=frame_idx)

    # finally, we reformat the outputs for visualization and plot the outputs every 60 frames
    outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

    plt.close("all")

    save_obj_id_masks(outputs_per_frame, out_path, video_frames_for_vis, obj_id=0)
    print(f"Saved extracted masks to: {Path(out_path).resolve()}")


def _run_batch(batch_file):
    """Load predictor ONCE and iterate through a list of items from batch_file.

    batch_file is a JSON list of dicts, each with keys:
      video_path (str, required)
      out_path (str, required)
      prompt_file (str, optional — load/save prompts here; interactive popup if missing)
      text_prompt (str, optional)
      point_coords (list[float], optional — flat x,y,x,y,...)
      point_labels (list[int], optional)
      box_coords (list[float,4], optional)
      check_mask_result (int, optional — default 0; 1 forces interactive review)
      show_detected_obj (int, optional)

    Model weights are loaded once; each item calls _process_session with the
    shared predictor, so wall time drops dramatically for many-seq runs.
    """
    with open(batch_file, "r") as f:
        items = json.load(f)
    if not isinstance(items, list) or not items:
        print(f"[batch] empty or malformed batch_file: {batch_file}")
        return

    print(f"[batch] loading SAM3 predictor (once) for {len(items)} item(s)...")
    predictor = _build_predictor()
    print("[batch] predictor ready.")

    for i, item in enumerate(items):
        vp = item.get("video_path")
        if not vp:
            print(f"[batch {i+1}/{len(items)}] skipping: missing video_path")
            continue
        # prompt-only items collect a single prompt per video, no propagation
        if item.get("prompt_only"):
            pf = item.get("prompt_file")
            if not pf:
                print(f"[batch {i+1}/{len(items)}] prompt_only but no prompt_file, skipping")
                continue
            print(f"\n[batch {i+1}/{len(items)}] prompt_only video={vp}  file={pf}")
            try:
                _prompt_collect_with_predictor(
                    predictor,
                    video_path=vp,
                    prompt_file=pf,
                    text_prompt=item.get("text_prompt"),
                    point_coords=item.get("point_coords"),
                    point_labels=item.get("point_labels"),
                    box_coords=item.get("box_coords"),
                    title=item.get("title"),
                )
            except KeyboardInterrupt:
                print(f"[batch] interrupted at {i+1}/{len(items)}, aborting.")
                raise
            except Exception as e:
                print(f"[batch {i+1}/{len(items)}] FAILED: {type(e).__name__}: {e}")
            continue
        # full inference item
        op = item.get("out_path")
        if not op:
            print(f"[batch {i+1}/{len(items)}] skipping: missing out_path for full inference")
            continue
        print(f"\n[batch {i+1}/{len(items)}] video={vp}  out={op}")
        try:
            _process_session(
                predictor,
                video_path=vp,
                out_path=op,
                prompt_file=item.get("prompt_file"),
                text_prompt=item.get("text_prompt"),
                point_coords=item.get("point_coords"),
                point_labels=item.get("point_labels"),
                box_coords=item.get("box_coords"),
                check_mask_result=int(item.get("check_mask_result", 0)),
                show_detected_obj=int(item.get("show_detected_obj", 0)),
            )
        except Exception as e:
            print(f"[batch {i+1}/{len(items)}] FAILED: {type(e).__name__}: {e}")
    print(f"\n[batch] done ({len(items)} items)")


def main(args):
    # Prompt-only mode: skip model + full-video loading, just grab the first frame.
    if args.prompt_only:
        if args.prompt_file is None:
            raise ValueError("--prompt_only requires --prompt_file to save the collected prompt.")
        _run_prompt_only(args)
        return

    # Batch mode: single predictor, multiple videos.
    if getattr(args, "batch_file", None):
        _run_batch(args.batch_file)
        return

    # Single-item mode: build predictor + run once.
    # --video_path / --prompt_file / --text_prompt are nargs="+", unwrap to scalars.
    if len(args.video_path) != 1:
        raise ValueError("Single-item inference mode requires exactly one --video_path (got "
                         f"{len(args.video_path)}). Use --batch_file for multi-video inference.")
    _scalar = lambda xs: xs[0] if xs else None
    predictor = _build_predictor()
    _process_session(
        predictor,
        video_path=args.video_path[0],
        out_path=args.out_path,
        prompt_file=_scalar(args.prompt_file),
        text_prompt=_scalar(args.text_prompt),
        point_coords=args.point_coords,
        point_labels=args.point_labels,
        box_coords=args.box_coords,
        check_mask_result=args.check_mask_result,
        show_detected_obj=args.show_detected_obj,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, nargs="+",
                        default=["/home/simba/Documents/dataset/BundleSDF/HO3D_v3/train/ABF10/rgb/"],
                        help="One or more video paths. In --prompt_only mode, paired with --prompt_file and --prompt_title.")
    parser.add_argument("--out_path", type=str, default="/home/simba/Documents/dataset/BundleSDF/HO3D_v3/train/ABF10/mask_hand/")
    parser.add_argument("--text_prompt", type=str, nargs="+", default=None,
                        help="Text prompt(s). Pass one (broadcast to all videos) or one per --video_path.")
    parser.add_argument("--point_coords", type=float, nargs="*", default=None,
                        help="Point coordinates as x1 y1 x2 y2 ... (pixel coords on frame 0).")
    parser.add_argument("--point_labels", type=int, nargs="*", default=None,
                        help="Point labels for --point_coords (1=positive, 0=negative).")
    parser.add_argument("--box_coords", type=float, nargs=4, default=None,
                        help="Bounding box on frame 0: x1 y1 x2 y2 (pixel coords).")
    parser.add_argument("--use_point_prompt_when_no_obj_detected", type=int, default=1)
    parser.add_argument("--use_both_text_and_point_prompt", type=int, default=0,
                        help="If 1, use both text prompt and point prompt together")
    parser.add_argument("--show_detected_obj", type=int, default=0)
    parser.add_argument(
        "--check_mask_result",
        type=int,
        default=0,
        help="If 1, always show the interactive prompt window to verify/edit the first-frame mask.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        nargs="+",
        default=None,
        help="Prompt JSON file(s). In --prompt_only mode, pass one per --video_path (paired by index).",
    )
    parser.add_argument(
        "--prompt_only",
        action="store_true",
        help="Collect/confirm prompts and save to --prompt_file(s), load SAM3 predictor ONCE across all --video_path entries, then exit without propagation.",
    )
    parser.add_argument(
        "--prompt_title",
        type=str,
        nargs="+",
        default=None,
        help="Popup title(s) in --prompt_only mode. Pass one (shared) or one per --video_path.",
    )
    parser.add_argument(
        "--batch_file",
        type=str,
        default=None,
        help="Path to a JSON list of items (video_path, out_path, prompt_file, text_prompt, ...). "
             "Builds the SAM3 predictor once and iterates items — saves ~(N-1)*model_load.",
    )

    args = parser.parse_args()
    main(args)
    

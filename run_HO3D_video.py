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


def collect_prompts_with_live_preview(
    predictor,
    session_id,
    frame_idx,
    frame_for_prompt,
    initial_text=None,
    initial_points=None,
    initial_labels=None,
    initial_box=None,
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
    ax.set_title(tips)
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


def main(args):
    video_path = args.video_path 
    out_path = args.out_path 


    # TODO if there is no .mp4 file, convert all the image files to a mp4 file
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "../")

    # use all available GPUs on the machine
    gpus_to_use = range(torch.cuda.device_count())
    # # use only a single GPU
    # gpus_to_use = [torch.cuda.current_device()]


    checkpoint_path = f"{sam3_root}/sam3/model/checkpoints/sam3.pt"
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use, checkpoint_path=checkpoint_path)
    

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
    prompt_text_str = args.text_prompt
    if prompt_text_str is not None and str(prompt_text_str).strip().lower() in {"", "none", "null"}:
        prompt_text_str = None
    prompt_points = []
    prompt_point_labels = []
    if args.point_coords is not None or args.point_labels is not None:
        if args.point_coords is None or args.point_labels is None:
            raise ValueError("Please provide both --point_coords and --point_labels together.")
        prompt_points, prompt_point_labels = parse_points(args.point_coords, args.point_labels)
    prompt_box = None
    if args.box_coords is not None:
        prompt_box = [float(v) for v in args.box_coords]
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
        bool(args.check_mask_result)
        or out is None
        or len(out["out_obj_ids"]) == 0
    )
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
    if args.show_detected_obj:
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="/home/simba/Documents/dataset/BundleSDF/HO3D_v3/train/ABF10/rgb/")
    parser.add_argument("--out_path", type=str, default="/home/simba/Documents/dataset/BundleSDF/HO3D_v3/train/ABF10/mask_hand/")
    parser.add_argument("--text_prompt", type=str, default=None)
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

    args = parser.parse_args()
    main(args)
    

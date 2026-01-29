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



def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
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


def try_point_prompt(
    predictor,
    session_id,
    frame_idx,
    video_frames_for_vis,
    use_point_prompt_when_no_obj_detected,
):
    if not use_point_prompt_when_no_obj_detected:
        return None, False, False
    obj_id = 0
    frame_for_prompt = load_frame(video_frames_for_vis[frame_idx])
    IMG_HEIGHT, IMG_WIDTH = frame_for_prompt.shape[:2]
    points_abs, labels = collect_points_with_labels(frame_for_prompt)
    if not points_abs:
        return None, True, False
    points_tensor = torch.tensor(
        abs_to_rel_coords(points_abs, IMG_WIDTH, IMG_HEIGHT, coord_type="point"),
        dtype=torch.float32,
    )
    points_labels_tensor = torch.tensor(labels, dtype=torch.int32)

    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_idx,
            points=points_tensor,
            point_labels=points_labels_tensor,
            obj_id=obj_id,
        )
    )
    return response["outputs"], False, True


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

    # note: in case you already ran one text prompt and now want to switch to another text prompt
    # it's required to reset the session first (otherwise the results would be wrong)
    _ = predictor.handle_request(
        request=dict(
            type="reset_session",
            session_id=session_id,
        )
    )

    prompt_text_str = args.text_prompt
    frame_idx = 0  # add a text prompt on frame 0
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_idx,
            text=prompt_text_str,
        )
    )
    out = response["outputs"]

    plt.close("all")

    used_point_prompt = False
    if len(out['out_obj_ids']) == 0:
        print(f"No objects detected from the text prompt {prompt_text_str}")
        ensure_cached_frame_outputs(predictor, session_id)
        out, should_skip, used_point_prompt = try_point_prompt(
            predictor,
            session_id,
            frame_idx,
            video_frames_for_vis,
            args.use_point_prompt_when_no_obj_detected,
        )
        if should_skip:
            print("No points selected, skipping the video.")
            return
        if out is None:
            print(f"Skipping the video {video_path} and there is no object detected for the point prompt")
            return
    else:
        print(f"{len(out['out_obj_ids'])} objects detected from the text prompt {prompt_text_str}")
        # Optionally add point prompts on top of text detection
        if args.use_both_text_and_point_prompt:
            print("Adding point prompts to refine/extend the text detection...")
            ensure_cached_frame_outputs(predictor, session_id)
            point_out, should_skip, used_point_prompt = try_point_prompt(
                predictor,
                session_id,
                frame_idx,
                video_frames_for_vis,
                use_point_prompt_when_no_obj_detected=True,
            )
            if not should_skip and point_out is not None:
                out = point_out  # Use the updated output with point prompts
    if args.show_detected_obj:
        visualize_formatted_frame_output(
            frame_idx,
            video_frames_for_vis,
            outputs_list=[prepare_masks_for_visualization({frame_idx: out})],
            titles=["SAM 3 Dense Tracking outputs"],
            figsize=(6, 4),
        )

    # now we propagate the outputs from frame 0 to the end of the video and collect all outputs
    outputs_per_frame = propagate_in_video(predictor, session_id)

    # finally, we reformat the outputs for visualization and plot the outputs every 60 frames
    outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

    plt.close("all")

    save_obj_id_masks(outputs_per_frame, out_path, video_frames_for_vis, obj_id=0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="/home/simba/Documents/dataset/BundleSDF/HO3D_v3/train/ABF10/rgb/")
    parser.add_argument("--out_path", type=str, default="/home/simba/Documents/dataset/BundleSDF/HO3D_v3/train/ABF10/mask_hand/")
    parser.add_argument("--text_prompt", type=str, default="right hand")
    parser.add_argument("--use_point_prompt_when_no_obj_detected", type=int, default=1)
    parser.add_argument("--use_both_text_and_point_prompt", type=int, default=0,
                        help="If 1, use both text prompt and point prompt together")
    parser.add_argument("--show_detected_obj", type=int, default=0)

    args = parser.parse_args()
    main(args)
    
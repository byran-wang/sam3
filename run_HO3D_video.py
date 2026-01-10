import os
import sam3
import torch
from sam3.model_builder import build_sam3_video_predictor

import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path


obj2text_prompt = {
    'AP': 'blue pitcher base',
    'MPM': 'potted meatal can',
    'SB': 'white clean bleach',
    'SM': 'yellow mustard bottle',
    "ABF": "white clean bleach",
    "BB": "yello banana",
    "GPMF": "potted meatal can",
    "GSF": "scissors",
    "MC": "red cracker_box",
    "MDF": "orange power drill",
    "ND": "orange power drill",
    "SMu": "red mug",
    "SS": "yellow sugar box",
    "ShSu": "yellow sugar box",
    "SiBF": "yellow banana",
    "SiS": "yellow sugar box",         
}

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
        out_path = output_dir / f"{frame_idx:04d}.png"
        cv2.imwrite(str(out_path), mask_to_save)


def main(args):
    video_path = args.video_path 
    out_path = args.out_path 
    scene_name = Path(video_path).parents[0].name
    #TODO: obj name is the first part of the scene name without the number, e.g. scene_name="ABF10", obj_name="ABF"
    # TODO: obj name is the first part of the scene name without the number, e.g. scene_name="ABF10", obj_name="ABF"

    obj_name = scene_name.rstrip("0123456789")

    prompt_text_str = obj2text_prompt[obj_name]


    # TODO if there is no .mp4 file, convert all the image files to a mp4 file
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "../")

    # use all available GPUs on the machine
    gpus_to_use = range(torch.cuda.device_count())
    # # use only a single GPU
    # gpus_to_use = [torch.cuda.current_device()]


    checkpoint_path = f"{sam3_root}/sam3/model/checkpoints/sam3.pt"
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use, checkpoint_path=checkpoint_path)
    
    from sam3.visualization_utils import (
        load_frame,
        prepare_masks_for_visualization,
        visualize_formatted_frame_output,
    )


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
    # visualize_formatted_frame_output(
    #     frame_idx,
    #     video_frames_for_vis,
    #     outputs_list=[prepare_masks_for_visualization({frame_idx: out})],
    #     titles=["SAM 3 Dense Tracking outputs"],
    #     figsize=(6, 4),
    # )

    # now we propagate the outputs from frame 0 to the end of the video and collect all outputs
    outputs_per_frame = propagate_in_video(predictor, session_id)

    # finally, we reformat the outputs for visualization and plot the outputs every 60 frames
    outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

    plt.close("all")

    save_obj_id_masks(outputs_per_frame, args.out_path, video_frames_for_vis, obj_id=0)

    # vis_frame_stride = 60
    # for frame_idx in range(0, len(outputs_per_frame), vis_frame_stride):
    #     visualize_formatted_frame_output(
    #         frame_idx,
    #         video_frames_for_vis,
    #         outputs_list=[outputs_per_frame],
    #         titles=["SAM 3 Dense Tracking outputs"],
    #         figsize=(6, 4),
    #     )


    # # we pick id 2, which is the dancer in the front
    # obj_id = 1
    # response = predictor.handle_request(
    #     request=dict(
    #         type="remove_object",
    #         session_id=session_id,
    #         obj_id=obj_id,
    #     )
    # )

    # # now we propagate the outputs from frame 0 to the end of the video and collect all outputs
    # outputs_per_frame = propagate_in_video(predictor, session_id)

    # # finally, we reformat the outputs for visualization and plot the outputs every 60 frames
    # outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

    # vis_frame_stride = 60
    # plt.close("all")
    # for frame_idx in range(0, len(outputs_per_frame), vis_frame_stride):
    #     visualize_formatted_frame_output(
    #         frame_idx,
    #         video_frames_for_vis,
    #         outputs_list=[outputs_per_frame],
    #         titles=["SAM 3 Dense Tracking outputs"],
    #         figsize=(6, 4),
    #     )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="/home/simba/Documents/dataset/BundleSDF/HO3D_v3/train/ABF10/rgb/")
    parser.add_argument("--out_path", type=str, default="/home/simba/Documents/dataset/BundleSDF/HO3D_v3/train/ABF10/mask_hand/")
    parser.add_argument("--text_prompt", type=str, default="right hand")

    args = parser.parse_args()
    main(args)
    
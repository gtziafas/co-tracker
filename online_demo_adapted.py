# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn.functional as F
import argparse
import imageio.v3 as iio
import numpy as np
from PIL import Image

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor


DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/apple.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")

    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.to(DEFAULT_DEVICE)

    window_frames = []

    # init_coords_yx = [
    #     [0, 224, 462],
    #     [0, 272, 465],
    #     [0, 299, 457],
    #     [0, 274, 163],
    #     [0, 245, 192],
    #     [0, 279, 201],
    # ]
    init_coords_yx = [[192, 347],
 [199, 366],
 [202, 337],
 [156, 339],
 [152, 361],
 [497, 318],
 [550, 302],
 [440, 279],
 [486, 255],
 [485, 370],
 [452, 317],
 [514, 341]]

 
    #queries = [[0,x[2],x[1]] for x in init_coords_yx]
    queries = [[0,x[0],x[1]] for x in init_coords_yx]
    num_points = len(queries)

    def _process_step(window_frames, queries, is_first_step, grid_size, grid_query_frame):
        queries = torch.tensor(queries, device=DEFAULT_DEVICE).float()
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  / 255. # (1, T, 3, H, W)
        # print(video_chunk.shape, video_chunk.max(), video_chunk.dtype)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            queries=queries[None],
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
            one_frame=False
        )

    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    for i, frame in enumerate(
        iio.imiter(
            args.video_path,
            plugin="FFMPEG",
        )
    ):
        #frame = np.array(Image.fromarray(frame).resize((640,480)))
        if i % model.step == 0 and i != 0:
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                queries,
                is_first_step,
                grid_size=args.grid_size,
                grid_query_frame=args.grid_query_frame,
            )
            is_first_step = False
        window_frames.append(frame)
    # Processing the final video frames in case video length is not a multiple of model.step
    pred_tracks, pred_visibility = _process_step(
        window_frames[-(i % model.step) - model.step - 1 :],
        queries,
        is_first_step,
        grid_size=args.grid_size,
        grid_query_frame=args.grid_query_frame,
    )

    print("Tracks are computed")
    #print(frame.shape, frame.dtype, frame.max())
    #print(pred_tracks.shape)
    pred_tracks = pred_tracks[:, :, 0:num_points, :]
    pred_visibility = pred_visibility[:, :, 0:num_points]
    #print(pred_tracks.shape, pred_tracks.max())
    
    # dump raw tracks
    save_path = args.video_path.replace("rgb.mp4", "saved_tracks.npy").replace("/videos", "")
    np.save(save_path, pred_tracks[0].detach().cpu().numpy())
    save_path = save_path.replace("saved_tracks", "saved_visibility")
    np.save(save_path, pred_visibility[0].detach().cpu().numpy())

    # save a video with predicted tracks
    seq_name = args.video_path.split("/")[-1]
    video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(
        0, 3, 1, 2
    )[None][:200]
    save_path = save_path.replace("/saved_visibility.npy", "/videos")
    vis = Visualizer(save_dir=save_path, pad_value=120, linewidth=3, fps=30)
    vis.visualize(
        video, pred_tracks, pred_visibility, query_frame=args.grid_query_frame
    )

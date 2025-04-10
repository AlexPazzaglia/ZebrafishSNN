
import os
import imageio.v2 as imageio

from tqdm import tqdm

def make_video(
    results_path,
    video_speeds = [0.25, 0.50, 1.0],
    video_skips = 1,
):
    ''' Make video from png files '''

    # Get available files (sorted by iteration, e.g. frame_138.png)
    get_iteration = lambda f: int(f.split("_")[-1][:-4])

    frame_files = [f"{results_path}/{f}" for f in os.listdir(results_path)]
    frame_files = sorted(frame_files, key=get_iteration)

    # Parameters
    time_step = 0.001
    save_skip = get_iteration(frame_files[1]) - get_iteration(frame_files[0])

    # Load frames
    ims = []
    for f in tqdm(frame_files[::video_skips], desc="Loading frames"):
        ims.append(imageio.imread(f))

    # Make video
    # save_path  = os.path.dirname(results_path)
    save_path  = results_path
    video_tag  = os.path.basename(save_path)

    for video_speed in video_speeds:
        video_fps  = video_speed / ( time_step * save_skip * video_skips )
        video_name = f'video_{video_tag}_{video_speed:.2f}x.mp4'
        video_path = f'{save_path}/{video_name}'

        print(f"Making video: {video_name}")
        imageio.mimwrite(video_path, ims, fps=video_fps)
        print(f"Video saved at: {video_path}")

    return


def main():

    restuls_root = '/data/pazzagli/simulation_results/frames'

    folder_names = [
        # "dynamic_water_vortices_closed_loop_fixed_head_030/20250122-200010",
        # "closed_loop_in_line_105",
        "open_loop_in_line_105",
    ]

    for folder_name in folder_names:
        results_path = f'{restuls_root}/{folder_name}'
        make_video(results_path)


if __name__ == "__main__":
    main()


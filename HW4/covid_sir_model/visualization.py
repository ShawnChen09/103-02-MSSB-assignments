import os
import subprocess

import matplotlib.pyplot as plt
from config import STEP_PER_DAY


def plot_results(history):
    days = [i / STEP_PER_DAY for i in range(len(history["s"]))]

    plt.figure(figsize=(10, 6))
    plt.plot(days, history["s"], label="Susceptible", color="blue")
    plt.plot(days, history["i"], label="Infected", color="red")
    plt.plot(days, history["r"], label="Recovered", color="green")
    plt.plot(days, history["d"], label="Dead", color="black")

    plt.xlabel("Time (days)")
    plt.ylabel("Population")
    plt.title("Epidemic Curve for COVID-19 Simulation")
    plt.legend()
    plt.grid(True, alpha=0.3)

    max_days = max(days)
    tick_spacing = (
        10
        if max_days > 100
        else 5
        if max_days > 30
        else 2
        if max_days > 10
        else 1
    )
    plt.xticks(range(0, int(max_days) + tick_spacing, tick_spacing))

    plt.tight_layout()
    plt.show()


def combine_plot_results(history_results):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()
    for i, (args, history) in enumerate(history_results.items()):
        days = [i / STEP_PER_DAY for i in range(len(history["s"]))]
        ax = axes[i]
        ax.plot(days, history["s"], label="Susceptible", color="blue")
        ax.plot(days, history["i"], label="Infected", color="red")
        ax.plot(days, history["r"], label="Recovered", color="green")
        ax.plot(days, history["d"], label="Dead", color="black")
        ax.set_title(args)
        ax.grid(True, alpha=0.3)
        # if i == 2:
        #     ax.set_legend()

        max_days = max(days)
        tick_spacing = (
            10
            if max_days > 100
            else 5
            if max_days > 30
            else 2
            if max_days > 10
            else 1
        )
        axes[i].set_xticks(
            range(0, int(max_days) + tick_spacing, tick_spacing)
        )

    fig.supxlabel("Time (days)")
    fig.supylabel("Population")
    plt.tight_layout()
    plt.show()


def create_video(
    frames_dir="frames",
    output_file="covid_simulation.mp4",
    framerate=30,
):
    """Create a video from saved frames using FFmpeg

    Args:
        frames_dir: Directory containing the frames
        output_file: Output video file name
        framerate: Frames per second in the output video

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if FFmpeg is installed
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)

        # Create video from frames
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-framerate",
            str(framerate),
            "-i",
            os.path.join(frames_dir, "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output_file,
        ]
        subprocess.run(cmd, check=True)
        print(f"Video created successfully: {output_file}")
        return True
    except subprocess.CalledProcessError:
        print(
            "Error: FFmpeg is not installed or an error occurred during video creation."
        )
        print("To install FFmpeg, visit: https://ffmpeg.org/download.html")
        return False
    except Exception as e:
        print(f"Error creating video: {e}")
        return False

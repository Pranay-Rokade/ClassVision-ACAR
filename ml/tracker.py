import subprocess

def convert_to_streamable_mp4(input_path, output_path):
    command = [
        "ffmpeg",
        "-y",  # overwrite if exists
        "-i", input_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        "-movflags", "+faststart",
        output_path
    ]
    subprocess.run(command, check=True)
convert_to_streamable_mp4("analyzed_using phone.mp4", "final_output.mp4")

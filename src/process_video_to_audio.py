import os
import subprocess

files = os.listdir("ML Playlist")
# print(files)

for file in files:
    filename = file.split(" [")[0]
    print(filename)
    subprocess.run(["ffmpeg", "-i", f"ML Playlist/{file}", f"audios/{filename}.mp3"])

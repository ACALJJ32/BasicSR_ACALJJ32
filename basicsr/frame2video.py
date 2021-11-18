import os, cv2
import os.path as osp
import shutil
import sys

def frames_to_video(root_path, video_path):
    # get video name
    video_name = osp.basename(video_path).split(".")[0]

    # get video path
    sr_video = osp.join(root_path, "sr_video" ,osp.basename(video_path))
    sr_video_frame = osp.join(root_path, "sr_video" , video_name)

    temp_video_no_audio = osp.join(root_path, "sr_video" , (video_name + "_no_audio.mp4"))

    # get fps
    lr_video = cv2.VideoCapture(video_path)
    fps = lr_video.get(cv2.CAP_PROP_FPS)

    # frame to video
    os.system('ffmpeg -y -r {} -i {}/%8d.png -c:v libx264 -pix_fmt yuv420p {}'.format(fps, sr_video_frame, temp_video_no_audio))

    # conbine audio and video
    temp_audio_file = osp.join(root_path, "sr_video", video_name + "_audio.mkv")
    if osp.exists(temp_audio_file):
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(temp_video_no_audio, temp_audio_file, sr_video))
        os.remove(temp_audio_file) # remove temp audio
        os.remove(temp_video_no_audio)
    else:
        os.rename(temp_video_no_audio, sr_video)
        if osp.exists(temp_video_no_audio):
            os.remove(temp_video_no_audio)

    # remove temp folder
    temp_lr_frame = osp.join(root_path, "sr_video", "temp_lr_frame")
    if osp.exists(temp_lr_frame):
        shutil.rmtree(temp_lr_frame)
    if osp.exists(sr_video_frame):
        shutil.rmtree(sr_video_frame)

if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    frames_to_video(root_path, sys.argv[1])
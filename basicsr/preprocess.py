import os.path as osp
import os, cv2, sys

def preprocess(root_path, video):
    # split video to frames and mkdir
    video = osp.join(root_path, video)

    lr_video_path = osp.join(root_path, "lr_video")
    sr_video_path = osp.join(root_path, "sr_video")

    video_list = [w for w in os.listdir(lr_video_path) if w.endswith((".mp4",".mov"))]

    assert len(video_list) > 0, print("No file in workspace.")

    # choose first video as default
    video_name = osp.basename(video).split(".")[0]  # get video name

    lr_video = osp.join(root_path, lr_video_path, video)
    sr_video = osp.join(root_path, sr_video_path, video_name)
    os.makedirs(sr_video, exist_ok=True)

    # extract audio from video
    tempAudioFileName = video_name + '_audio.mkv'
    tempAudioFileName = osp.join(sr_video_path, tempAudioFileName)

    print("audio: ", tempAudioFileName)
    if not osp.exists(tempAudioFileName):
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(lr_video, tempAudioFileName))

    # video to frames
    tempLrFrames = osp.join(sr_video_path, "temp_lr_frame") # make a temp frame folder
    if not osp.exists(tempLrFrames):
        os.mkdir(tempLrFrames)

    cap = cv2.VideoCapture(lr_video)
    frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # get video frames
    start = 0
    while True:
        success, frame = cap.read()
        if success:
            frame_name = osp.join(tempLrFrames, "{:08d}.png".format(start))
            print("imwrite:{} | total: {}".format(osp.basename(frame_name), int(frame_num)))
            if not osp.exists(frame_name):
                cv2.imwrite(frame_name, frame)
            start += 1
        else: break

if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    preprocess(root_path, sys.argv[1])
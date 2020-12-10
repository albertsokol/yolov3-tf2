import os
import cv2
from tqdm import tqdm


def extract_images(path_in, path_out):
    """
    Simple function that uses cv2 to convert videos into frames.

    Parameters
    ----------
    path_in: str: a path to the video that you'd like to convert to frames
    path_out: str: the directory to save the image to. Will be created if it does not exist already. Path must end in /
    """
    print(f'converting {path_in} video to {path_out} folder')
    if not os.path.isdir(path_out):
        os.mkdir(path_out)

    vid_cap = cv2.VideoCapture(path_in)
    count = 1

    success = True
    while success:
        success, image = vid_cap.read()
        try:
            cv2.imwrite(f'{path_out}{count:05d}.jpg', image)  # save frame as JPEG file
        except cv2.error:
            print(f'Completed converting video {path_in} to frames, saved at {path_out}')
            break
        count += 1
        if count % 100 == 0:
            print('Successfully read another 100 frames')


def create_video(path_in, path_out, fps):
    """
    Simple function that uses cv2 to convert a directory of frames into an mp4 video.

    Parameters
    ----------
    path_in: str: a path to the directory of frames that you'd like to convert into a video
    path_out: str: the path to save the video to
    fps: int: the frames per second to render the video at
    """
    print(f'converting {path_in} folder to {path_out} video')
    images = sorted(os.listdir(path_in))
    frame = cv2.imread(os.path.join(path_in, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in tqdm(images):
        video.write(cv2.imread(os.path.join(path_in, image)))

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    extract_images('/home/y4tsu/Videos/bhutan.mp4', '/home/y4tsu/Videos/bhutan/')
    create_video('/home/y4tsu/Videos/bhutan_yolo/', '/home/y4tsu/Videos/bhutan_yolo.mp4', 30)

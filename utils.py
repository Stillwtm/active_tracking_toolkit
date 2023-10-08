import cv2
import numpy as np

def show_img(img, bbox=None, win_name='frame'):
    """display image and draw bounding box

    Args:
        img (ndarray): HxWxC, in BGR
        bbox (ndarray): (4,)
    """
    img_dis = img.copy()
    if bbox is not None:
        bbox = bbox.astype(np.int32)
        cv2.rectangle(img_dis, bbox[:2], bbox[:2] + bbox[2:], (0, 0, 255), 2)
    cv2.imshow(win_name, img_dis)
    cv2.waitKey(1)

    return img_dis

def make_video(imgs, video_name):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 15
    height, width = imgs[0].shape[:2]
    video_writer = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
    for img in imgs:
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video_writer.write(img)
    video_writer.release()
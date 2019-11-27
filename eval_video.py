import argparse
import os

import cv2
import logging
from tracker.mot_tracker import OnlineTracker

from utils import visualization as vis

from utils.log import logger
from utils.timer import Timer

import numpy as np
from ml_serving.drivers import driver


def detect_persons_tf(drv, frame, threshold = .5):
    input_name, input_shape = list(drv.inputs.items())[0]
    inference_frame = np.expand_dims(frame, axis=0)
    outputs = drv.predict({input_name: inference_frame})
    boxes = outputs["detection_boxes"].copy().reshape([-1, 4])
    scores = outputs["detection_scores"].copy().reshape([-1])
    classes = np.int32((outputs["detection_classes"].copy())).reshape([-1])

    cropped_scores = scores > threshold
    boxes = boxes[cropped_scores]
    scores = scores[cropped_scores]
    classes = classes[cropped_scores]

    human_classes = classes == 1
    boxes = boxes[human_classes]
    scores = scores[human_classes]

    boxes[:, 0] *= frame.shape[0]
    boxes[:, 1] *= frame.shape[1]
    boxes[:, 2] *= frame.shape[0]
    boxes[:, 3] *= frame.shape[1]
    boxes[:,[0, 1, 2, 3]] = boxes[:,[1, 0, 3, 2]]

    tlwhs = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2]-boxes[:, 0], boxes[:, 3]-boxes[:, 1]]).transpose()

    return tlwhs, scores


def eval_video(**kwargs):

    logger.setLevel(logging.INFO)

    cap = cv2.VideoCapture(kwargs['video_source'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # int(cap.get(cv2.CAP_PROP_FOURCC))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = -1
    iter_count = 0
    each_frame = kwargs['each_frame']
    save_dir = kwargs['save_dir']
    frames_limit = kwargs['frames_limit']

    video_writer = None
    video_output = kwargs['video_output']
    if video_output is not None:
        logger.info(f'Write video to {video_output} ({width}x{height}, {fps/each_frame} fps) ...' )
        video_writer = cv2.VideoWriter(video_output, fourcc, fps / each_frame, frameSize=(width, height))

    tracker = OnlineTracker(**kwargs)
    timer = Timer()
    results = []
    wait_time = 1

    drv = driver.load_driver('tensorflow')

    logger.info(f'init person detection driver...')
    person_detect_driver = drv()
    person_detect_model = kwargs['person_detect_model']
    logger.info(f'loading person detection model {person_detect_model}...')
    person_detect_driver.load_model(person_detect_model)
    logger.info(f'person detection model {person_detect_model} loaded')

    try:
        while True:

            frame_count += 1
            if frames_limit is not None and frame_count > frames_limit:
                logger.warn('frames limit {} reached'.format(frames_limit))
                break
            if frame_count % each_frame > 0:
                continue

            # read each X bgr frame
            frame = cap.read()  # bgr
            if isinstance(frame, tuple):
                frame = frame[1]
            if frame is None:
                logger.warn('video capturing finished')
                break

            if iter_count % 20 == 0:
                logger.info('Processing frame {} (iteration {}) ({:.2f} fps)'.format(
                    frame_count, iter_count, 1. / max(1e-5, timer.average_time)))

            det_tlwhs, det_scores = detect_persons_tf(person_detect_driver, frame, threshold=.5)

            # run tracking
            timer.tic()
            online_targets = tracker.update(frame, det_tlwhs, None)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                online_tlwhs.append(t.tlwh)
                online_ids.append(t.track_id)
            timer.toc()

            # save results
            frame_id = frame_count  # or make it incremental?
            results.append((frame_id + 1, online_tlwhs, online_ids))

            online_im = vis.plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)

            for tlwh in det_tlwhs:
                cv2.rectangle(
                    online_im,
                    (tlwh[0], tlwh[1]),  # (left, top)
                    (tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]),  # (right, bottom)
                    (0, 255, 0),
                    1,
                )

            if kwargs['show_image']:
                cv2.imshow('online_im', online_im)
            if save_dir is not None:
                save_to = os.path.join(save_dir, '{:05d}.jpg'.format(frame_id))
                cv2.imwrite(save_to, online_im)

            if video_writer is not None:
                video_writer.write(cv2.resize(online_im, (width, height)))

            key = cv2.waitKey(wait_time)
            key = chr(key % 128).lower()
            if key in [ord('q'), 202, 27]:  # 'q' or Esc or 'q' in russian layout
                exit(0)
            elif key == 'p':
                cv2.waitKey(0)
            elif key == 'a':
                wait_time = int(not wait_time)

            iter_count += 1

    except (KeyboardInterrupt, SystemExit) as e:
        logger.info('Caught %s: %s' % (e.__class__.__name__, e))
    finally:
        cv2.destroyAllWindows()
        if video_writer is not None:
            logger.info('Written video to %s.' % video_output)
            video_writer.release()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'video_source',
        type=str,
        help='Video source',
    )
    parser.add_argument(
        '--person_detect_model',
        type=str,
        default='/assets/faster_rcnn_resnet101_coco_2018_01_28/saved_model',
        help='Person detection model',
    )
    parser.add_argument(
        '--frames_limit',
        type=int,
        default=None,
        help='Frames (whole video, not only processed) limit',
    )
    parser.add_argument(
        '--each_frame',
        type=int,
        default=5,
        help='Process each X frame',
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Save result to dir',
    )
    parser.add_argument(
        '--video_output',
        type=str,
        default=None,
        help='Save result video to dir',
    )
    parser.add_argument(
        '--tracker_min_cls_score',
        type=float,
        default=0.4,
    )
    parser.add_argument(
        '--tracker_min_ap_dist',
        type=float,
        default=0.64,
    )
    parser.add_argument(
        '--tracker_max_time_lost',
        type=int,
        default=30,
    )
    parser.add_argument(
        '--tracker_squeezenet_ckpt',
        type=str,
        default='/assets/squeezenet_small40_coco_mot16_ckpt_10.h5',
        help='Squeezenet path',
    )
    parser.add_argument(
        '--tracker_googlenet_ckpt',
        type=str,
        default='/assets/googlenet_part8_all_xavier_ckpt_56.h5',
        help='Googlenet path',
    )
    parser.add_argument(
        '--tracker_no_tracking',
        action='store_true',
    )
    parser.add_argument(
        '--tracker_no_refind',
        action='store_true',
    )

    args = parser.parse_args()
    kwargs = vars(args)
    kwargs['show_image'] = False

    eval_video(**kwargs)


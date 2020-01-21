import argparse
import base64
import os

import cv2
import logging

import jinja2

from tracker.mot_tracker import OnlineTracker

from utils import visualization as vis

from utils.log import logger
from utils.timer import Timer

import numpy as np
from ml_serving.drivers import driver

try:
    from mlboardclient.api import client
except ImportError:
    client = None


template = """
<!DOCTYPE html>
<html>
<head>
<style>
table, th, td {
  border: 1px solid black;
  border-collapse: collapse;
}
th, td {
  padding: 7px;
}
table tr:nth-child(even) {
  background-color: #eee;
}
table tr:nth-child(odd) {
 background-color: #fff;
}
body {
font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Noto Sans", "Ubuntu", "Droid Sans", "Helvetica Neue", sans-serif;
}
</style>
</head>
<body>
<h1 style="text-align:center">Persons</h1>
<table style="width:100%">
  <tr>
    <th>##</th>
    <th>Tracker index</th> 
    <th>Images</th> 
    <th>Duration, sec</th>
    <th>Intervals, sec</th>
  </tr>
  {% for person in data %}
  <tr>
      <td>{{ loop.index }}</td>
      <td>{{ person['index'] }}</td>
      <td>
      {% for img in person['images'] %}
        <img src="data:image/jpeg;base64,{{ img }}"/>
      {% endfor %}
      </td>
      <td>{{ person['duration_sec'] }}</td>
      <td>{{ person['intervals_str'] }}</td>
  </tr>
  {% endfor %}
</table>
</body>
</html>
"""


def update_data(data, use_mlboard, mlboard):
    if use_mlboard and mlboard:
        mlboard.update_task_info(data)


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

    write_report_to = None
    data = {}
    if kwargs['report_output']:
        write_report_to = kwargs['report_output']

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

            # read each X bgr frame
            frame = cap.read()  # bgr
            if frame_count % each_frame > 0:
                continue

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

            if write_report_to:

                for i, id in enumerate(online_ids):
                    if id not in data:
                        data[id] = {
                            'intervals': [],
                            'images': [],
                            'last_image': None,
                        }
                    di = data[id]['intervals']
                    if len(di) == 0 or di[-1][1] < frame_count - each_frame:
                        if len(di) > 0 and di[-1][0] == di[-1][1]:
                            di = di[:-1]
                        di.append([frame_count, frame_count])
                    else:
                        di[-1][1] = frame_count
                    if not data[id]['last_image'] or data[id]['last_image'] < frame_count - fps * 10:
                        data[id]['last_image'] = frame_count
                        tlwh = [max(0, int(o)) for o in online_tlwhs[i]]
                        pers_img = frame[tlwh[1]:tlwh[1]+tlwh[3], tlwh[0]:tlwh[0]+tlwh[2]].copy()
                        if max(pers_img.shape[0], pers_img.shape[1]) > 100:
                            coef = max(pers_img.shape[0], pers_img.shape[1]) / 100
                            pers_img = cv2.resize(pers_img,
                                                  (int(pers_img.shape[1] / coef), int(pers_img.shape[0] / coef)))
                        _, pers_img = cv2.imencode('.jpeg', pers_img)
                        data[id]['images'].append(base64.b64encode(pers_img).decode())

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

        if write_report_to:

            for i in data:
                di = data[i]
                di['index'] = i
                di['duration'] = sum([i[1] - i[0] for i in di['intervals']])
                di['duration_sec'] = '{:.2f}'.format(di['duration'] / fps)
                di['intervals_str'] = ', '.join(
                    ['{:.2f}-{:.2f}'.format(i[0] / fps, i[1] / fps) for i in di['intervals']])

            data = data.values()
            data = sorted(data, key=lambda x: x['duration'], reverse=True)

            # prepare html
            tpl = jinja2.Template(template)

            html = tpl.render(data=data)
            with open(write_report_to, 'w') as f:
                f.write(html)

            update_data({'#documents.persons.html': html}, use_mlboard, mlboard)


if __name__ == '__main__':

    use_mlboard = False
    mlboard = None
    if client:
        mlboard = client.Client()
        try:
            mlboard.apps.get()
        except Exception:
            mlboard = None
            logger.info('Do not use mlboard.')
        else:
            logger.info('Use mlboard parameters logging.')
            use_mlboard = True

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
        '--report_output',
        type=str,
        default=None,
        help='Save html report to file',
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


import argparse
import os
# from datetime import datetime

import cv2
import logging
import motmetrics as mm
from tracker.mot_tracker import OnlineTracker

# from datasets.mot_seq import get_loader
from utils import visualization as vis
# from PIL import Image

from utils.log import logger
from utils.timer import Timer
from utils.evaluation import Evaluator

import numpy as np
from ml_serving.drivers import driver


def mkdirs(path):
    if os.path.exists(path):
        return
    os.makedirs(path)


def write_results(filename, results, data_type):
    if data_type == 'mot' or data_type == 'video':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def detect_persons_tf(drv, frame, threshold = .5):
    input_name, input_shape = list(drv.inputs.items())[0]
    # output_name = list(drv.outputs)[0]
    inference_frame = frame[:, :, ::-1]
    inference_frame  = np.expand_dims(frame, axis=0)
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
    # classes = classes[human_classes]

    # masks = None
    # if "detection_masks" in outputs:
    #     masks = outputs["detection_masks"].copy()
    #     masks = masks.reshape([-1, masks.shape[2], masks.shape[3]])

    # scores = scores[np.where(scores > threshold)]
    # boxes = boxes[:len(scores)]
    # classes = classes[:len(scores)]
    # if masks is not None:
    #     masks = masks[:len(scores)]

    boxes[:, 0] *= frame.shape[0]
    boxes[:, 1] *= frame.shape[1]
    boxes[:, 2] *= frame.shape[0]
    boxes[:, 3] *= frame.shape[1]
    boxes[:,[0, 1, 2, 3]] = boxes[:,[1, 0, 3, 2]].astype(int)
    return boxes, scores  # , classes  # , masks


def eval_video(video_file,
                each_frame=5,
                person_detect_model='./data/faster_rcnn_resnet101_coco_2018_01_28/saved_model',
                squeezenet_ckpt=None,
                googlenet_ckpt=None,
                frames_limit=None,
                save_dir=None,
                show_image=True
               ):
    logger.setLevel(logging.INFO)

    cap = cv2.VideoCapture(video_file)
    frame_count = -1
    iter_count = 0
    
    tracker = OnlineTracker(squeezenet_ckpt=squeezenet_ckpt, googlenet_ckpt=googlenet_ckpt)
    timer = Timer()
    results = []
    wait_time = 1

    drv = driver.load_driver('tensorflow')

    logger.info(f'init person detection driver...')
    person_detect_driver = drv()
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

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # logger.info(f'frame {frame_count} read')

            det_tlwhs, det_scores = detect_persons_tf(person_detect_driver, frame, threshold=.5)
            # logger.info(f'frame {frame_count}: detected {len(det_tlwhs)} boxes')

            # run tracking
            timer.tic()
            online_targets = tracker.update(frame, det_tlwhs, None)
            # logger.info(f'frame {frame_count}: {len(online_targets)} online targets')
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                online_tlwhs.append(t.tlwh)
                online_ids.append(t.track_id)
            timer.toc()

            # logger.info(f'tracker done')

            # save results
            frame_id = frame_count  # or make it incremental?
            results.append((frame_id + 1, online_tlwhs, online_ids))

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            online_im = vis.plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)

            # logger.info(f'draw {len(det_tlwhs)} bboxes')
            for bbox in det_tlwhs:
                cv2.rectangle(
                    online_im,
                    (bbox[0], bbox[1]),  # (left, top)
                    (bbox[2], bbox[3]),  # (right, bottom)
                    (0, 255, 0),
                    1,
                )

            # logger.info(f'tracking plotted')
            if show_image:
                # logger.info(f'show image')
                cv2.imshow('online_im', online_im)
            if save_dir is not None:
                save_to = os.path.join(save_dir, '{:05d}.jpg'.format(frame_id))
                # logger.info(f'save data to {save_to}')
                cv2.imwrite(save_to, online_im)

            # logger.info(f'wait key {wait_time}')
            key = cv2.waitKey(wait_time)
            key = chr(key % 128).lower()
            if key in [ord('q'), 202, 27]:  # 'q' or Esc or 'q' in russian layout
                exit(0)
            elif key == 'p':
                cv2.waitKey(0)
            elif key == 'a':
                wait_time = int(not wait_time)

            iter_count += 1
            # logger.info(f'done')

    except (KeyboardInterrupt, SystemExit) as e:
        logger.info('Caught %s: %s' % (e.__class__.__name__, e))
    finally:
        cv2.destroyAllWindows()


    # save results
    data_root = os.path.dirname(video_file)
    seq = os.path.splitext(os.path.basename(video_file))[0]
    exp_name = 'demo'
    data_type = 'video'
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    result_filename = os.path.join(result_root, '{}.txt'.format(seq))
    write_results(result_filename, results, data_type)

    # eval
    accs = []
    logger.info('Evaluate seq: {}'.format(seq))
    evaluator = Evaluator(data_root, seq, data_type)
    accs.append(evaluator.eval_file(result_filename))

    # get summary
    # metrics = ['mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall']
    metrics = mm.metrics.motchallenge_metrics
    # metrics = None
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, [seq], metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, f'summary_{exp_name}.xlsx'))


    # result_root = os.path.join(data_root, '..', 'results', exp_name)
    # mkdirs(result_root)
    # data_type = 'mot'
    #
    # # run tracking
    # accs = []
    # for seq in seqs:
    #     output_dir = os.path.join(data_root, 'outputs', seq) if save_image else None
    #
    #     logger.info('start seq: {}'.format(seq))
    #     loader = get_loader(data_root, det_root, seq)
    #     result_filename = os.path.join(result_root, '{}.txt'.format(seq))
    #     eval_seq(loader, data_type, result_filename,
    #              save_dir=output_dir, show_image=show_image)
    #
    #     # eval
    #     logger.info('Evaluate seq: {}'.format(seq))
    #     evaluator = Evaluator(data_root, seq, data_type)
    #     accs.append(evaluator.eval_file(result_filename))
    #
    # # get summary
    # # metrics = ['mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall']
    # metrics = mm.metrics.motchallenge_metrics
    # # metrics = None
    # mh = mm.metrics.create()
    # summary = Evaluator.get_summary(accs, seqs, metrics)
    # strsummary = mm.io.render_summary(
    #     summary,
    #     formatters=mh.formatters,
    #     namemap=mm.io.motchallenge_metric_names
    # )
    # print(strsummary)
    # Evaluator.save_summary(summary, os.path.join(result_root, f'summary_{exp_name}.xlsx'))

    # # eval
    # try:
    #     import matlab.engine as matlab_engine
    #     eval_root = '/data/MOT17/amilan-motchallenge-devkit'
    #     seqmap = 'eval_mot_generated.txt'
    #     with open(os.path.join(eval_root, 'seqmaps', seqmap), 'w') as f:
    #         f.write('name\n')
    #         for seq in seqs:
    #             f.write('{}\n'.format(seq))
    #
    #     logger.info('start eval {} in matlab...'.format(result_root))
    #     eng = matlab_engine.start_matlab()
    #     eng.cd(eval_root)
    #     eng.run_eval(data_root, result_root, seqmap, '', nargout=0)
    # except ImportError:
    #     logger.warning('import matlab.engine failed...')



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
        default='./data/faster_rcnn_resnet101_coco_2018_01_28/saved_model',
        help='Person detection model',
    )
    parser.add_argument(
        '--squeezenet_ckpt',
        type=str,
        default=None,
        help='Squeezenet path',
    )
    parser.add_argument(
        '--googlenet_ckpt',
        type=str,
        default=None,
        help='Googlenet path',
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
    args = parser.parse_args()

    # import fire
    # fire.Fire(main)

    # seqs_str = '''MOT16-02
    #             MOT16-05
    #             MOT16-09
    #             MOT16-11
    #             MOT16-13'''
    # seqs = [seq.strip() for seq in seqs_str.split()]

    eval_video(
        video_file=args.video_source,
        each_frame=args.each_frame,
        person_detect_model=args.person_detect_model,
        squeezenet_ckpt=args.squeezenet_ckpt,
        googlenet_ckpt=args.googlenet_ckpt,
        frames_limit=args.frames_limit,
        save_dir=args.save_dir,
        show_image=False,
    )


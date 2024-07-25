import numpy as np
import tensorflow as tf
import threading
import itertools
import json
import math
from functools import partial
from os.path import join, splitext, exists
from glob import glob
from scipy.misc import imresize, imread
import data_label

# Utility functions
def chunks(lst, size, offset=0):
    for i in range(0, len(lst), size):
        yield lst[i + offset:i + size]

def build_frame_list(path, frame_label):
    jpg_list = []
    if exists(path):
        jpg_list = glob(join(path, "*.jpg"))
        jpg_list = sorted(jpg_list, key=lambda jpg: int(splitext(jpg)[0]))
    frame_list = [{"json_path": splitext(jpg)[0] + ".json", "img_path": jpg, "frame_label": frame_label} for jpg in jpg_list]
    return frame_list

def build_scene_list(path, frames_per_scene, frames_to_remove, frame_label):
    scene_list = list(chunks(build_frame_list(path, frame_label), frames_per_scene, offset=frames_to_remove))
    return scene_list

def get_vehicle_label(bbox, motion_model, ttc_threshold):
    if 'label' in bbox:
        data_driven_label = bbox['label']
        rule_based_label = bbox['label']
        rule_based_prob = None
    else:
        data_driven_label = bbox['syntheticLabel']
        if motion_model == 'ctra':
            rule_based_label = bbox['adaptedLabel']
            rule_based_prob = None  # TBA
        elif motion_model == 'cca':
            rule_based_label = None  # TBA
            rule_based_prob = None  # TBA
        else:
            raise ValueError("Unknown motion model")
    return data_driven_label, rule_based_label, rule_based_prob

def frame_per_vehicle(frame, vehicle_bbox):
    frame_per_vehicle = {
        'img_path': frame['img_path'],
        'json_path': frame['json_path'],
        'bbox': vehicle_bbox,
    }
    return frame_per_vehicle

def get_bbox(frame_info, hashcode):
    for bbox in frame_info['vehicleInfo']:
        if hashcode == bbox['hashcode']:
            return bbox
    return None

def build_sample_list(scenes, frames_per_sample, label_method, motion_model, ttc_threshold):
    n_samples = 0
    n_acc_samples = 0
    n_nonacc_samples = 0
    samples_grouped_by_frame = []

    for scene in scenes:
        for f in range(len(scene)):
            samples_per_frame = []
            if f < frames_per_sample - 1:
                continue

            frame_t_minus_info = [json.load(open(scene[f - i]['json_path'])) for i in range(frames_per_sample)]
            frame_label = scene[f]['frame_label']

            for bbox in frame_t_minus_info[0]['vehicleInfo']:
                data_driven_label, rule_based_label, rule_based_prob = get_vehicle_label(bbox, motion_model, ttc_threshold)
                if data_driven_label is not None and rule_based_label is not None:
                    sample = {
                        'hashcode': bbox['hashcode'],
                        'frame_label': frame_label,
                        **{f'frame_t-{i/10.0}s': frame_per_vehicle(scene[f - i], get_bbox(frame_t_minus_info[i], bbox['hashcode']))
                           if frames_per_sample > i else None for i in range(frames_per_sample)}
                    }
                    n_samples += 1
                    if label_method == "data_driven":
                        sample['label'] = data_driven_label
                        sample['label_prob'] = data_driven_label
                        if data_driven_label:
                            n_acc_samples += 1
                        else:
                            n_nonacc_samples += 1
                    elif label_method == "rule_based":
                        sample['label'] = rule_based_label
                        sample['label_prob'] = rule_based_label
                        if rule_based_label:
                            n_acc_samples += 1
                        else:
                            n_nonacc_samples += 1
                    elif label_method == "rule_based_prob":
                        sample['label'] = rule_based_label
                        sample['label_prob'] = rule_based_prob
                        if rule_based_label:
                            n_acc_samples += 1
                        else:
                            n_nonacc_samples += 1
                    samples_per_frame.append(sample)
            samples_grouped_by_frame.append(samples_per_frame)
    return samples_grouped_by_frame, n_acc_samples, n_nonacc_samples, n_samples

class Dataset:
    def initialize(self, opt, sess):
        self.opt = opt
        self.sess = sess
        self.train_root = opt.train_root
        self.valid_root = opt.valid_root
        self.test_root = opt.test_root
        self.n_test_splits = opt.n_test_splits
        self.train_frames_per_scene = opt.train_frames_per_scene
        self.valid_frames_per_scene = opt.valid_frames_per_scene
        self.test_frames_per_scene = opt.test_frames_per_scene
        self.train_frames_to_remove = opt.train_frames_to_remove
        self.valid_frames_to_remove = opt.valid_frames_to_remove
        self.test_frames_to_remove = opt.test_frames_to_remove
        self.frames_per_sample = opt.frames_per_sample
        self.no_shuffle_per_epoch = opt.no_shuffle_per_epoch
        self.label_method = opt.label_method
        self.motion_model = opt.motion_model
        self.ttc_threshold = opt.ttc_threshold
        # self.pool = ThreadPool(16)  # Number of threads

        self.train_accident_scenes = build_scene_list(
            join(self.train_root, "accident/"), self.train_frames_per_scene,
            self.train_frames_to_remove, data_label.ACCIDENT
        )
        self.train_nonaccident_scenes = build_scene_list(
            join(self.train_root, "nonaccident/"), self.train_frames_per_scene,
            self.train_frames_to_remove, data_label.NONACCIDENT
        )
        self.train_scenes = self.train_accident_scenes + self.train_nonaccident_scenes

        self.valid_accident_scenes = build_scene_list(
            join(self.valid_root, "accident/"), self.valid_frames_per_scene,
            self.valid_frames_to_remove, data_label.ACCIDENT
        )
        self.valid_nonaccident_scenes = build_scene_list(
            join(self.valid_root, "nonaccident/"), self.valid_frames_per_scene,
            self.valid_frames_to_remove, data_label.NONACCIDENT
        )
        self.valid_scenes = self.valid_accident_scenes + self.valid_nonaccident_scenes

        dataset_random = random.Random()
        dataset_random.seed(2018)
        dataset_random.shuffle(self.train_scenes)

        self.train_scenes = self.train_scenes[:int(self.opt.train_dataset_proportion * len(self.train_scenes))]

        self.train_samples_grouped_by_frame, self.n_train_acc_samples, self.n_train_nonacc_samples, self.train_data_size = \
            build_sample_list(self.train_scenes, self.frames_per_sample, self.label_method, self.motion_model, self.ttc_threshold)

        self.valid_samples_grouped_by_frame, self.n_valid_acc_samples, self.n_valid_nonacc_samples, self.valid_data_size = \
            build_sample_list(self.valid_scenes, self.frames_per_sample, self.label_method, self.motion_model, self.ttc_threshold)

        if self.opt.isTrain and self.n_train_acc_samples != 0:
            self.pos_ratio = self.n_train_nonacc_samples / self.n_train_acc_samples
        else:
            self.pos_ratio = 1

        self.train_steps_per_epoch = int(math.ceil(self.train_data_size / opt.batchSize))
        self.valid_steps_per_epoch = int(math.ceil(self.valid_data_size / opt.batchSize))

        self.test_datasets = []
        if self.n_test_splits > 0:
            test_roots = [join(self.test_root, str(split)) for split in range(self.n_test_splits)]
        else:
            test_roots = [self.test_root]
        for dataroot in test_roots:
            if dataroot != '':
                test_accident_scenes = build_scene_list(join(dataroot, "accident/"), self.test_frames_per_scene, self.test_frames_to_remove, data_label.ACCIDENT)
                test_nonaccident_scenes = build_scene_list(join(dataroot, "nonaccident/"), self.test_frames_per_scene, self.test_frames_to_remove, data_label.NONACCIDENT)
                test_scenes = test_accident_scenes + test_nonaccident_scenes
                test_samples_grouped_by_frame, n_test_acc_samples, n_test_nonacc_samples, test_data_size = build_sample_list(
                    test_scenes, self.frames_per_sample, self.label_method, self.motion_model, self.ttc_threshold)
                self.test_datasets.append({
                    "dataroot": dataroot,
                    "n_test_acc_samples": n_test_acc_samples,
                    "n_test_nonacc_samples": n_test_nonacc_samples,
                    "test_samples_grouped_by_frame": test_samples_grouped_by_frame,
                    "test_data_size": test_data_size,
                    "test_steps_per_epoch": int(math.ceil(test_data_size / opt.batchSize)),
                })

        self.train_queue = tf.FIFOQueue(512, [tf.float32, tf.int32, tf.float32, tf.int32], shapes=[[130, 355, opt.input_channel_dim], [], [], []])
        self
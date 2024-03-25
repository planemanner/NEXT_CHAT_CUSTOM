from typing import List, Dict
import json
import os
import numpy as np


def normalize_confidence(confidences):
    min_confidence = np.min(confidences)
    max_confidence = np.max(confidences)
    normalized_confidences = (confidences - min_confidence) / (max_confidence - min_confidence)
    return normalized_confidences


def compute_precision_recall(sorted_tp_fp):
    tps = np.cumsum(sorted_tp_fp)
    fps = np.cumsum(~sorted_tp_fp)
    
    recall = tps / (tps[-1] + fps[-1])
    precision = tps / (tps + fps)
    
    return precision, recall

def compute_interpolated_precision(recall, precision):
    recall_values = np.linspace(0, 1, 11)[::-1]
    interpolated_precision = []
    
    for r in recall_values:
        mask = recall >= r
        if mask.any():
            interpolated_precision.append(np.max(precision[mask]))
        else:
            interpolated_precision.append(0)
    
    return np.array(interpolated_precision)

def compute_average_precision(interpolated_precision):
    return np.mean(interpolated_precision)

def compute_ap(results):
    sorted_indices = np.argsort(results['confidences'])[::-1]
    sorted_tp_fp = np.array([results['tp_or_fp'][idx] for idx in sorted_indices], dtype=bool)

    precision, recall = compute_precision_recall(sorted_tp_fp)
    interpolated_precision = compute_interpolated_precision(recall, precision)
    ap = compute_average_precision(interpolated_precision)
    
    return ap


def get_iou(box1, box2):
    # 각 박스의 좌표 추출
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # 교차하는 부분의 좌표 계산
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)

    # 교차하는 부분의 넓이 계산
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)

    # 각 박스의 넓이 계산
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)

    # 합집합의 넓이 계산
    union_area = box1_area + box2_area - intersection_area

    # IoU 계산
    iou = intersection_area / union_area

    return iou


def convert2json(result: List[Dict], save_dir: str):
    save_path = os.path.join(save_dir, "caption_result.json")
    with open(save_path, 'w') as f:
        json.dump(result, f)

    return save_path
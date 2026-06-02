import cv2
import numpy as np

from vlm.coco_classes import COCO_CLASSES
from vlm.config import (
    VLM_GROUNDING_DINO_HTTP_MODE,
    VLM_GROUNDING_DINO_PORT,
    VLM_MOBILE_SAM_HTTP_MODE,
    VLM_MOBILE_SAM_PORT,
    VLM_YOLOV7_HTTP_MODE,
    VLM_YOLOV7_PORT,
)
from vlm.detector.grounding_dino import GroundingDINOInterface
from vlm.detector.yolov7 import YOLOv7Interface
from vlm.segmentor.sam import MobileSAMInterface

yolov7_detector = YOLOv7Interface(http_mode=VLM_YOLOV7_HTTP_MODE, port=VLM_YOLOV7_PORT)
sam_segmentor = MobileSAMInterface(
    http_mode=VLM_MOBILE_SAM_HTTP_MODE, port=VLM_MOBILE_SAM_PORT
)
dino_detector = GroundingDINOInterface(
    http_mode=VLM_GROUNDING_DINO_HTTP_MODE, port=VLM_GROUNDING_DINO_PORT
)


def get_segmentation(segmented_img_rgb, idx, detections, img, label, score, bgr_color):
    object_mask = np.zeros((480, 640), dtype=np.uint8)
    bbox_denorm = detections.boxes[idx] * np.array(
        [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
    )
    x1, y1, x2, y2 = [int(v) for v in bbox_denorm]
    bbox_area = (x2 - x1) * (y2 - y1)
    img_area = img.shape[0] * img.shape[1]
    segmented_img_bgr = cv2.cvtColor(segmented_img_rgb, cv2.COLOR_RGB2BGR)

    if bbox_area / img_area < 0.99:
        object_mask = sam_segmentor.segment_bbox(img, bbox_denorm.tolist())
        contours, _ = cv2.findContours(
            object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            cv2.drawContours(segmented_img_bgr, [contour], 0, bgr_color, 4)

        cv2.rectangle(
            segmented_img_bgr,
            (x1, y1),
            (x2, y2),
            bgr_color,
            2,
        )

        label_text = f"{label} ({score:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2
        )
        label_x = x1
        label_y = y1 - text_height
        cv2.rectangle(
            segmented_img_bgr,
            (label_x, label_y - 30),
            (label_x + text_width, label_y + text_height),
            bgr_color,
            2,
        )
        cv2.putText(
            segmented_img_bgr,
            label_text,
            (label_x, label_y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 255),
            1,
        )
    segmented_img_rgb = cv2.cvtColor(segmented_img_bgr, cv2.COLOR_BGR2RGB)
    return segmented_img_rgb, object_mask


def crop_and_expand_box(img, detections, idx, expand_pixels=0.4):
    # Get box coordinates in format [x_min, y_min, x_max, y_max]
    x_min, y_min, x_max, y_max = detections.boxes[idx]
    x_min = int(x_min * img.shape[1])
    y_min = int(y_min * img.shape[0])
    x_max = int(x_max * img.shape[1])
    y_max = int(y_max * img.shape[0])

    # Expand box outward, ensure not to exceed image boundaries
    x_min = max(int(x_min * (1 - expand_pixels)), 0)
    y_min = max(int(y_min * (1 - expand_pixels)), 0)
    x_max = min(int(x_max * (1 + expand_pixels)), img.shape[1] - 1)
    y_max = min(int(y_max * (1 + expand_pixels)), img.shape[0] - 1)

    # Crop image, keeping only content within the box
    img_detected = img[y_min : y_max + 1, x_min : x_max + 1]

    return img_detected


def detect_objects(object_classes, img, cfg):
    score_list = []
    object_masks_list = []
    class_indices = []
    rgb_patchs = []
    segmented_img = img.copy()

    # Separate classes into COCO standard classes and custom classes
    coco_classes = []
    custom_classes = []
    for cls in object_classes:
        if cls in COCO_CLASSES:
            coco_classes.append(cls)
        else:
            custom_classes.append(cls)

    # Detect COCO standard classes
    if coco_classes:
        detections = yolov7_detector.predict(
            img,
            agnostic_nms=cfg.yolo.agnostic_nms,
            conf_thres=cfg.yolo.confidence_threshold_yolo,
            iou_thres=cfg.yolo.iou_threshold_yolo,
        )

        for idx in range(len(detections.logits)):
            class_name = detections.phrases[idx]
            score = detections.logits[idx].item()

            # Keep only detection results in target class list
            if class_name in coco_classes:
                segmented_img, object_mask = get_segmentation(
                    segmented_img,
                    idx,
                    detections,
                    img,
                    class_name,
                    score,
                    bgr_color=(0, 255, 0),  # COCO class is green
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                class_indices.append(object_classes.index(class_name))
                rgb_patchs.append(crop_and_expand_box(img, detections, idx))

    # Detect custom classes
    if custom_classes:
        # Build text prompts required by GroundingDINO
        caption = " . ".join(custom_classes)
        detections = dino_detector.predict(
            img,
            caption=caption,
            box_threshold=cfg.groundingDINO.confidence_threshold_dino,
            text_threshold=cfg.groundingDINO.text_threshold,
        )
        detections.filter_by_class(custom_classes)

        for idx in range(len(detections.logits)):
            class_name = detections.phrases[idx]
            score = detections.logits[idx].item()

            # Keep only detection results in target class list
            if class_name in custom_classes:
                segmented_img, object_mask = get_segmentation(
                    segmented_img,
                    idx,
                    detections,
                    img,
                    class_name,
                    score,
                    bgr_color=(255, 0, 0),  # Custom class is blue
                )
                score_list.append(score)
                object_masks_list.append(object_mask)
                class_indices.append(object_classes.index(class_name))
                rgb_patchs.append(crop_and_expand_box(img, detections, idx))

    return segmented_img, score_list, object_masks_list, class_indices, rgb_patchs

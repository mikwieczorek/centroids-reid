def transform_bbox_s2s_to_coco(bbox):
    """Function that rearranges bbox annotations from Street2Shop format to COCO"""
    return [bbox["left"], bbox["top"], bbox["width"], bbox["height"]]


def bbox_coco_to_corners(bbox):
    # Input:
    # [top_left_x, top_left_y, width, height]
    # Output:
    # [top_left_x, top_left_y, bottom_left_x, bottom_left_y]
    bbox[0] = bbox[0]
    bbox[1] = bbox[1]
    bbox[2] = bbox[0] + bbox[2]
    bbox[3] = bbox[1] + bbox[3]

    return bbox


def bbox_coco_to_center(bbox):
    # Input:
    # [top_left_x, top_left_y, width, height]
    # Output:
    # [x_center_of_bbox, y_center_of_bbox, width, height]
    bbox[0] = bbox[0] + bbox[2] / 2
    bbox[1] = bbox[1] + bbox[3] / 2
    bbox[2] = bbox[2]
    bbox[3] = bbox[3]

    return bbox


def bbox_center_to_yolo(bbox, width, height):
    # Input:
    # [x_center_of_bbox, y_center_of_bbox, width, height]
    # Output:
    # [x_center_of_bbox, y_center_of_bbox, width, height] / {image_width || image_height}
    bbox[0] = bbox[0] / width
    bbox[1] = bbox[1] / height
    bbox[2] = bbox[2] / width
    bbox[3] = bbox[3] / height

    return bbox


def bbox_yolo_to_center(bbox, width, height):
    # Input:
    # [x_center_of_bbox, y_center_of_bbox, width, height] / {image_width || image_height}
    # Output:
    # [x_center_of_bbox, y_center_of_bbox, width, height]
    bbox[0] = bbox[0] * width
    bbox[1] = bbox[1] * height
    bbox[2] = bbox[2] * width
    bbox[3] = bbox[3] * height

    return bbox


def bbox_center_to_coco(bbox):
    # Input:
    # [x_center_of_bbox, y_center_of_bbox, width, height]
    # Output:
    # [top_left_x, top_left_y, width, height]
    bbox[0] = bbox[0] - bbox[2] / 2
    bbox[1] = bbox[1] - bbox[3] / 2

    return bbox


def bbox_coco_to_yolo(bbox, width, height):
    # Input:
    # [top_left_x, top_left_y, width, height]
    # Output:
    # [x_center_of_bbox, y_center_of_bbox, width, height] / {image_width || image_height}
    bbox = bbox_coco_to_center(bbox)
    bbox = bbox_center_to_yolo(bbox, width, height)

    return bbox


def bbox_yolo_to_coco(bbox, width, height):
    # Input:
    # [x_center_of_bbox, y_center_of_bbox, width, height] / {image_width || image_height}
    # Output:
    # [top_left_x, top_left_y, width, height]
    bbox = [float(item) for item in bbox]
    bbox = bbox_yolo_to_center(bbox, width, height)
    bbox = bbox_center_to_coco(bbox)
    bbox = [int(item) for item in bbox]

    return bbox

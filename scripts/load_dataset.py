import json
from paco.data.datasets.builtin import _PREDEFINED_PACO
import os
from collections import defaultdict

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import numpy as np
from PIL import Image

dataset_name = "paco_lvis_v1_test"

dummy_meta = MetadataCatalog.get("sem2").set(
    stuff_classes=['' for _i in range(76)],
    stuff_colors=[(0, 0, 255)] * 75 + [(255, 255, 255)]
)


def get_obj_and_part_anns(annotations):
    """
    Returns a map between an object annotation ID and
    (object annotation, list of part annotations) pair.
    """
    obj_ann_id_to_anns = {ann["id"]: (ann, []) for ann in annotations if ann["id"] == ann["obj_ann_id"]}
    for ann in annotations:
        if ann["id"] != ann["obj_ann_id"]:
            obj_ann_id_to_anns[ann["obj_ann_id"]][1].append(ann)
    return obj_ann_id_to_anns

def get_image_with_boxes(im_fn, anns, cat_id_to_name):
    """
    Reads the image, overlays boxes, and returns a numpy array with an image in RGB format.
    """
    # Load image.
    im = np.asarray(Image.open(im_fn))

    # Extract boxes (in XYXY format) and labels.
    boxes = []
    labels = []
    for ann, _ in anns:
        boxes.append(ann["bbox"])
        labels.append(cat_id_to_name[ann["category_id"]].split("_(")[0])
    boxes = np.array(boxes)
    boxes[:, 2:] += boxes[:, :2]

    # Use LVIS color list (https://github.com/lvis-dataset/lvis-api/blob/master/lvis/colormap.py).
    color_list = np.array(
        [0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494, 0.184, 0.556, 0.466, 0.674, 0.188, 0.301,
         0.745, 0.933, 0.635, 0.078, 0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000, 1.000, 0.500,
         0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000,
         0.333, 0.667, 0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000, 0.667, 1.000, 0.000, 1.000,
         0.333, 0.000, 1.000, 0.667, 0.000, 1.000, 1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
         0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500, 0.333, 1.000, 0.500, 0.667, 0.000, 0.500,
         0.667, 0.333, 0.500, 0.667, 0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333, 0.500, 1.000,
         0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000, 0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000,
         1.000, 0.333, 0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000, 1.000, 0.667, 0.333, 1.000,
         0.667, 0.667, 1.000, 0.667, 1.000, 1.000, 1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.167,
         0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000,
         0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000,
         0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000,
         0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286, 0.286, 0.286, 0.429, 0.429,
         0.429, 0.571, 0.571, 0.571, 0.714, 0.714, 0.714, 0.857, 0.857, 0.857, 1.000, 1.000, 1.000])
    color_list = color_list.astype(np.float32).reshape((-1, 3))
    box_colors = [color_list[idx % len(color_list)] for idx in range(len(boxes))]

    # Overlay boxes.
    viz = Visualizer(im, dummy_meta)
    im_w_boxes = viz.overlay_instances(
        labels=labels,
        boxes=boxes,
        masks=None,
        keypoints=None,
        assigned_colors=box_colors,
    ).get_image()

    return im_w_boxes



# Derived parameters.
dataset_file_name, image_root_dir = _PREDEFINED_PACO[dataset_name]
print("Dataset file name: {}".format(dataset_file_name))
print("Image root dir: {}".format(image_root_dir))

# Load dataset.
with open(dataset_file_name) as f:
    dataset = json.load(f)

# Extract maps from dataset.
cat_id_to_name = {d["id"]: d["name"] for d in dataset["categories"]}
attr_id_to_name = {d["id"]: d["name"] for d in dataset["attributes"]}

image_id_to_image_file_name = {d["id"]: os.path.join(image_root_dir, d["file_name"]) for d in dataset["images"]}

obj_ann_id_to_anns = get_obj_and_part_anns(dataset["annotations"])

cat_name_to_anns = defaultdict(list)
attr_name_to_anns = defaultdict(list)

cat_name_to_attr_name_to_anns = defaultdict(lambda: defaultdict(list))

for ann in dataset["annotations"]:
    anns = obj_ann_id_to_anns[ann["obj_ann_id"]]
    im_fn = image_id_to_image_file_name[ann['image_id']]
    im1 = get_image_with_boxes(im_fn, ann, cat_id_to_name)


    attr_ids = ann["attribute_ids"]
    cat_name = cat_id_to_name[ann["category_id"]]
    if len(attr_ids) > 0:
        cat_name_to_anns[cat_name].append(anns)
    attr_list = []
    for attr_id in attr_ids:
        attr_name = attr_id_to_name[attr_id]
        attr_list.append(attr_id)
        attr_name_to_anns[attr_name].append(anns)
        cat_name_to_attr_name_to_anns[cat_name][attr_name].append(anns)
    print("Attributes: {}".format(' '.join(attr_list)))
    im1.show()

cat_name_to_anns = dict(cat_name_to_anns)
attr_name_to_anns = dict(attr_name_to_anns)
cat_name_to_attr_name_to_anns = {k: dict(v) for k, v in cat_name_to_attr_name_to_anns.items()}

print("Available categories:", sorted(cat_name_to_anns.keys()))
print("Available attributes:", sorted(attr_name_to_anns.keys()))

print("Total Annotations: {}".format(len(dataset["annotations"])))


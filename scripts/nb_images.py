import json
from paco.data.datasets.builtin import _PREDEFINED_PACO
import os
from collections import defaultdict

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import numpy as np
from PIL import Image


def get_obj_and_part_anns(annotations):
    """
    Returns a map between an object annotation ID and
    (object annotation, list of part annotations) pair.
    """
    ann_id_to_anns = {ann["id"]: (ann, []) for ann in annotations if ann["id"] == ann["obj_ann_id"]}
    for ann in annotations:
        if ann["id"] != ann["obj_ann_id"]:
            ann_id_to_anns[ann["obj_ann_id"]][1].append(ann)
    return ann_id_to_anns


dummy_meta = MetadataCatalog.get("sem2").set(
    stuff_classes=['' for _i in range(76)],
    stuff_colors=[(0, 0, 255)] * 75 + [(255, 255, 255)]
)


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


def get_image_with_masks(im_fn, anns, cat_id_to_name, mask_type="part"):
    """
    Reads the image, overlays masks, and returns a numpy array with an image in RGB format.
    """
    # Load image.
    im = np.asarray(Image.open(im_fn))

    # Build overlay masks and labels.
    masks = []
    labels = []
    for ann, part_anns in anns:
        if mask_type == "part":
            for part_ann in part_anns:
                if part_ann["segmentation"] != []:
                    masks.append(part_ann["segmentation"])
                    labels.append(cat_id_to_name[part_ann["category_id"]].split(":")[-1])
        else:
            if ann["segmentation"] != []:
                masks.append(ann["segmentation"])
                labels.append(cat_id_to_name[ann["category_id"]].split("_(")[0])

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
    mask_colors = [color_list[idx % len(color_list)] for idx in range(len(masks))]

    # Overlay masks.
    viz = Visualizer(im, dummy_meta)
    im_w_masks = viz.overlay_instances(
        labels=labels,
        boxes=None,
        masks=masks,
        keypoints=None,
        assigned_colors=mask_colors,
    ).get_image()

    return im_w_masks


dataset_name = "paco_lvis_v1_test"

# Derived parameters.
dataset_file_name, image_root_dir = _PREDEFINED_PACO[dataset_name]
print("Dataset file name: {}".format(dataset_file_name))
print("Image root dir: {}".format(image_root_dir))

# Load dataset.
with open(dataset_file_name) as f:
    dataset = json.load(f)

# Extract maps from dataset (for filtering and display).
cat_id_to_name = {d["id"]: d["name"] for d in dataset["categories"]}
attr_id_to_name = {d["id"]: d["name"] for d in dataset["attributes"]}
image_id_to_image_file_name = {d["id"]: os.path.join(image_root_dir, d["file_name"]) for d in dataset["images"]}
obj_ann_id_to_anns = get_obj_and_part_anns(dataset["annotations"])

im_id_to_anns = defaultdict(list)
im_id_to_cats = defaultdict(set)
for ann, part_anns in obj_ann_id_to_anns.values():
    im_id_to_anns[ann["image_id"]].append((ann, part_anns))
    im_id_to_cats[ann["image_id"]].add(ann["category_id"])

im_id_to_im_area = {d["id"]: d["height"] * d["width"] for d in dataset["images"]}
im_id_to_mean_box_area = {}
for im_id, anns in im_id_to_anns.items():
    im_area = im_id_to_im_area[im_id]
    box_areas = {ann["area"] for ann, _ in anns}
    im_id_to_mean_box_area[im_id] = sum(box_areas) / len(box_areas) / im_area

vis_offset = 0  # Offset into the list of images to display
vis_num_im = 10  # Number of images to display
vis_im_ids = None  # A user specified list of image IDs to display, set to None to disable
vis_num_cats = None  # Include only images that have the number of categories in this set, set to None to disable
vis_num_boxes = None  # Include only images that have the number of boxes in this set, set to None to disable
vis_num_parts = None  # Include only images that have the number of parts in this set, set to None to disable
vis_mask_type = "obj"  # Mask type, one of "part" or "obj"

if vis_im_ids:
    im_ids = vis_im_ids
else:
    # Start with all images.
    im_ids = im_id_to_anns.keys()
    # Select images satisfying a limit on the number of categories.
    if vis_num_cats is not None:
        im_ids = [im_id for im_id in im_ids if len(im_id_to_cats[im_id]) in vis_num_cats]
    # Further select a subset of images satisfying a limit on the number of boxes.
    if vis_num_boxes is not None:
        im_ids = [im_id for im_id in im_ids if len(im_id_to_anns[im_id]) in vis_num_boxes]
    # Narrow down further by limiting the number of part annotations.
    if vis_num_parts is not None:
        im_ids = [im_id for im_id in im_ids if len(sum(list(zip(*im_id_to_anns[im_id]))[1], start=[])) in vis_num_parts]
    # Sort by box area.
    im_ids = set(im_ids)
    im_ids = [im_id for im_id, mean_box_area in sorted(im_id_to_mean_box_area.items(), key=lambda x: x[1], reverse=True)
              if im_id in im_ids]
print("Number of images to visualize:", len(im_ids))

for im_id in im_ids[vis_offset:vis_offset + vis_num_im]:
    im_fn = image_id_to_image_file_name[im_id]
    anns = im_id_to_anns[im_id]
    im1 = get_image_with_boxes(im_fn, anns, cat_id_to_name)
    im2 = get_image_with_masks(im_fn, anns, cat_id_to_name, vis_mask_type)
    im = Image.fromarray(np.concatenate([im1, im2], axis=1))
    print("Image ID:", im_id)

    im.show()

    #display(im)
    #display(get_text_markdown(anns, cat_id_to_name, attr_id_to_name))

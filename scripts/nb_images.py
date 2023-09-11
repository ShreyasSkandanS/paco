import json
from paco.data.datasets.builtin import _PREDEFINED_PACO
import os
from collections import defaultdict
from PIL import Image
import numpy as np
from paco_utils import get_image_with_boxes, get_image_with_masks, get_obj_and_part_anns

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
vis_mask_type = "part"  # Mask type, one of "part" or "obj"

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

# for im_id in im_ids[vis_offset:vis_offset + vis_num_im]:
for im_id in im_ids[500:520]:
    im_fn = image_id_to_image_file_name[im_id]
    anns = im_id_to_anns[im_id]
    im1 = get_image_with_boxes(im_fn, anns, cat_id_to_name)
    im2 = get_image_with_masks(im_fn, anns, cat_id_to_name, vis_mask_type)
    im = Image.fromarray(np.concatenate([im1, im2], axis=1))
    print("Image ID:", im_id)

    im.show()

    # display(im)
    # display(get_text_markdown(anns, cat_id_to_name, attr_id_to_name))

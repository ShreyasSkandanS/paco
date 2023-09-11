import json
from paco.data.datasets.builtin import _PREDEFINED_PACO
import os
from collections import defaultdict, Counter

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import numpy as np
from PIL import Image

from paco_utils import get_image_with_cat_boxes, get_image_with_masks, get_obj_and_part_anns

from paco.data.datasets.paco import load_json

dataset_name = "paco_lvis_v1_train"

# Derived parameters.
dataset_file_name, image_root_dir = _PREDEFINED_PACO[dataset_name]
print("Dataset file name: {}".format(dataset_file_name))
print("Image root dir: {}".format(image_root_dir))

dset_coco, lvis_api = load_json(dataset_file_name, image_root_dir)

cat_id_ctr = Counter()
for img_anno in dset_coco:
    annos = img_anno['annotations']
    for anno in annos:
        cat_id = anno['category_id']
        cat_id_ctr[cat_id] += 1

category_meta = lvis_api.dataset["categories"]
cat_id_to_name = defaultdict(str)
old_id_to_new = dict()
for idx, cat_id in enumerate(cat_id_ctr.keys()):
    for cat in category_meta:
        if cat['id'] == cat_id:
            cat_name = cat['name']
    cat_id_to_name[cat_id] = cat_name
    old_id_to_new[cat_id] = idx

print("")

# # Load dataset.
# with open(dataset_file_name) as f:
#     dataset = json.load(f)
#
# # Extract maps from dataset (for filtering and display).
# cat_id_to_name = {d["id"]: d["name"] for d in dataset["categories"]}
# attr_id_to_name = {d["id"]: d["name"] for d in dataset["attributes"]}
# image_id_to_image_file_name = {d["id"]: os.path.join(image_root_dir, d["file_name"]) for d in dataset["images"]}
# obj_ann_id_to_anns = get_obj_and_part_anns(dataset["annotations"])
#
# im_id_to_anns = defaultdict(list)
# im_id_to_cats = defaultdict(set)
# for ann, _ in obj_ann_id_to_anns.values():
#     im_id_to_anns[ann["image_id"]].append(ann)
#     im_id_to_cats[ann["image_id"]].add(ann["category_id"])
#
# im_id_to_im_area = {d["id"]: d["height"] * d["width"] for d in dataset["images"]}
# im_id_to_mean_box_area = {}
# for im_id, anns in im_id_to_anns.items():
#     im_area = im_id_to_im_area[im_id]
#     box_areas = {ann["area"] for ann in anns}
#     im_id_to_mean_box_area[im_id] = sum(box_areas) / len(box_areas) / im_area
#
# im_ids = list(im_id_to_anns.keys())
#
# for im_id in im_ids[500:505]:
#     im_fn = image_id_to_image_file_name[im_id]
#     anns = im_id_to_anns[im_id]
#     im1 = get_image_with_cat_boxes(im_fn, anns, cat_id_to_name)
#     print("Image ID:", im_id)
#     im = Image.fromarray(im1)
#     im.show()
#
# print("")
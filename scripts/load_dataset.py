import json
from paco.data.datasets.builtin import _PREDEFINED_PACO
import os
from collections import defaultdict

from paco_utils import get_obj_and_part_anns, get_image_with_boxes, get_image_with_masks

dataset_name = "paco_lvis_v1_test"

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
    attr_ids = ann["attribute_ids"]
    cat_name = cat_id_to_name[ann["category_id"]]
    if len(attr_ids) > 0:
        cat_name_to_anns[cat_name].append(anns)
    for attr_id in attr_ids:
        attr_name = attr_id_to_name[attr_id]
        attr_name_to_anns[attr_name].append(anns)
        cat_name_to_attr_name_to_anns[cat_name][attr_name].append(anns)
cat_name_to_anns = dict(cat_name_to_anns)
attr_name_to_anns = dict(attr_name_to_anns)
cat_name_to_attr_name_to_anns = {k: dict(v) for k, v in cat_name_to_attr_name_to_anns.items()}
print("Available categories:", sorted(cat_name_to_anns.keys()))
print("Available attributes:", sorted(attr_name_to_anns.keys()))
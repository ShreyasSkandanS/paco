import json
from paco.data.datasets.builtin import _PREDEFINED_PACO
from collections import Counter

from paco.data.datasets.paco import load_json


def write_json(json_obj, filename, indent):
    """
    Write JSON object to filename
    """
    with open(filename, 'w') as f:
        json.dump(json_obj, f, indent=indent)


def verify_stats(dset_coco):
    # Verify bounding box and image annotations
    img_counter = Counter()
    for img_anno in dset_coco:
        annos = img_anno['annotations']
        image_id = img_anno['image_id']
        for ann in annos:
            if len(ann['attr_labels']) > 0:
                img_counter[image_id] += 1
    print("Total images with obj & attr: {}".format(len(img_counter)))
    print("Total bounding boxes with obj & attr: {}".format(sum(img_counter.values())))


tr_dataset_name = "paco_lvis_v1_train"
te_dataset_name = "paco_lvis_v1_val"
val_dataset_name = "paco_lvis_v1_test"

# Derived parameters.
tr_dataset_file_name, tr_image_root_dir = _PREDEFINED_PACO[tr_dataset_name]
print("[Train] Dataset file name: {}".format(tr_dataset_file_name))
print("[Train] Image root dir: {}".format(tr_image_root_dir))

te_dataset_file_name, te_image_root_dir = _PREDEFINED_PACO[te_dataset_name]
print("[Test] Dataset file name: {}".format(te_dataset_file_name))
print("[Test] Image root dir: {}".format(te_image_root_dir))

val_dataset_file_name, val_image_root_dir = _PREDEFINED_PACO[val_dataset_name]
print("[Val] Dataset file name: {}".format(val_dataset_file_name))
print("[Val] Image root dir: {}".format(val_image_root_dir))

# Use PACO API to prepare dataset JSON
tr_dset_coco, _ = load_json(tr_dataset_file_name, tr_image_root_dir)
te_dset_coco, _ = load_json(te_dataset_file_name, te_image_root_dir)
val_dset_coco, _ = load_json(val_dataset_file_name, val_image_root_dir)

print("[Training Dataset]:")
verify_stats(tr_dset_coco)

print("[Test Dataset]:")
verify_stats(te_dset_coco)

print("[Val Dataset]:")
verify_stats(val_dset_coco)

new_test_data = te_dset_coco + val_dset_coco

write_json(tr_dset_coco, 'paco_cat_v1_train.json', 4)
write_json(new_test_data, 'paco_cat_v1_test.json', 4)

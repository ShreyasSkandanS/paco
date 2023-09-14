import json
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

train_json_path = 'paco_cat_v1_train.json'
test_json_path = 'paco_cat_v1_test.json'
cat_id_to_name = 'paco_cat_v1_name2id.json'

# Load dataset.
with open(train_json_path) as f:
    train_dataset = json.load(f)

with open(test_json_path) as f:
    test_dataset = json.load(f)

with open(cat_id_to_name) as f:
    cat_id_to_name = json.load(f)

print("Total training samples: {}".format(len(train_dataset)))
print("Total test samples: {}".format(len(test_dataset)))

train_indices = random.sample(range(0, len(train_dataset)), 5)
test_indices = random.sample(range(0, len(train_dataset)), 5)

# indices = test_indices
# dataset = test_dataset
# indices = train_indices
# dataset = train_dataset
indices = range(0, len(train_dataset))
dataset = train_dataset

image_area_list = []
bbox_area_list = []
relative_area_list = []
cat_count = 0
cat_name = 'drill'
cat_bbox_list = []
for idx in indices:
    sample = train_dataset[idx]
    fname = sample['file_name']
    image_area = sample['height'] * sample['width']
    image_area_list.append(image_area)
    for anno in sample['annotations']:
        bbox_xywh = anno['bbox']
        bbox_area = bbox_xywh[2] * bbox_xywh[3]
        bbox_area_list.append(bbox_area)
        relative_area_list.append(bbox_area / image_area)
        if cat_id_to_name[str(anno['category_id'])] == cat_name:
            cat_count += 1
            cat_bbox_list.append(bbox_area)

rel_bbox_areas = np.asarray(relative_area_list)
print("[Rel] Average bbox size: {}".format(np.mean(rel_bbox_areas)))
print("[Rel] Minimum bbox size: {}".format(np.min(rel_bbox_areas)))
print("[Rel] Maximum bbox size: {}".format(np.max(rel_bbox_areas)))

abs_bbox_areas = np.asarray(bbox_area_list)
print("[Abs] Average bbox size: {}".format(np.mean(abs_bbox_areas)))
print("[Abs] Minimum bbox size: {}".format(np.min(abs_bbox_areas)))
print("[Abs] Maximum bbox size: {}".format(np.max(abs_bbox_areas)))

hist, bin_edges = np.histogram(rel_bbox_areas, bins=10, range=(0.0, 1.0))
plt.bar(bin_edges[:-1], hist, width=0.1, color='blue', align='edge')
plt.title('Relative Box Area Histogram')
plt.show()

bbox_small = np.sum(abs_bbox_areas < 32**2)
print("Total COCO-small: {}".format(bbox_small))
bbox_med = np.sum(abs_bbox_areas < 96**2) - bbox_small
print("Total COCO-medium: {}".format(bbox_med))
bbox_large = np.sum(abs_bbox_areas < 1e5**2) - bbox_med - bbox_small
print("Total COCO-large: {}".format(bbox_large))

for idx in indices:
    sample = train_dataset[idx]
    fname = sample['file_name']
    im_cv2 = cv2.imread(fname)

    for anno in sample['annotations']:
        bbox_xywh = anno['bbox']
        attr_labels = anno['attr_labels']
        category = anno['category_id']

        category_name = cat_id_to_name[str(category)]

        cv2.rectangle(im_cv2,
                      (int(bbox_xywh[0]), int(bbox_xywh[1])),
                      (int(bbox_xywh[0] + bbox_xywh[2]), int(bbox_xywh[1] + bbox_xywh[3])),
                      (255, 0, 0),
                      )
        im_cv2 = cv2.putText(im_cv2,
                             category_name,
                             (int(bbox_xywh[0] + 5), int(bbox_xywh[1] - 5)),
                             cv2.FONT_HERSHEY_SIMPLEX,
                             0.5,
                             (0, 0, 255),
                             1,
                             )

    cv2.imshow("Image", im_cv2)
    cv2.waitKey()

    print("")

import json
import cv2

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

train_indices = [34, 200, 455]
test_indices = [45, 129, 873]

indices = test_indices
dataset = test_dataset
# indices = train_indices
# dataset = train_dataset

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

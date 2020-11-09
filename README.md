# coco4darknet
Repo that does the process of generating all the files and formats needed to train a specific class of the coco dataset using darknet

## Instructions
* Clone the repo.
* Download http://images.cocodataset.org/annotations/annotations_trainval2017.zip and save instances_train2017.json inside the repo.
* update the save_train_path inside create_dataset to the name of your dataset.
* update categories and n_images inside create_images.
* update dict_ids to map the classes into 0-n_classes based on https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
* pip install the requirements.txt
* Run python create_dataset.py

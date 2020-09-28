from pycocotools.coco import COCO
import requests
import csv 
import os
from string import Template
import random


save_train_path = "coco_person_suitcase"
annotations_path = './instances_train2017.json'
categories = ["person", "suitcase"]
#n_images for each class plus n_images for all classes
n_img = 4000

# Dict of mapping of ids where the key is the original coco id and the value
# is the new one such as the final classes are from 0 to nclasses.
dict_ids = {
    1: 0,
    33:1
}


# Create folders if the doesnt exist
if not os.path.exists(save_train_path):
    os.makedirs(save_train_path)
    os.makedirs(f"{save_train_path}/images/")

save_path = f'{save_train_path}/images/'





def download_data(categories):
    existing_images = os.listdir(save_path)
    existing_images = [ei for ei in existing_images if ".jpg" in ei]
    # instantiate COCO specifying the annotations json path
    coco = COCO(annotations_path)
    # Specify a list of category names of interest
    catIds = coco.getCatIds(catNms=categories)
    # Get the corresponding image ids and images using loadImgs
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)
    i = 0
    cont = 0
    new_images = []
    if len(images) < n_img:
    	n_images = len(images)-1
    else:
    	n_images = n_img
    print(n_images, len(images))
    while (cont < n_images) & (i < len(images)):

        if (images[i]["file_name"] not in existing_images):
            new_images.append(images[i])
            cont += 1
        i += 1

    for im in new_images:
        print(im['coco_url'])
        try:
            img_data = requests.get(im['coco_url'], timeout=5).content
        except Exception as e:
            print(e)
            continue

        with open(save_path + im['file_name'], 'wb') as handler:
            handler.write(img_data)

        annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        text_filename = im['file_name'].split(".")[0] + ".txt"
        with open(save_path + text_filename, 'w+') as f:
            writer = csv.writer(f, delimiter=' ')
            for i in range(len(anns)):
                row = [
                    dict_ids[anns[i]["category_id"]],
                    int(round(anns[i]['bbox'][0] + anns[i]['bbox'][2]/2))/ im["width"],
                    int(round(anns[i]['bbox'][1] + anns[i]['bbox'][3]/2))/ im["height"],
                    int(round(anns[i]['bbox'][2])) / im["width"],
                    int(round(anns[i]['bbox'][3]))/ im["height"]
                ]
                writer.writerow(row)


# images with all labels
download_data(categories)
# images with unique label
for cat in categories:
    download_data([cat])


# download_data(categories[0])

existing_images = os.listdir(save_path)

paths = [save_path + p for p in os.listdir(save_path) if ".jpg" in p]

random.shuffle(paths)

train_data = paths[:int(len(paths)*0.9)]
test_data = paths[int(len(paths)*0.9):]


with open(save_train_path + '/train.txt', 'w') as f:
    for item in train_data:
        f.write("%s\n" % item)

with open(save_train_path + '/val.txt', 'w') as f:
    for item in test_data:
        f.write("%s\n" % item)


coco_data = f"""classes = {len(categories)}
names = {save_train_path}/{save_train_path}.names
train  = {save_train_path}/train.txt
valid  = {save_train_path}/val.txt
backup = backup
"""

with open(f"{save_train_path}/{save_train_path}.data", "w") as f: 
    f.write(coco_data) 

coco_names = "\n".join(categories)

with open(f"{save_train_path}/{save_train_path}.names", "w") as f: 
    f.write(coco_names) 


filein = open( 'yolov4_template.cfg' )
src = Template( filein.read() )

n_classes = len(categories)
n_filters = (n_classes + 5) * 3
max_batches = n_classes*2000
d={ 'classes':n_classes, 'filters':n_filters, 'max_batches': max_batches}
conf = src.substitute(d)

with open(f"{save_train_path}/{save_train_path}.cfg", "w") as f: 
    f.write(conf) 


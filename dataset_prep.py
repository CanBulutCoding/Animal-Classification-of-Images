import glob
import csv
import shutil


translate = {"dog": "cane", "horse": "cavallo", "elephant" : "elefante", "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto", "cow": "mucca", "spider": "ragno", "squirrel": "scoiattolo", "sheep": "pecora"}

class_labels = {"dog": 0, "horse": 1, "elephant": 2, "butterfly": 3, "chicken": 4, "cat": 5, "cow": 6, "sheep": 7, "squirrel": 8, "spider": 9}

train_dataset = []
test_dataset = []

for class_name, class_label in class_labels.items():
    class_dir = "archive/raw-img/" + translate[class_name]

    images = glob.glob(class_dir + "/*")

    # using only half of the dataset
    images = images[:len(images) // 2]

    for idx, image_path in enumerate(images):

        if idx > len(images) * 0.8:
            path = f"data/test/images/{idx}.jpeg"
            shutil.copy(image_path, path)
            test_dataset.append((f"images/{idx}.jpeg", class_name, class_label))
        else:
            path = f"data/train/images/{idx}.jpeg"
            shutil.copy(image_path, path)
            train_dataset.append((f"images/{idx}.jpeg", class_name, class_label))

with open("data/train/dataset.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "class_name", "class_label"])
    writer.writerows(train_dataset)

with open("data/test/dataset.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "class_name", "class_label"])
    writer.writerows(test_dataset)

print("Dataset saved")

import csv

import cv2
import numpy as np



class Dataloader:

    def get_sample(self, index: int) -> dict:
        pass

    def __len__(self) -> int:
        pass


class AnimalsDataloader(Dataloader):

    def __init__(self, data_dir: str, num_class: int) -> None:
        self.data_dir = data_dir
        self.num_class = num_class

        samples = []
        with open(f"{data_dir}/dataset.csv", "r") as f:
            reader = csv.DictReader(f)
            for line in reader:
                samples.append(line)
        self.samples = samples

    def get_sample(self, index: int) -> dict:
        sample = self.samples[index]

        full_path = f"{self.data_dir}/{sample['image_path']}"
        image = cv2.imread(full_path, 0)
        image = cv2.resize(image, (256, 256))
        image = image.flatten()
        image = (image / 255) - 0.5
        image = np.expand_dims(image, axis=0)

        class_label = int(sample["class_label"])
        label = np.zeros((1, self.num_class))
        label[0][class_label] = 1

        return {"input": image, "label": label}

    def __len__(self) -> int:
        """
        Return the number of iterations possible considering batch size
        """
        return len(self.samples)

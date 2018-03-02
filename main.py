from batch_generator import BatchGenerator
from PIL import Image
import matplotlib.pyplot as plt

#TODO: Normalize RGB channelse to zero mean and unit variance


dataset_path = 'dataset'
gen = BatchGenerator()
gen.load_dataset(dataset_path)

for batch in gen.triplet_batches(batch_size=5, num_batches=1):
    print(batch)
    for triplet in batch:
        fig = plt.figure()
        for i in range(3):
            fig.add_subplot(1, 3, i+1)
            img = plt.imread(triplet[i])
            plt.imshow(img)
        plt.show()

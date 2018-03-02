import matplotlib.pyplot as plt

def sequence_to_triplets(arr):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(arr), 3):
        yield arr[i:i + 3]

def plot_triplet_sequence(triplet_sequence):
    for triplet in sequence_to_triplets(triplet_sequence):
        fig = plt.figure()
        for i in range(3):
            fig.add_subplot(1, 3, i + 1)
            img = plt.imread(triplet[i])
            plt.imshow(img)
        plt.show()
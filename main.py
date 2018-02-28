from batch_generator import BatchGenerator

dataset_path = 'dataset'
gen = BatchGenerator()
gen.load_dataset(dataset_path)

for batch in gen.triplet_batches(batch_size=1, num_batches=5):
    print(batch)

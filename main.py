from batch_generator import BatchGenerator

dataset_path = 'dataset'
gen = BatchGenerator()
gen.load_dataset(dataset_path)

print(gen._train['cat'])
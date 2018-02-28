import glob

class BatchGenerator():

    def __init__(self):
        self._db = {}
        self._train = {}
        self._test = {}


    def load_dataset(self, path):
        print('Loading Dataset...')
        #db set

        for obj_folder in glob.glob('{}/{}/*/'.format(path,'coarse')):
            obj = obj_folder.rstrip('/').split('/')[-1]
            print(obj_folder)
            self._db[obj] = self.load_object_poses(obj_folder)

        for obj_folder in glob.glob('{}/{}/*/'.format(path,'fine')):
            obj = obj_folder.rstrip('/').split('/')[-1]
            print(obj_folder)
            self._train[obj] = self.load_object_poses(obj_folder)

        with open(path+'/real/training_split.txt') as f:
            training_indexes = [int(i) for i in f.read().strip().split(', ')]

        for obj_folder in glob.glob('{}/{}/*/'.format(path,'real')):
            obj = obj_folder.rstrip('/').split('/')[-1]
            print(obj_folder)
            self._test[obj] = self.load_object_poses(obj_folder, training_indexes.copy(), invert_index=True)
            self._train[obj].update(self.load_object_poses(obj_folder, training_indexes.copy(), invert_index=False))

        print('Finished Loading Dataset!')


    #loads image paths and corresponding poses from folder. returns dict.
    def load_object_poses(self, folder_path, include_index=None, invert_index=None):
        poses = {}
        if include_index is not None:
            include_index.sort()
            index = include_index.pop(0)
        with open(folder_path+'/poses.txt') as f:
            for line in f.readlines():
                if '#' in line:
                    num = self._extract_number(line)
                    img = ''
                    if (include_index is None) or (index==num and not invert_index) or (index!=num and invert_index):
                        if include_index is not None and include_index:
                            index = include_index.pop(0)
                        img = line.strip(' #\n')
                else:
                    if img:
                        pose = [float(i) for i in line.strip('\n').split()]
                        poses[img] = pose
        return poses

    def _extract_number(self, string):
        num_array = [c for c in string if c.isnumeric()]
        num = int(''.join(num_array))
        return num

    #for batch in batchGenerator.triplet_batches(batch_size)
    def triplet_batches(self, batch_size=1):
        pass
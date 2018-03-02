import glob
import numpy as np
import random

class BatchGenerator():

    def __init__(self):
        self._db = {}
        self._train = {}
        self._test = {}


    def load_dataset(self, path):
        print('Loading Dataset...')

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

        self._train_flat = [[obj, img, self._train[obj][img]] for obj in self._train.keys() for img in self._train[obj].keys()]
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
                    img_path = ''
                    if (include_index is None) or (index==num and not invert_index) or (index!=num and invert_index):
                        if include_index is not None and include_index:
                            index = include_index.pop(0)
                        img_path = folder_path + line.strip(' #\n')
                else:
                    if img_path:
                        pose = [float(i) for i in line.strip('\n').split()]
                        poses[img_path] = np.array(pose)
        return poses

    def _extract_number(self, string):
        num_array = [c for c in string if c.isnumeric()]
        num = int(''.join(num_array))
        return num

    #for batch in gen.triplet_batches()
    def triplet_batches(self, batch_size=1, num_batches=1):
        train_flat = self._train_flat.copy()
        np.random.shuffle(train_flat)
        for _ in range(num_batches):
            batch = []
            for _ in range(batch_size):
                if not train_flat:
                    return batch #yield batch return?
                anchor = train_flat.pop(0)
                _, anchor_img, _ = anchor
                puller_img = self._get_puller(anchor)
                pusher_img = self._get_pusher(anchor)
                triplet = [anchor_img, puller_img, pusher_img]
                batch.append(triplet)
            yield batch



    def _get_puller(self, anchor):
        #Find the most similar DB image of the same object type
        anchor_obj, anchor_img, anchor_q = anchor
        smallest_qam = 99   #know arbitrary bound
        puller = ''
        for img, q in self._db[anchor_obj].items():
            qam = self._quaternion_angular_metric(anchor_q, q)
            if qam < smallest_qam:
                smallest_qam = qam
                puller = img
        return puller


    def _get_pusher(self, anchor):
    #random DB image of different object type than anchor
        anchor_obj, anchor_img, _ = anchor
        non_anchor_count = sum([len(self._db[obj]) for obj in self._db.keys() if obj != anchor_obj])
        i = random.randint(0, non_anchor_count-1)
        for obj in self._db.keys():
            if obj == anchor_obj:
                continue
            if i - len(self._db[obj]) > 0:
                i -= len(self._db[obj])
                continue
            return list(self._db[obj].keys())[i]

    def _quaternion_angular_metric(self,q1, q2):
        return 2*np.arccos(np.fabs(np.dot(q1, q2)))

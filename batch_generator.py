__author__ = 'opeide'

import glob
import numpy as np
import random
import cv2
import util
import tensorflow as tf

#Todo: Handle transition between epochs smoothly

class BatchGenerator():

    def __init__(self):
        self._db = {}
        self._train = {}
        self._test = {}


    def load_dataset(self, path):
        print('Loading Dataset...')

        for obj_folder in glob.glob('{}/{}/*/'.format(path,'coarse')):
            obj = obj_folder.rstrip('/').split('/')[-1]
            self._db[obj] = self.load_object_poses(obj_folder)

        for obj_folder in glob.glob('{}/{}/*/'.format(path,'fine')):
            obj = obj_folder.rstrip('/').split('/')[-1]
            self._train[obj] = self.load_object_poses(obj_folder)

        with open(path+'/real/training_split.txt') as f:
            training_indexes = [int(i) for i in f.read().strip().split(', ')]

        for obj_folder in glob.glob('{}/{}/*/'.format(path,'real')):
            obj = obj_folder.rstrip('/').split('/')[-1]
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

    def all_triplets(self):
        return list(self.triplet_batches(batch_size=len(self._train_flat), num_batches=1))[0]

    #for batch in gen.triplet_batches()
    def triplet_batches(self, batch_size=1, num_batches=1):
        train_flat = self._train_flat.copy()
        np.random.shuffle(train_flat)
        for _ in range(num_batches):
            batch = []
            for i in range(batch_size):
                if not train_flat:
                    print('OUT OF DATA! NEW EPOCH!')
                    return batch #yield batch return?
                anchor = train_flat.pop(0)
                _, anchor_img, _ = anchor
                puller_img = self._get_puller(anchor)
                pusher_img = self._get_pusher(anchor)
                triplet = [anchor_img, puller_img, pusher_img]
                batch.extend(triplet)
            yield batch


    #Loads the image paths as tensors
    def train_input_gen(self, num_triplets=1):
        while True:
            batch_img_paths = list(self.triplet_batches(batch_size=num_triplets))[0] #TODO: use next()?
            loaded_batch = []
            for path in batch_img_paths:
                loaded_img = util.loaded_normalized_img(path)
                loaded_img_f = np.float32(loaded_img)
                loaded_batch.append(loaded_img_f)
            loaded_batch_np = np.reshape(loaded_batch, [3*num_triplets,64,64,3])
            labels = None
            yield ({'x': loaded_batch_np}, labels)

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
            return list(self._db[obj].keys())[i-1]

    def _quaternion_angular_metric(self,q1, q2):
        return 2*np.arccos(np.fabs(np.dot(q1, q2)))

    def gen_db(self):
        loaded_batch = []
        i = 0
        for clasification in self._db:
            print(clasification)
            for path in self._db[clasification]:
                loaded_img = cv2.imread(path)
                loaded_img_f = np.float32(loaded_img)
                loaded_batch.append(loaded_img_f)
                i += 1
        print("Set: db", "antal ellementer: ", i)
        return np.reshape(loaded_batch, [i, 64, 64, 3])


    def test_gen(self):
        for clasification in self._test:
            for image in self._test[clasification]:
                loaded_img = cv2.imread(image)
                loaded_img_f = np.float32(loaded_img)
                yield np.reshape(loaded_img_f, [-1, 64, 64, 3]), clasification, self._test[clasification][image]

    def get_classification_and_quaternion_db(self, nr):
        i = 0
        for classification in self._db:
            for image in self._db[classification]:
                if i >= nr:
                    return classification, self._db[classification][image]
                i += 1


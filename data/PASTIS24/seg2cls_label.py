import os
import pickle
from tqdm import tqdm
import numpy as np
import argparse


def process_pickle_files(src_path, num_classes):
    pickle_files = [f for f in os.listdir(src_path) if f.endswith('.pickle')]

    for filename in tqdm(pickle_files, desc="Processing files"):
        file_path = os.path.join(src_path, filename)
        
        with open(file_path, 'rb') as f:
            sample = pickle.load(f, encoding='latin1')
        
        semantic_label = sample['labels'].copy()
        semantic_label[semantic_label==num_classes+1] = 0
        ground_truth = np.zeros(num_classes, dtype=np.uint8)
        unique_values, unique_nums = np.unique(semantic_label[:,:], return_counts=True)
        for j in range(unique_values.shape[0]):
            unique_value = unique_values[j]
            unique_num = unique_nums[j]
            if unique_value!=0 and unique_num>semantic_label.shape[0]*semantic_label.shape[1]*0.01:
                ground_truth[unique_value-1] = 1

        sample['cls_labels'] = ground_truth
        
        target_file_path = os.path.join(src_path, filename)
        with open(target_file_path, 'wb') as f:
            pickle.dump(sample, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Seg2Cls_labels')
    parser.add_argument('--pickle_path', type=str, default="/PASTIS/pickle24x24",
                        help='PASTIS pickles dir')
    parser.add_argument('--num_classes', type=int, default=18,
                        help='number of the categories')
    args = parser.parse_args()

    src_path = args.pickle_path
    num_classes = args.num_classes

    process_pickle_files(src_path, num_classes)
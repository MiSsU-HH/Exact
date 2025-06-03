import os
import pandas as pd
import numpy as np
import multiprocessing
import argparse
import pickle
from scipy.ndimage import zoom
import time
import PIL.Image as Image
from tqdm import tqdm


def eval_cams_oa_and_iou(cam_path, dataset_path, name_list, num_cls=19, threshold=0.0, num_workers=8, img_shape=[24, 24], patch_size=2):
    """
    Evaluate CAMs using Overall Accuracy (OA) and mean Intersection over Union (mIoU).

    Args:
        cam_path (str): Path to the saved CAMs.
        dataset_path (str): Path to the dataset containing ground truth.
        name_list (list): List of image sample names.
        num_cls (int): Number of classes.
        threshold (float): Background score threshold.
        num_workers (int): Number of parallel workers.
        patch_size (int): Size factor to upscale the CAM predictions.

    Returns:
        dict: Dictionary with keys 'mIoU' and 'OA' representing evaluation results.
    """
    # Shared variables for multiprocessing
    TP = [multiprocessing.Value('i', 0, lock=True) for _ in range(num_cls)]
    P = [multiprocessing.Value('i', 0, lock=True) for _ in range(num_cls)]
    T = [multiprocessing.Value('i', 0, lock=True) for _ in range(num_cls)]
    TPTN = [multiprocessing.Value('i', 0, lock=True)]
    TOTAL = [multiprocessing.Value('i', 0, lock=True)]

    def compare(start, step, TP, P, T, TPTN, TOTAL, threshold):
        """
        Worker function for parallel evaluation of predictions.

        Args:
            start (int): Start index for parallel loop.
            step (int): Step size for parallel loop.
            TP, P, T, TPTN, TOTAL: Shared counters for metrics.
            threshold (float): Background threshold for predictions.
        """
        for idx in range(start, len(name_list),step):
            name = name_list[idx].split('/')[1].split('.')[0]
            with open(os.path.join(dataset_path, 'pickle24x24','%s.pickle'%name), 'rb') as handle:
                sample = pickle.load(handle, encoding='latin1')
            seg_gt = sample["labels"]

            predict_dict = np.load(os.path.join(cam_path, '%s.npy'%name), allow_pickle=True).item()
            predict_dict = predict_dict['temporal_cam']
            seg_predict = np.zeros((num_cls, img_shape[0] // patch_size, img_shape[1] // patch_size), np.float32)
            for key in predict_dict.keys():
                seg_predict[key + 1] = predict_dict[key]
            
            # Resize to original shape
            seg_predict = zoom(seg_predict, zoom=(1, patch_size, patch_size), order=1)
            seg_predict[0, :, :] = threshold 
            seg_predict = np.argmax(seg_predict, axis=0).astype(np.uint8)

            # Valid area mask  
            cal = seg_gt < num_cls
            mask = (seg_predict == seg_gt) * cal

            # Update shared counters
            with TPTN[0].get_lock():
                TPTN[0].value += np.sum(mask)
            with TOTAL[0].get_lock():
                TOTAL[0].value += np.sum((seg_gt == seg_gt) * cal)
            for i in range(num_cls):
                with P[i].get_lock():
                    P[i].value += np.sum((seg_predict == i) * cal)
                with T[i].get_lock():
                    T[i].value += np.sum((seg_gt == i) * cal)
                with TP[i].get_lock():
                    TP[i].value += np.sum((seg_gt == i) * mask)  

    # Launch multiprocessing
    p_list = []
    for i in range(num_workers):
        p = multiprocessing.Process(target=compare, args=(i, num_workers, TP, P, T, TPTN, TOTAL, threshold))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    
    # Compute IoU and OA
    IoU = [TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10) for i in range(num_cls)]            
    miou = np.mean(np.array(IoU))
    oa = TPTN[0].value * 1.0 / (TOTAL[0].value + 1e-10)

    return {'mIoU': miou * 100, 'OA': oa * 100}

def writelog(filepath, metric: dict):
    """
    Append a dictionary of evaluation metrics to a log file with timestamp.

    Args:
        filepath (str): Path to the log file.
        metric (dict): Dictionary of metrics to write.
    """
    with open(filepath, 'a') as logfile:
        logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
        line = ''.join([f'{key}:{value}  ' for key, value in metric.items()])
        logfile.write(line + '\n')
        logfile.write('=====================================\n')

def generate_pseudo_label(save_cams_path, save_pseudo_label_path, step, bg_thr, num_classes, img_shape=[24, 24], patch_size=2):
    """
    Generate pseudo labels from CAMs.
    
    Args:
        save_cams_path (str): Path to directory containing CAM files
        save_pseudo_label_path (str): Output directory for pseudo labels
        step (int): 
        bg_thr (float): 
        num_classes (int): Total number of classes (including background)
        img_shape (list): Original image dimensions [height, width]
        patch_size (int): Patch size used in the model
    """
    save_cams_path = os.path.join(args.save_cams_path, 'cams_'+str(step))
    if not os.path.exists(save_pseudo_label_path):
        os.makedirs(save_pseudo_label_path) 
    file_list = [file_name for file_name in os.listdir(save_cams_path)]
    for file_name in tqdm(file_list, desc='Converting', unit='file'):
        file_path = os.path.join(save_cams_path, file_name)
        cam = np.load(file_path, allow_pickle=True).item()
        cam = cam['temporal_cam']
        tensor = np.zeros((num_classes, img_shape[0] // patch_size, img_shape[1] // patch_size), np.float32)
        for key in cam.keys():
            tensor[key+1] = cam[key]
        tensor = zoom(tensor, zoom=(1, patch_size, patch_size), order=1)
        tensor[0,:,:] = bg_thr 
        pseudo_label = np.argmax(tensor, axis=0).astype(np.uint8)      
        image = Image.fromarray(pseudo_label)
        output_file_name = os.path.splitext(file_name)[0] + '.png'        
        output_file_path = os.path.join(save_pseudo_label_path, output_file_name)
        image.save(output_file_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_list", type=str)
    parser.add_argument("--save_cams_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--steps", nargs='+', default=[4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000], type=int)
    parser.add_argument('--eval_log_path', default='./eval_log.txt',type=str)
    parser.add_argument('--num_classes', default=19, type=int)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=60, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument("--save_pseudo_label_path", type=str)
    parser.add_argument("--img_shape", default=[24,24], type=int)
    args = parser.parse_args()

    if isinstance(args.steps, int):
        args.steps = [args.steps]
    
    # Load sample list
    df = pd.read_csv(args.sample_list, names=['filename'])
    name_list = df['filename'].values

    global_max_oa = 0.0
    global_best_thr_of_oa = 0.0
    global_best_step_of_oa = -1

    for step in args.steps:
        cam_path = os.path.join(args.save_cams_path, 'cams_'+str(step))
        if not os.path.exists(cam_path):
            continue     

        max_mIoU = 0.0
        max_oa = 0.0
        best_thr_of_miou = 0.0
        best_thr_of_oa = 0.0
        print('Evaluate step %d'%step)
        for i in range(args.start, args.end):
            threshold = i/100.0
            loglist = eval_cams_oa_and_iou(cam_path, args.dataset_path, name_list, args.num_classes, threshold, args.num_workers, args.img_shape)
            print('%d/60 background score: %.3f\tmIoU: %.3f%%\tOA: %.3f%%'%(i, threshold, loglist['mIoU'], loglist['OA']))
            if loglist['mIoU'] > max_mIoU:
                max_mIoU = loglist['mIoU']
                best_thr_of_miou = threshold
            if loglist['OA'] > max_oa:
                max_oa = loglist['OA']
                best_thr_of_oa = threshold
        print('Best background score(miou): %.3f\tmIoU: %.3f%%' % (best_thr_of_miou, max_mIoU))
        print('Best background score(OA): %.3f\tOA: %.3f%%' % (best_thr_of_oa, max_oa))
        writelog(args.eval_log_path, {
            'Step': step,
            'Best mIoU': max_mIoU,
            'Best threshold(mIoU)': best_thr_of_miou,
            'Best OA': max_oa,
            'Best threshold(OA)': best_thr_of_oa
        })
        if max_oa > global_max_oa:
            global_max_oa = max_oa
            global_best_thr_of_oa = best_thr_of_oa
            global_best_step_of_oa = step
    print('best step: %d\tBest background score(OA): %.3f\tOA: %.3f%%'%(global_best_step_of_oa, global_best_thr_of_oa, global_max_oa))
    writelog(args.eval_log_path, {
        'best step': global_best_step_of_oa,
        'best threshold(OA)': global_best_thr_of_oa,
        'best OA': global_max_oa
    })

    generate_pseudo_label(args.save_cams_path, args.save_pseudo_label_path, global_best_step_of_oa, global_best_thr_of_oa, args.num_classes, args.img_shape, patch_size=2)

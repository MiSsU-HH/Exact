# [CVPR 2025 Highlight ✨] *Exact*: Exploring Space-Time Perceptive Clues for Weakly Supervised Satellite Image Time Series Semantic Segmentation 

This repository contains the source code of [*Exact*: Exploring Space-Time Perceptive Clues for Weakly Supervised Satellite Image Time Series Semantic Segmentation.](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Exact_Exploring_Space-Time_Perceptive_Clues_for_Weakly_Supervised_Satellite_Image_CVPR_2025_paper.pdf)

<br>
  <img width="100%" alt="AFA flowchart" src="./docs/exact.png">
</div>



## :white_check_mark:Updates

* **`Jun. 3th, 2025`**:  The implementation code has been released.




## Get Started

### Environment

- Ubuntu 20.04, with Python 3.8.0, PyTorch 1.12.0, CUDA 11.6, multi gpus(8) - Nvidia RTX 3090.
- You can install all dependencies with the provided requirements file.

```bash
pip install -r requirements.txt
```

### Data Preparations

<details>
<summary>
PASTIS dataset
</summary>

The original PASTIS dataset is accessible [here](https://github.com/VSainteuf/pastis-benchmark). We follow the [TSViT](https://github.com/michaeltrs/DeepSatModels/blob/main/README_TSVIT.md) to divide each sample into 24x24 patches by running the script:

```bash
python data/PASTIS24/data2windows.py --rootdir <...> --savedir <...> --HWout 24
```

The reorganized directory should be:

```
PASTIS
├── pickle24x24
│   ├── 40562_9.pickle
│   └── ...
├── fold-paths
│   ├── fold_1_paths.csv
│   └── ...
```

In addition, we generate multi-class labels for each patch by running the following script:

```bash
python data/PASTIS24/seg2cls_label.py --pickle_path <...>/PASTIS/pickle24x24 
```

</details>

<details>
<summary>
Germany dataset
</summary>


The original Germany dataset is accessible [here](https://github.com/MarcCoru/MTLCC), we can download the dataset (40GB) via:

```bash
wget https://zenodo.org/record/5712933/files/data_IJGI18.zip
```

The size of each sample in Germany dataset is 24x24, so we only need to generate the multi-class labels with the above script without splitting. 

</details>

## Usage
```bash
bash run.sh $workspace $dataset_path
```



## Citation

Please cite our work if you find it helpful to your research.

```bibtex
@inproceedings{zhu2025exact,
  title={Exact: Exploring space-time perceptive clues for weakly supervised satellite image time series semantic segmentation},
  author={Zhu, Hao and Zhu, Yan and Xiao, Jiayu and Xiao, Tianxiang and Ma, Yike and Zhang, Yucheng and Dai, Feng},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={14036--14045},
  year={2025}
}
```



## Acknowledgement

This repo is built upon [TSViT](https://github.com/michaeltrs/DeepSatModels) and [PASTIS](https://github.com/VSainteuf/pastis-benchmark), thanks for their excellent works!


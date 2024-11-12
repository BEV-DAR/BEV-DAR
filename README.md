## Hi there 👋

<!--
**BEV-DAR/BEV-DAR** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.
![image]([https://user-images.githubusercontent.com/77472466/162715638-145897ba-2c35-4734-b6a7-b30048ab80f8.png](https://github.com/BEV-DAR/BEV-DAR/blob/main/compare.png))
Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
This repositary contains the official Pytorch implementation for paper BEV-DAR: Enhancing Monocular Bird's Eye View Segmentation with Depth-Aware Rasterization


## Install
To use our code, please install the following dependencies:
* torch==1.9.1
* torchvison==0.10.1
* mmcv-full==1.3.15
* CUDA 9.2+

## Data Preparation
We conduct experiments of [nuScenes](https://www.nuscenes.org/download), [Argoverse](https://www.argoverse.org/),
Please down the datasets and place them under /data/nuscenes/ and so on. Note that *calib.json* contains the intrinsics and extrinsics matrixes of every image. 
Refer to the script *[make_labels](https://github.com/tom-roddick/mono-semantic-maps/blob/master/scripts)* to get the BEV annotation for nuScenes and Argoverse, respectively. The datasets' structures look like: 
### Dataset Structure
```
data
├── nuscenes
|   ├── img_dir
|   ├── ann_bev_dir
|   ├── calib.json
├── argoversev1.0
|   ├── img_dir
|   ├── ann_bev_dir
|   ├── calib.json
├── kitti_processed
|   ├── kitti_raw
|   |   ├── img_dir
|   |   ├── ann_bev_dir
|   |   ├── calib.json
|   ├── kitti_odometry
|   |   ├── img_dir
|   |   ├── ann_bev_dir
|   |   ├── calib.json
|   ├── kitti_object
|   |   ├── img_dir
|   |   ├── ann_bev_dir
|   |   ├── calib.json

### Prepare calib.json
"calib.json" contains the camera parameters of each image. Readers can generate the "calib.json" file by the instruction of [nuScenes](https://www.nuscenes.org/nuscenes#download), [Argoverse](https://www.argoverse.org/), [Kitti Raw](http://www.cvlibs.net/datasets/kitti/raw_data.php), [Kitti Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php), and [Kitti 3D Object](http://www.cvlibs.net/datasets/kitti/eval_3dobject.php). We also upload *calib.json* for each dataset to [google drive](https://drive.google.com/drive/folders/1Ahaed1OsA1EqlJOCHHN-MQQr2VpF8H7U?usp=sharing) and [Baidu Net Disk](https://pan.baidu.com/s/1wEzHWkazS5vLPZJVjpzHMw?pwd=2022).

## Training
Take nuScenes as an example. To train a semantic segmentation model under a specific configuration, run:
```
cd DAR
```
python -m torch.distributed.launch --nproc_per_node 1 --master_port 14300 tools/train.py ./configs/DAR/DAR_nuscenes.py --work-dir ./models_dir/DAR_nuscenes --launcher pytorch
```
## Evaluation
To evaluate the performance, run the following command:
```
cd DAR
```
python -m torch.distributed.launch --nproc_per_node ${NUM_GPU} --master_port ${PORT} tools/test.py ${CONFIG} ${MODEL_PATH} --out ${SAVE_RESULT_PATH} --eval ${METRIC} --launcher pytorch
```
For example, we evaluate the mIoU on nuScenes by running:
```
cd DAR
```
python -m torch.distributed.launch --nproc_per_node 1 --master_port 14300 tools/test.py ./configs/DAR/DAR_nuscenes.py ./models_dir/DAR_nuscenes/iter_90000.pth  --out ./results/DAR_nuscenes/DAR_nuscenes.pkl --eval mIoU --launcher pytorch
```
## Visulization
To get the visulization results of the model, we first change the output_type from 'iou' to 'seg' in the testing process.
```
    # change the output_type from 'iou' to 'seg'
    test_cfg=dict(mode='whole',output_type='seg',positive_thred=0.5)
)
```
And then, we can generate the visualization results by running the following command:
```
python -m torch.distributed.launch --nproc_per_node 4 --master_port 14300 tools/test.py ./configs/DAR/DAR_nuscenes.py ./models_dir/DAR_nuscenes/iter_20000.pth --format-only --eval-options "imgfile_prefix=./models_dir/pyva_swin_argoverse" --launcher pytorch
```
## Acknowledgement
Our work is partially based on [mmseg](https://github.com/open-mmlab/mmsegmentation) and [HFT](https://github.com/JiayuZou2020/HFT/tree/main). Thanks for their contributions to the research community.


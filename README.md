# EV-Eye: Rethinking High-frequency Eye Tracking through the Lenses of Event Cameras
## Citation
If you are using this dataset in your paper, please cite the following paper: 

```
@inproceedings{  
  title = {EV-Eye: Rethinking High-frequency Eye Tracking through the Lenses of Event Cameras},  
  author = {Guangrong Zhao, Yiran Shen, Yurun Yang, Jingwei Liu, Ning Chen, Hongkai Wen, Guohao Lan},
  booktitle = {Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, New Orleans, USA},
  year = {2023}  
} 
```

## Introduction EV-Eye
EV-Eye introduces the largest and most diverse multi-modal frame-event dataset for high frequency eye tracking in the literature. 

<div align=center style="display:flex;">
  <img src="pictures/samples.png" alt="iou" style="flex:1;" width="350" height="180">
</div>

EV-Eye proposes a novel hybrid frame-event eye tracking benchmarking approach tailored to the collected dataset, capable of tracking the pupil at a frequency up to 38.4kHz. 
<div align=center style="display:flex;">
 <img src="pictures/main.jpg" alt="iou" style="flex:1;" width="900" height="300" >
</div>

[//]: # (![summary]&#40;pictures/samples.png&#41;)
[//]: # ()
[//]: # (![summary]&#40;pictures/main.jpg&#41;)

<br/>

## Overview
The repository includes an introduction to EV-Eye **Dataset organization** and how to **Running the benchmark** in Python and Matlab. **If you need the original paper in PDF format, please contact the author's email: guangrong.zhao@sdu.edu.cn**
<!-- ## A quick Youtube demo for introduction
[![IMAGE_ALT](pictures/EV.png)](https://youtu.be/Yi03mFAyslU)
 -->
 



## Dataset Organization
The EV-Eye dataset is a first-of-its-kind large scale multimodal eye tracking dataset, utilizing an emerging bio-inspired event camera to capture independent pixel-level intensity changes induced by eye movement, achieving submicrosecond latency. 

The dataset was collected from 48 participants encompassing diverse genders and age group. Each participant participates in four sessions of data collection. The first tow sessions capture both saccade and fixation state of the movement, the last two sessions record eye movement in smooth pursuit.

The dataset comprises over 1.5 million near-eye grayscale images and 2.7 billion event samples generated by two DAVIS346 event cameras.

Additionally, the dataset contains 675 thousands scene images and 2.7 million gaze references captured by Tobii Pro Glasses 3 eye tracker for cross-modality validation.

You can download the **EV_Eye_dataset** from [https://1drv.ms/f/s!Ar4TcaawWPssqmu-0vJ45vYR3OHw](https://1drv.ms/f/s!Ar4TcaawWPssqmu-0vJ45vYR3OHw), the **EV_Eye_dataset** consisting of `raw_data` and `processed_data`. 
- The **raw_data** includes three folders, i.e., `Data_davis`, `Data_davis_labelled_with_mask` and `Data_tobii` which will be described in detail in the following. 
  - We add a new folder `Data_davis_pupil_iris_label` to the **raw_data** directory for future work.
- The **processed data** is the data/results processed by our Python and Matlab code, it will be described in the **Running the benchmark** section.

### raw_data

----------------------

#### 1. Data_davis

The **Data_davis** folder consists of the data generated by DAVIS346 event cameras. 

This directory contains 48 subdirectories corresponding to 48 participants. The data are divided as `'left'` and `'right'` corresponding to each eye in every user directory. The four sessions' data is contained in four separate directory. Within each session directory is a `'frames'` folder contains near-eye grayscale images and a `'events'` folder contains the event streams.

We leverage the [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html) to label the pupil region of 9,011 near-eye images selected uniformly across the image dataset, annotation results are recorded in Excel tables in the last three sessions, e.g., `/EV_Eye_dataset/raw_data/Data_davis/user1/left/session_1_0_2/user_1.csv`. The `creation_time.txt` file records the system time of the computer when DAVIS346 started collecting.

  ```
  ─Data_davis
  ├─user1
  │  ├─left
  │  │  ├─session_1_0_1
  │  │  │  ├─events
  │  │  │  └─frames
  │  │  ├─session_1_0_2
  │  │  │  ├─events
  │  │  │  ├─frames
  │  │  │  └─user_1.csv
  │  │  ├─session_2_0_1
  │  │  │  ├─events
  │  │  │  ├─frames
  │  │  │  └─user_1.csv
  │  │  ├─session_2_0_2
  │  │  │  ├─events
  │  │  │  ├─frames
  │  │  │  └─user_1.csv
  │  │  ├─creation_time.txt
  │  └─right
  │      ..........
  ```
  
--------------------

#### 2. Data_davis_labelled_with_mask
**Data_davis_labelled_with_mask** offers the pre-generated binarized pupil masks according to the above-mentioned annotation results, i.e., the Excel tables in `/EV_Eye_dataset/raw_data/Data_davis`. The code to generate masks can be found in `/matlab_processed/generate_pupil_mask.m`. 

The masks are saved in hdf5 files, which are then used for training the DL-based pupil segmentation network.
  ```
  ─Data_davis_labelled_with_mask
  ├─left
  │  ├─user1_session_1_0_2.h5
  │  ├─user1_session_2_0_1.h5
  │  │─user1_session_2_0_2.h5
  │  ..........
  ├─right
  │  ├─user1_session_1_0_2.h5
  │  ├─user1_session_2_0_1.h5
  │  │─user1_session_2_0_2.h5
  │  ..........
  ```
  
--------------------

#### 3. Data_tobii
**Data_tobii** includes the gaze references provided by Tobii Pro Glasses 3. The detailed description about `gazedata`, `scenevideo`, `imudata` and `eventdata` can be find in: [https://www.tobii.com/products/eye-trackers/wearables/tobii-pro-glasses-3#form](https://www.tobii.com/products/eye-trackers/wearables/tobii-pro-glasses-3#form). The `tobiisend.txt` file records the system time of the computer when TTL signal is send to Tobii Pro Glasses 3, the `tobiittl.txt` records the TTL signal receiving time in the glasses internal clock. 
  ```
  -Data_tobii
  ├─ user1 
  │  ├─tobiisend.txt
  │  ├─tobiittl.txt
  │  ├─session_1_0_1
  │        ├─gazedata
  │        ├─scenevideo
  │        ├─imudata
  │        ├─eventdata
  |  ..........
  ```
  
--------------------

#### 4. Data_davis_pupil_iris_label

**Data_davis_pupil_iris_label** includes labels of pupil and iris region for frames in a continuous period in each session. The labeled period for each session are recorded in `label_statistic.xlsx`,  the statistics in the file provide information about the state of the eyes during those periods, such as blink and moving direction.

We leverage the [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html) tool to label the pupil region and iris region. The labels in the same image are indexed with `1` and `2`, representing to pupil and iris region, respectively.

 ```
  ─Data_davis_pupil_iris_label
  ├─left
  │  ├─session_1_0_1
  │  │  ├─left101user1_labelled.csv
  │  │  ├─left101user2_labelled.csv
  |  |  ├─left101user3_labelled.csv
  |  |  | ........
  │  ├─session_1_0_2
  │  ├─session_2_0_1
  │  └─session_2_0_2
  ├─right
  └─label_statistic.xlsx
  ```

--------------------

To access more information about the data curation process and data characteristics, kindly refer to Section 3 of the corresponding paper.
<br/>


## Running the Benchmark
Four metrics are adopted for the dataset evaluation, namely **IoU and F1 score**, **Pixel error (PE) of frame-based pupil segmentation**, **PE of event-based pupil tracking**, **Difference of direction (DoD) in gaze tracking**. 

* The **IoU and F1 score** are used to evaluate pupil region segmentation task, and we use pytorch framework in Python to train and evaluate our DL-based Pupil Segmentation network.

* The **PE of frame-based pupil segmentation**, **PE of event-based pupil tracking**, **DoD in gaze tracking** implemented through Matlab code.
 
### Download Dataset

You can download the `raw_data` and `processed_data` in **EV_Eye_dataset** from [https://1drv.ms/f/s!Ar4TcaawWPssqmu-0vJ45vYR3OHw](https://1drv.ms/f/s!Ar4TcaawWPssqmu-0vJ45vYR3OHw) to the `/path/to/EV_Eye_dataset` directory and run the following code to unzip them:

```
cd /path/to/EV_Eye_dataset #choose your own path

#upzip. Following command will extract all the datasets.

find . -mindepth 1 -maxdepth 3 -name '*.rar' -execdir unrar x {} \; -execdir mv {} ./ \;
```

Please place the unzipped data in the `/path/to/EV_Eye_dataset` directory and arrange it according to the following path.

```angular2html
  EV_Eye_dataset
  ├─ raw_data 
  │  ├─Data_davis
  │  ├─Data_davis_labelled_with_mask
  │  ├─Data_tobii
  ├─ processed_data 
  │  ├─Data_davis_predict
  │  ├─Frame_event_pupil_track_result
  │  ├─Pixel_error_evaluation
  │  ├─Pre-trained
```

<!-- By default, our code find the dataset in the `./EV_Eye_dataset` directory. So if you use the default settings, you need to download and unzip the dataset into `./EV_Eye_dataset`. -->

### Python
Note: please use Python >= 3.6.0

#### Python Requirements

```
torch>=1.9.0
numpy>=1.21.0
tqdm>=4.61.1
h5py>=3.2.1
torchvision>=0.10.0
argparse>=1.1
```

To install requirements:

```angular2html
pip install -r requirements.txt
```

#### Training

To train the DL-based Pupil Segmentation network models, run this command:

```
python train.py 
```

Optional arguments can be passed :
* `--whicheye`  to select which eye data to use for training, such as "L" or "R".
* `--batch_size ` 

#### Evaluation of **IoU and F1 score**
The following code provides the calculation method of **IoU and F1 score**:

```
python evaluate.py 
```

#### Predict
```angular2html
python predict.py
```
Optional arguments can be passed :
* `--whicheye`  to select which eye data to use for prediction, such as "L" or "R".
* `--predict` the user ID , for example, '1'. 
* `--output` the output directory for the prediction results, default`./EV_Eye_dataset/processed_data/Data_davis_predict`.

#### Pre-trained models

you can find Pre-trained_models in `./EV_Eye_dataset/processed_data/Pre-trained_models`, it contains our DL-based Pupil Segmentation network pre-trained models using the left and right eyes of each of the 48 participants.

### Matlab
#### Install the Requirement

```angular2html
matlab -batch "pkg install -forge io"
matlab -batch "pkg install -forge curvefit"
```

#### Evaluation of PE of frame-based pupil segmentation & PE of event-based pupil tracking
Run the following codes to estimated the Euclidean distance in pixels between the estimated and manually marked groundtruth pupil centers, and the results  will be saved by default in folder `./EV_Eye_dataset/processed_data/Pixel_error_evaluation/frame` and `./EV_Eye_dataset/processed_data/Pixel_error_evaluation/event`, respectively. You can also find the results that we got before in those folders.
```
./matlab_processed/pe_of_frame_based_pupil_track.m
./matlab_processed/pe_of_event_based_pupil_track.m

#plot the results as bar charts
./matlab_processed/plot_bar_frame_pe.m
./matlab_processed/plot_bar_event_pe.m
``` 

#### Evaluation of DoD in gaze tracking
We use the following matlab scripts to obtain **DoD in gaze tracking**.

First, the following code shows how to obtain frame&event-based pupil tracking results for 48 participants. The results will be saved by default in folder `./EV_Eye_dataset/processed_data/Frame_event_pupil_track_result`, and you can also find the results that we got before in that folder.
```
./matlab_processed/frame_event_pupil_track.m
```
A corresponding visual demonstration is as follows,
```
./matlab_processed/frame_event_pupil_track_plot.m
```
Second, the following code shows how to get the frame&event-based pupil tracking estimated results in folder `./EV_Eye_dataset/processed_data/Frame_event_pupil_track_result` correspond to the reference provided by the Tobii Pro Glasses 3 in  `./EV_Eye_dataset/raw_data/Data_tobii`
```
./matlab_processed/frame_event_pupil_track_result_find_tobii_reference.m
```
Third, the following code shows how to estimate the difference between the estimated and reference gaze directions. 
```
./matlab_processed/evaluation_on_gaze_tracking_with_polynomial_regression.m

#visual 
./matlab_processed/plot_gaze_tracking_dod.m
```



[//]: # (## Results)

[//]: # (##### IoUs and F1 scores on frame-based pupil segmentation.)

[//]: # ()
[//]: # (<br/>)

[//]: # (<div style="display:flex;">)

[//]: # (  <img src="pictures/iou_new.png" alt="iou" style="flex:1;">)

[//]: # (  <img src="pictures/dice.png" alt="iou" style="flex:1;">)

[//]: # (</div>)

[//]: # ()
[//]: # ()
[//]: # (<br/>)

[//]: # ()
[//]: # (##### The pixel error of frame-based and event-based pupil tracking.)

[//]: # ()
[//]: # (<br/>)

[//]: # ()
[//]: # ()
[//]: # (![event]&#40;pictures/event_pixel.png&#41;)

[//]: # (![frame]&#40;pictures/frame_pixel.png&#41;)

[//]: # ()
[//]: # (<br/>)

[//]: # ()
[//]: # (##### DoDs of model-based method vs. ours with respect to the gaze references.)

[//]: # ()
[//]: # (<br/>)

[//]: # ()
[//]: # (<img src="pictures/distance.png" style="margin-left: 6px">)


<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons
Attribution-NonCommercial 4.0 International License</a>.

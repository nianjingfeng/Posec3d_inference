## Guide

This model utilizes the action recognition method in the factory assembly operation, and this work applies this algorithm to a real-life situation for real-time inference. You can understand how the model is used by [realtime_inference_v101.py](https://github.com/nianjingfeng/Posec3d_inference/blob/master/realtime_inference_v101.py).<br>[Click here](https://openaccess.thecvf.com/content/CVPR2022/papers/Duan_Revisiting_Skeleton-Based_Action_Recognition_CVPR_2022_paper.pdf) to read the algorithm structure about Posec3d.

Setting
---
The Posec3d model is built in mmaction packages, [st-gcn](https://arxiv.org/pdf/1801.07455.pdf) can also be used from this package. [Click here](https://github.com/open-mmlab/mmaction2#installation) to set up the environment.
[Click here](https://google.github.io/mediapipe/getting_started/python.html) to set up the mediapipe environment, or you can use built-in pose estimation model in mmaction.

Models
---
This model contains two stage:
<br>Pose Estimation : Posec3d ask the keypoint of human body as input data, pose estimation is an algorithm to detect the joint and edge of human, and translate to position information. <br>In this model, I apply [mediapipe](https://google.github.io/mediapipe/), an superior pose estimation algorithm which built by google research. Please note that mediapipe does not currently support multi-detection, this also means that only single-human action recognition is supported in this model. <br>I utilized 17 points in mediapipe hands module as model's input, there are number 0,1,4,5,8,9,12,16,20 of left hand and 0,1,4,8,9,12,16,20 of right hand in the image. You can also select the module or input node as you want in the model, just modify the config file to train and inference.
![hand image](https://mediapipe.dev/images/mobile/hand_landmarks.png)
Action recognition : In this stage, you can easily inference the model by running inference file, you can edit line 95 to change the source of video, or you can use camera to collect the video. The input size of the data is set as 30 frames, and update the input data per frame.


Inference
---

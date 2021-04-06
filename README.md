# plant-pathology-2021-fgvc8
## ~~See as a multi-classification problem~~
1. ~~'scab'~~
2. ~~'healthy'~~
3. ~~'frog_eye_leaf_spot'~~
4. ~~'rust'~~
5. ~~'complex'~~
6. ~~'powdery_mildew'~~
7. ~~'scab frog_eye_leaf_spot'~~
8. ~~'scab frog_eye_leaf_spot complex'~~
9. ~~'frog_eye_leaf_spot complex'~~
10. ~~'rust frog_eye_leaf_spot'~~
11. ~~'rust complex'~~
12. ~~'powdery_mildew complex'~~
## See as a multi label problem
* label:{<br/>
&emsp;'scab': [1, 0, 0, 0, 0, 0],<br/>
&emsp;'healthy': [0, 1, 0, 0, 0, 0],<br/>
&emsp;'frog_eye_leaf_spot': [0, 0, 1, 0, 0, 0],<br/>
&emsp;'rust': [0, 0, 0, 1, 0, 0],<br/>
&emsp;'complex': [0, 0, 0, 0, 1, 0],<br/>
&emsp;'powdery_mildew': [0, 0, 0, 0, 0, 1],<br/>
&emsp;'scab frog_eye_leaf_spot': [1, 0, 1, 0, 0, 0],<br/>
&emsp;'scab frog_eye_leaf_spot complex': [1, 0, 1, 0, 1, 0],<br/>
&emsp;'frog_eye_leaf_spot complex': [0, 0, 1, 0, 1, 0],<br/>
&emsp;'rust frog_eye_leaf_spot': [0, 0, 1, 1, 0, 0],<br/>
&emsp;'rust complex': [0, 0, 0, 1, 1, 0],<br/>
&emsp;'powdery_mildew complex': [0, 0, 0, 0, 1, 1]<br/>
}<br/>
* F1-Score的计算应该基于'scab'、'healthy'、'frog_eye_leaf_spot‘等而非'rust complex'<br/>
* 所以之前的训练集上的F1-Score都只能达到~50％<br/>
## EfficientNetB0 Train
* 学习率lr ∈ [1e-4, 2e-4] for batch_size = 16<br/>
* 学习率lr = 1e-3 for batch_size = 128 or 64<br/>
* 应用学习率衰减，val_f1_score连续5个epoch不下降就降低学习率<br/>
* EfficientNetB0 - EfficientNetB7均采用相同参数，使用 noisy-student 权重<br/>
## 训练集上采用的图像增强方法
* tf.keras.layers.GaussianNoise()<br/>
* tf.image.random_contrast()<br/>
* tf.image.random_saturation()<br/>
* tf.image.random_brightness()<br/>
* tf.image.random_flip_left_right()<br/>
* tf.image.random_flip_up_down()<br/>
* tf.image.rot90()<br/>
## TODO
* 训练集上增加 随机遮挡 数据增强<br/>
* 训练集上使用 MixUp 数据增强<br/>
* 训练集上使用 labelsmooth<br/>
* TTA（测试时增强）<br/>
* ~~清洗训练集中的标签错误标签数据（imagehash）~~<br/>
* 生成tfrecords的时候移除duplicates_compare_result中的
* 不平衡数据处理 过采样/欠采样<br/>
* 试 ResNet 系列<br/>

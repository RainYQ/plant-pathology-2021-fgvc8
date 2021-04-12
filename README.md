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
## See as a multi-label problem
* label:{<br/>
&emsp;'scab': [1, 0, 0, 0, 0], <br/>
&emsp;'healthy': [0, 0, 0, 0, 0], <br/>
&emsp;'frog_eye_leaf_spot': [0, 1, 0, 0, 0], <br/>
&emsp;'rust': [0, 0, 1, 0, 0], <br/>
&emsp;'complex': [0, 0, 0, 1, 0], <br/>
&emsp;'powdery_mildew': [0, 0, 0, 0, 1], <br/>
&emsp;'scab frog_eye_leaf_spot': [1, 1, 0, 0, 0], <br/>
&emsp;'scab frog_eye_leaf_spot complex': [1, 1, 0, 1, 0], <br/>
&emsp;'frog_eye_leaf_spot complex': [0, 1, 0, 1, 0], <br/>
&emsp;'rust frog_eye_leaf_spot': [0, 1, 1, 0, 0], <br/>
&emsp;'rust complex': [0, 0, 1, 1, 0], <br/>
&emsp;'powdery_mildew complex': [0, 0, 0, 1, 1] <br/>
}<br/>
* F1-Score 的计算应该基于 'scab'、'healthy'、'frog_eye_leaf_spot‘等而非'rust complex' <br/>
* 所以之前的训练集上的 F1-Score 都只能达到~50％ <br/>

## EfficientNet Train

* 学习率 lr ∈ [1e-4, 2e-4] for batch_size = 16 <br/>
* 学习率 lr = 1e-3 for batch_size = 128 or 64 <br/>
* 应用学习率衰减，val_f1_score 连续 5 个 epoch 不下降就降低学习率 <br/>
* EfficientNet-B0 - EfficientNet-B7 均采用相同参数，使用 noisy-student 权重作为初始权重 <br/>
* EfficientNet-B4 metric = ~86％ for batch_size = 128 <br/>
* EfficientNet-B4 F1-Score = ~85％ for batch_size = 128 <br/>
* EfficientNet-B7 metric = ~82％ for batch_size = 128 <br/>
* EfficientNet-B7 F1-Score = ~84％ for batch_size = 128 <br/>
* Use FocalLoss (处理数据集不平衡) <br/>
* EfficientNet B7 single model (th = 0.5) LB = 0.638 <br/>
* EfficientNet B4 single model (th = 0.5) LB = 0.585 <br/>

## ResNet50 Train

* 学习率lr = 5e-5 for batch_size = 16 <br/>
* 应用学习率衰减，val_f1_score 连续 5 个 epoch 不下降就降低学习率 <br/>
* metric = ~78％ for batch_size = 16 <br/>
* F1-Score = ~62％ for batch_size = 16 <br/>
* Use FocalLoss (处理数据集不平衡)<br/>

## 训练集上采用的图像增强方法

* tf.keras.layers.GaussianNoise() <br/>
* tf.image.random_contrast() <br/>
* tf.image.random_saturation() <br/>
* tf.image.random_brightness() <br/>
* tf.image.random_flip_left_right() <br/>
* tf.image.random_flip_up_down() <br/>
* tf.image.rot90() <br/>

## TODO

**说明优先级高 <br/>
* 训练集上增加 随机遮挡 数据增强 <br/>
* 训练集上使用 MixUp 数据增强 <br/>
* 训练集上使用 labelsmooth <br/>
* 改成两个模型，第一个分辨是 'healthy' 还是 'ill' , 第二个分辨具体是哪种疾病 <br/>
* ~~试试看做异常检出问题, 标签中删除 'healthy' , 没有疾病检出时即为 healthy~~  <br/>
* TTA (测试时增强) (TTA 步长不能太大，容易超时) (需要加速 Inference) <br/>
* **加速 Inference 为 Test Dataset 生成 tfrecords <br/>
* 试试不使用 Focal Loss 时的准确率 <br/>
* 试试余弦学习率衰减 <br/>
* ~~清洗训练集中的标签错误标签数据（imagehash）~~ <br/>
* ~~生成tfrecords的时候移除重复图片~~ <br/>
* 不平衡数据处理 过采样/欠采样 <br/>
* ~~试 ResNet 系列~~ <br/>
* ~~tfa.metrics.F1Score 只支持 { 'none' 'macro' 'micro' 'weighted' } , 不支持{ 'sample' },
  需要重写或者使用 scikit-learn 中的 F1-Score~~ <br/>
* Adam优化器在训练快结束的时候效果不是很好，看看有没有更合适的优化器 <br/>
* 默认的 F1-Score 计算方法是每个 Batch 计算一次然后取均值, 应该改成每个 Epoch 结束时计算一次更合适, 
  reference: https://zhuanlan.zhihu.com/p/51356820 (不能在 tensorflow 2.0 以上使用) <br/>

# MathJax Need!
* https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima/related
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
* F1-Score 的计算应该基于 'scab'、'healthy'、'frog_eye_leaf_spot‘ 等而非 'rust complex' <br/>
* 所以之前的训练集上的 F1-Score 都只能达到 ~50％ <br/>

## EfficientNet Train

* 学习率 lr ∈ [1e-4, 2e-4] for batch_size = 16 <br/>
* 学习率 lr = 1e-3 for batch_size = 128 or 64 <br/>
* 应用余弦退火学习率调整策略，调整周期为 10 epochs <br/>
* EfficientNet-B0 - EfficientNet-B7 均采用相同参数
* 使用 noisy-student 权重作为初始权重 <br/>
* Loss : Use Soft Sample-Wise F1 Loss <br/> 
* Metrics : Use Sample-Wise F1 Score <br/> 
* EfficientNet B7 single model (th = 0.5) LB = 0.830 <br/>

## ResNet50 Train

* 学习率 lr = 5e-5 for batch_size = 16 <br/>
* 应用学习率衰减，val_f1_score 连续 5 个 epoch 不下降就降低学习率 <br/>
* (batch_size = 16) Macro F1-Score = ~62％<br/>
* Use FocalLoss (处理数据集不平衡) <br/>

## 训练集上采用的图像增强方法

* tf.image.random_jpeg_quality() <br/>
* tf.keras.layers.GaussianNoise() <br/>
* tf.image.random_contrast() <br/>
* tf.image.random_saturation() <br/>
* tf.image.random_brightness() <br/>
* tf.image.random_flip_left_right() <br/>
* tf.image.random_flip_up_down() <br/>
* tfa.image.random_cutout() <br/>
* tfa.image.rotate() <br/>
* tf.image.random_crop() <br/>

## TODO

**说明优先级高 <br/>
**加粗**说明效果优秀 <br/>
* 训练集上使用 labelsmooth 按照上次比赛的经验, labelsmooth 一般不会对性能有提升但也不会下降 <br/>
* 不平衡数据处理 过采样/欠采样 <br/>
* **ResNet 用当前的优化器和学习率策略无法正常收敛 <br/>
* 关注 'complex' 这个过于 noisy 的类 <br/>  
* **移除错误标签 没试, 但是从预测结果来看模型没被错误标签带偏, 都很信任自己的判断结果 <br/> 
* ~~**尝试一下去除图像标准化**~~ 效果优秀 <br/>
* ~~阈值选择~~ 没有效果, 全部都是更糟糕了 <br/>  
* ~~训练集上使用 cutout 数据增强~~ 该死的 TPU 和 tfa.image.random_cutout 不兼容, 把整个函数 copy 过来结束战斗, 
  可以正常运行在 TPU <br/>
* ~~**试试余弦学习率衰减/周期学习率衰减/Warmup**~~ 效果优秀, 中后期能稳定提高准确率 ( **仅在 EfficientNet 系列中有效 ) <br/>
* ~~Adam优化器在训练快结束的时候效果不是很好，看看有没有更合适的优化器~~ <br/>
* ~~**训练集上使用 MixUp 数据增强**~~ 效果优秀 LB 0.830  <br/>
* ~~**Use Soft-Samples-Wise F1 Loss**~~ 感觉效果不错 Single Model LB 0.803 <br/>
* ~~试试看做异常检出问题, 标签中删除 'healthy' , 没有疾病检出时即为 healthy~~  <br/>
* ~~TTA (测试时增强) (TTA 步长不能太大，容易超时) (需要加速 Inference)~~ TTA 效果不稳定, 无法稳定提高准确率, 一般都会变差 <br/>
* ~~**加速 Inference 为 Test Dataset 生成 tfrecords~~ <br/>
* ~~试试不使用 Focal Loss 时的准确率~~ <br/>
* ~~生成 tfrecords 的时候移除重复图片和错误标签~~ <br/>
* ~~试 ResNet 系列~~ 本地效果比 EfficientNet 系列差很多 <br/>
* ~~tfa.metrics.F1Score 只支持 { 'none' 'macro' 'micro' 'weighted' } , 不支持{ 'sample' },
  需要重写或者使用 scikit-learn 中的 F1-Score~~ 重写了 Metrics <br/>
* ~~默认的 F1-Score 计算方法是每个 Batch 计算一次然后取均值, 应该改成每个 Epoch 结束时计算一次更合适, 
  reference: https://zhuanlan.zhihu.com/p/51356820 (不能在 tensorflow 2.0 以上使用)~~ 
  Sample Wise F1-Score 与 batch_size 无关, 不需要考虑 <br/>

## Kaggle
### DataSet
* https://www.kaggle.com/rainyq/tfrecords-rainyq-600 Train DataSet (600x600) <br/>
* https://www.kaggle.com/rainyq/tfrecords-rainyq-512 Train DataSet (512x512) <br/>
### Model
* https://www.kaggle.com/rainyq/efficientnetb4tpu <br/>
### Code
* https://www.kaggle.com/rainyq/tfrecords-generator <br/>
* https://www.kaggle.com/rainyq/tfrecords-600 <br/>
* https://www.kaggle.com/rainyq/inference <br/>
* https://www.kaggle.com/rainyq/train <br/>
### Others
* https://www.kaggle.com/rainyq/train-data-without-rep <br/>
* https://www.kaggle.com/rainyq/offiline-pip-package <br/>

## Information
### Train
* Time Limit: ~32400s <br/>
* ~7000s per 1-fold train for EfficientNet-B7 (Epoch = 30, size = 600x600) <br/>
* ~4400s per 1-fold train for EfficientNet-B7 (Epoch = 30, size = 512x512) <br/>
* ~2480s per 1-fold train for EfficientNet-B4 (Epoch = 30, size = 512x512) <br/>
### Inference
* Test TFRecords Generate ~20 minutes <br/>
* Efficient-B7 Model Predict ~4 minutes per model (512x512) <br/>
* Run Inference ~40 minutes in K=1 TTA_STEP=4 (512x512) <br/>
* Commit Inference ~42 minutes in K=1 TTA_STEP=4 (600x600) <br/>
### F1-Score
#### Micro-F1
$$ \{P} = \frac{{\overline {TP} }}{{\overline {TP}  + \overline {FP} }}\ $$
$$ \{R} = \frac{{\overline {TP} }}{{\overline {TP}  + \overline {FN} }}\ $$
$$ \{F1} = 2 * \frac{{P * R}}{{P + R}}\ $$
#### Macro-F1
$$ \{P_i} = \frac{{{TP_i}}}{{{TP_i} + {FP_i}}}\ $$
$$ \{R_i} = \frac{{{TP_i}}}{{{TP_i} + {FN_i}}}\ $$
$$ \{F1} = 2 * \frac{{\overline {P}  * \overline {R} }}{{\overline {P}  + \overline {R} }}\ $$
#### Samples-F1
* Sample-Wise Multilabel Confusion Matrix <br/>
$$
 \begin{bmatrix}
   TN & FP \\\\
   FN & TP \\\\
  \end{bmatrix}
$$
* Example: <br/>
  * y_true: [[1,1,0,0,0,0]] <br/>
  * y_pred: [[0,1,1,0,0,0]] <br/>
* MCM: <br/>
$$
 \begin{bmatrix}
   3 & 1 \\\\
   1 & 1 \\\\
  \end{bmatrix}
$$
* Calculate P R in sample_wise:  <br/>
$$ \{P_1} = \frac{1}{{1 + 1}}\ $$
$$ \{R_1} = \frac{1}{{1 + 1}}\  $$
$$ \{F1_1} = 2 * \frac{{0.5 * 0.5}}{{0.5 + 0.5}}\ $$
* Calculate Average P R F1 in sample_wise <br/>
$$ \{F1} = 0.5$$

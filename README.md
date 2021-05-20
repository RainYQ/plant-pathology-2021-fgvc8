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

* Adam 学习率 lr ∈ [1e-4, 2e-4] batch_size = 16 <br/>
* Adam 学习率 lr = 1e-3 batch_size = 128 or 64 <br/>
* SGD 学习率 lr_max = 6e-2 lr_min = 1e-4 Restart_Cycle = 10 epochs batch_size = 128 or 64 <br/>
* 应用带重启机制的余弦退火学习率调整策略，调整周期为 10 epochs <br/>
* EfficientNet-B0 - EfficientNet-B7 均采用相同参数
* 使用 noisy-student 权重作为初始权重 <br/>
* Loss : Use Soft Sample-Wise F1 Loss <br/> 
* Metrics : Use Sample-Wise F1 Score <br/> 
* EfficientNet B7 single model (th = 0.5) LB = 0.830 <br/>
* EfficientNet B0 single model (th = 0.5) TTA_STEP = 8 LB = 0.831 <br/>
* EfficientNet B7 5-fold model (th = 0.5) TTA_STEP = 4 LB = 0.847 <br/>

## 训练集上采用的图像增强方法

* tf.image.random_jpeg_quality() <br/>
* tf.keras.layers.GaussianNoise() <br/>
* tf.image.random_contrast() <br/>
* tf.image.random_saturation() <br/>
* tf.image.random_hue() <br/>
* tf.image.random_brightness() <br/>
* tf.image.random_flip_left_right() <br/>
* tf.image.random_flip_up_down() <br/>
* tfa.image.random_cutout() <br/>
* tfa.image.rotate() <br/>
* tf.image.random_crop() <br/>

## TODO

**说明优先级高 <br/>
**加粗**说明效果优秀 <br/>
* ~~训练集上使用 labelsmooth 按照上次比赛的经验, labelsmooth 一般不会对性能有提升但也不会下降~~ 
  有的 fold 提升大约 1%, 但很多时候几乎没有提升 <br/>
* 不平衡数据处理 过采样/欠采样 <br/>
* ~~Pseudo Labeling~~ 现在采用的策略是:
  - plant-pathology-2020-fgvc7/train 中的图像直接使用 train.csv 中的标记
  , 将 ’multiple_diseases‘ 视为 ‘complex’, 'scab'、'rust'、'healthy' 保持不变, 'powdery_mildew'、
  'frog_eye_leaf_spot' 视为没有 <br/> 
  - plant-pathology-2020-fgvc7/test 中的图像使用 plant-pathology-2020-fgvc7 code 中能找到的最高 Score 的 submission.csv
     一样按照上述方法进行标签映射, **不做阈值化处理**, 直接作为 Soft Labels, 将这两份 labels 合并记为 Ground Truth Labels <br/> 
  - 获取 EfficientNetB7 - 5 Fold - 4 TTA STEP ( LB 0.847 ) 在整个 plant-pathology-2020-fgvc7/train + 
    plant-pathology-2020-fgvc7/test 中的预测, 同样按照上述方法进行标签映射, **不做阈值化处理**, 直接作为 Soft Labels, 
    记为 Model Predict Labels <br/> 
  - 上述两份 Label 按照 0.7 * Ground Truth Labels + 0.3 * Model Predict Labels 记为 Pseudo Labels <br/> 
  - 暂时没做置信度过滤, 因为考虑到 'rust' 类的泛化性特别差, 经常以高置信度判断错误, 做置信度过滤应该没啥大作用并且可靠性差 <br/> 
  - 仅将这个额外的数据集合入训练集, 验证集中不包含此数据集 <br/> 
  - 代码已经合入本地的 train.py 和 https://www.kaggle.com/rainyq/train-pseudo , 
    在 EfficientNetB7 Fold 0 上测试, 没有任何性能提升 <br/> 
* ~~ResNet 用当前的优化器和学习率策略无法正常收敛~~ ResNet 学习率调整至 5e-4, 其他各模型均为 1e-3 <br/>
* 关注 'complex' 这个过于 noisy 的类 <br/> 
* 模型没办法找到准确的 boundary for 'scab' and 'rust', 在去年的数据集中存在大量将其他类型误判为 'rust' 的错误 
* **移除错误标签 没试, 但是从预测结果来看模型没被错误标签带偏, 都很信任自己的判断结果 <br/> 
* ~~**尝试一下去除图像标准化**~~ 效果优秀, 但是不知道怎么解释这个现象 <br/>
* ~~阈值选择~~ 没有效果, 全部都是更糟糕了, **但有必要将 'rust' 的阈值调高** <br/>  
* ~~训练集上使用 cutout 数据增强~~ 该死的 TPU 和 tfa.image.random_cutout 不兼容, 把整个函数 copy 过来结束战斗, 
  可以正常运行在 TPU <br/>
* ~~**试试余弦学习率衰减/周期学习率衰减/Warmup**~~ 效果优秀, 中后期能稳定提高准确率 ( **仅在 EfficientNet 系列中有效 ) <br/>
* ~~Adam优化器在训练快结束的时候效果不是很好，看看有没有更合适的优化器~~ <br/>
* ~~**训练集上使用 MixUp 数据增强**~~ 效果优秀 LB 0.830  <br/>
* ~~**Use Soft-Samples-Wise F1 Loss**~~ 感觉效果不错 Single Model LB 0.803 <br/>
* ~~**试试看做异常检出问题, 标签中删除 'healthy' , 没有疾病检出时即为 healthy**~~  <br/>
* ~~TTA (测试时增强) (TTA 步数不能太多，容易超时) (需要加速 Inference)~~ TTA STEP >= 4 时有稳定的提升 <br/>
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
* https://www.kaggle.com/rainyq/extra-train-dateset Pseudo Labels
### Model
* https://www.kaggle.com/rainyq/efficientnetb4tpu <br/>
### Code
* https://www.kaggle.com/rainyq/tfrecords-generator <br/>
* https://www.kaggle.com/rainyq/tfrecords-600 <br/>
* https://www.kaggle.com/rainyq/inference <br/>
* https://www.kaggle.com/rainyq/train <br/>
* https://www.kaggle.com/rainyq/train-pseudo <br/>
### Others
* https://www.kaggle.com/rainyq/train-data-without-rep <br/>
* https://www.kaggle.com/rainyq/offiline-pip-package <br/>

## Information
### Train
* Time Limit: ~32400s <br/>
* ~11000s 1-fold for EfficientNet-B7 Add pseudo labels (Epoch = 80, size = 512x512) <br/>
* ~12304s 2-fold  for InceptionResNetV2 (Epoch = 80, size = 512x512) <br/>
* ~9329s 2-fold for ResNet50 (Epoch = 80, size = 512x512) <br/>
### Inference
* Test TFRecords Generate ~5.3 minutes ( Use GPU Speed up )<br/>
* Efficient-B7 Model Predict ~4 minutes per model (512x512) <br/>
* InceptionResNetV2 Model Predict ~3 minutes per model (512x512) <br/>
* ResNet50 Model Predict ~2 minutes per model (512x512) <br/>
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

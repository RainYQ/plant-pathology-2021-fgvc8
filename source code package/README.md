# tfrecords generate
- Default
```commandline
py -3.8 ./tfrecords_generator.py --train_images './train_images/' --number_in_tfrecord 128 --label './train_without_rep.csv' --njob 16
```
- visulization on
```commandline
py -3.8 ./tfrecords_generator.py --train_images './train_images/' --number_in_tfrecord 128 --label './train_without_rep.csv' --visualization --njob 16
```
# pseudo labels generate
```commandline
py -3.8 ./pseudo_tfrecords.py --extra_train_images './plant-pathology-2020-fgvc7/all/' --number_in_tfrecord 128 --pseudo_labels './pseudo_data.csv'
```
# train
- Default
```commandline
py -3.8 ./train.py --train_images ./train_images --label './train_without_rep.csv' --mix_up --random_resize --label_smooth --batch_size 128 --iteration 128 --epoch 100 --learning_rate_max 1e-3 --learning_rate_min 1e-6 --cycle 10
```
- SGD
```commandline
py -3.8 ./train.py --train_images ./train_images --label './train_without_rep.csv' --mix_up --random_resize --label_smooth --batch_size 128 --iteration 128 --epoch 100 --learning_rate_max 6e-2 --learning_rate_min 1e-4 --cycle 10 --use_sgd
```
- Pseudo Labels
```commandline
py -3.8 ./train.py --train_images ./train_images --label './train_without_rep.csv' --mix_up --random_resize --label_smooth --batch_size 128 --iteration 128 --epoch 100 --learning_rate_max 1e-3 --learning_rate_min 1e-6 --cycle 10 --pseudo_labels
```
# inference
- Default
```commandline
py -3.8 ./inference.py --test_image './test_images/ad8770db05586b59.jpg' --model_path './model' --random_crop --use_tta --tta_step 4 --resize 600 --crop 512
```
- Use probability
```commandline
py -3.8 ./inference.py --test_image './test_images/ad8770db05586b59.jpg' --model_path './model' --random_crop --use_tta --tta_step 4 --resize 600 --crop 512 --use_probability --prob_vector 0.25 0.25 0.5 0.5 0.75
```
- No TTA
```commandline
py -3.8 ./inference.py --test_image './test_images/ad8770db05586b59.jpg' --model_path './model' --resize 600 --crop 512
```

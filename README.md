# triplet_loss_playground

Based on https://www.tensorflow.org/addons/tutorials/losses_triplet

Also see https://youtu.be/AwQHqWyHRpU?t=1450

```bash
sudo docker build ~/triplet_loss_playground --tag=triplet_loss_playground
mkdir -p ~/triplet_loss_playground/tfds_data
sudo docker run -u $(id -u):$(id -g) --gpus all -it -v ~/triplet_loss_playground:/home/triplet_loss_playground triplet_loss_playground bash
python src/fit_model.py
```

```
Epoch 200/200
1875/1875 [==============================] - 28s 15ms/step - loss: 0.1982
Sanity check: are all model outputs unit vectors, as expected?
True
```

```
Distances between my hand drawn Xs:
[[0.         1.14174511 0.79931824]
 [1.14174511 0.         0.68824712]
 [0.79931824 0.68824712 0.        ]]
Distances between mnist digits and my hand drawn Xs:
min=0.3518602813396214, mean=1.3891414920301075, median=1.4871171013090958, max=1.9194239968854707
Min distances between mnist digits and my hand drawn Xs:
[0.45689708 0.35186028 0.3848151 ]
Classes of the mnist digits closest to my hand drawn Xs:
[6 8 8]
```

```
Distances between my hand drawn 8s:
[[0.         0.71040955 0.54930204]
 [0.71040955 0.         0.59276718]
 [0.54930204 0.59276718 0.        ]]
Distances between mnist digits and my hand drawn 8s:
min=0.19103638002534976, mean=1.3873805569013555, median=1.4724980990818684, max=1.8657132767900848
Min distances between mnist digits and my hand drawn 8s:
[0.19740385 0.53029324 0.19103638]
Classes of the mnist digits closest to my hand drawn 8s:
[8 8 8]
Distances between my hand-drawn 8s and Xs:
[[1.07054016 0.50638428 0.59712251]
 [1.409789   0.62809385 1.11967489]
 [1.23026855 0.50905983 0.78410262]]
```
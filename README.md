# triplet_loss_playground

Based on https://www.tensorflow.org/addons/tutorials/losses_triplet

Also see https://youtu.be/AwQHqWyHRpU?t=1450

```bash
sudo docker build ~/triplet_loss_playground --tag=triplet_loss_playground
sudo docker run -u $(id -u):$(id -g) --gpus all -it -v ~/triplet_loss_playground:/home/triplet_loss_playground triplet_loss_playground bash
python src/fit_model.py
```
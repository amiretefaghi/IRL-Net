import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from PIL import Image
from networks_p2 import SE_Proposed_model
from tqdm import tqdm
import matplotlib.pyplot as plt
from os import listdir, walk
import argparse


def im_file_to_tensor(img,mask):
  def _im_file_to_tensor(img,mask):
    path = f"{img.numpy().decode()}"
    im = Image.open(path)
    im = im.resize((256,256))
    im = np.array(im).astype(float) / 255.0
    path = f"{mask.numpy().decode()}"
    mask = Image.open(path).convert('L')
    mask = mask.resize((256,256))
    mask = np.array(mask).astype(float) / 255.0
    # if len(mask.shape) == 3:
    #   mask = np.mean(mask,axis=-1)
    mask = np.expand_dims(mask,axis=-1)
    return im, mask
  return tf.py_function(_im_file_to_tensor,
                        inp=(img,mask),
                        Tout=(tf.float32,tf.float32))


def Create_dataset(images_path,masks_path,text_file='../imglist_train.txt',batch_size = 8, type_dir='dir',train_data=True):

  img_paths = []
  mask_paths = []
  if type(images_path) == list:
    for i in range(len(images_path)):
      for (root,dirs,files) in walk(images_path[i], topdown=True):
        if files != []:
          for name_file in files:
            img_paths.append(root + '/' + name_file)
            mask_paths.append(root.replace(images_path[i],masks_path[i]) + '/' + name_file.split('.')[0] + '.jpg')
  elif type(images_path) == str:
      for (root,dirs,files) in walk(images_path, topdown=True):
        if files != []:
          for name_file in files:
            img_paths.append(root  + '/' + name_file)
            mask_paths.append(root.replace(images_path,masks_path)  + '/' + name_file.split('.')[0] + '.jpg')

  img_paths = np.array(img_paths)
  mask_paths = np.array(mask_paths)
  print(len(img_paths))

  indx = np.asarray(range(len(img_paths)))
  np.random.shuffle(indx)
  img_paths = img_paths[indx]
  mask_paths = mask_paths[indx]

  #step 1
  img_names = tf.constant(img_paths)
  mask_names = tf.constant(mask_paths)

  # step 2: create a dataset returning slices of `filenames`
  dataset = tf.data.Dataset.from_tensor_slices((img_names,mask_names))

  dataset = dataset.map(im_file_to_tensor)
  batch_dataset = dataset.prefetch(4000).batch(batch_size)

  return batch_dataset


def rgb2gray(rgb):
    b, g, r = rgb[:, :, :,0], rgb[:, :, :,1], rgb[:, :, :,2]
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    gray = tf.expand_dims(gray, axis=-1)
    return gray


def DiceBCELoss(targets, inputs, smooth=1e-6):
    alfa = 0.75
    landa = 2
    #flatten label and prediction tensors

    inputs = keras.layers.Flatten()(inputs)
    targets = keras.layers.Flatten()(targets)

    BCE =  tf.reduce_mean(alfa*((1-inputs)**landa)*targets*tf.math.log(inputs+smooth)+(1-alfa)*(inputs**landa)*(1-targets)*tf.math.log(1-inputs+smooth),axis=1)
    BCE =  -tf.reduce_mean(BCE,axis=0)
    intersection = tf.reduce_sum(inputs*targets,axis=1)
    dice_loss = 1 - (2*intersection) / (tf.reduce_sum(targets**2,axis=1) + tf.reduce_sum(inputs**2,axis=1) + smooth)
    dice_loss = tf.reduce_mean(dice_loss,axis=0)
    Dice_BCE = BCE + dice_loss

    return Dice_BCE

@tf.function
def train(images,masks,model,opt):
  # gray_images = rgb2gray(images)

  with tf.GradientTape() as tape1:
    # noisy = GaussianNoise(images,training=True)
    # gray = rgb2gray(images)
    out_ = model(images,training=True)

    loss_ = DiceBCELoss(masks,out_)


    loss = loss_ #+ loss_refine
  grads = tape1.gradient(loss,model.trainable_variables)
  opt.apply_gradients(zip(grads,model.trainable_variables))


  return loss, out_

def main(args):
    train_dataset = Create_dataset(images_path=[args.train_images_path],
                                   masks_path=[args.train_masks_path],
                                   batch_size=args.batch_size)
    test_dataset = Create_dataset(images_path=[args.test_images_path],
                                  masks_path=[args.test_masks_path],
                                  batch_size=args.batch_size)

    GaussianNoise = tf.keras.layers.GaussianNoise(stddev=args.stddev)

    version = args.version
    total_epoch = args.total_epoch
    initial_epoch = args.initial_epoch
    continue_ = args.continue_

    model_ = SE_Proposed_model(input_shape=(256,256,3))

    if continue_:
        model_.load_weights(f'./Weights/{version}_latest.h5')

    step = tf.Variable(initial_epoch*(50000/16), trainable=False)
    boundaries = [25000, 45000]
    values = [4e-4, 2e-4, 1e-4]
    learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)

    # Later, whenever we perform an optimization step, we pass in the step.
    learning_rate = learning_rate_fn(step)

    # opt = keras.optimizers.Adam(learning_rate)
    opt = keras.optimizers.Adam(1e-4)


    train_IoU = tf.keras.metrics.MeanIoU(num_classes=2)
    val_IoU = tf.keras.metrics.MeanIoU(num_classes=2)
    val_recall = tf.keras.metrics.Recall()
    val_precision = tf.keras.metrics.Precision()

    for epoch in tqdm(range(total_epoch)):
        epoch_ = epoch + initial_epoch
        total_loss = 0
        train_IoU.reset_state()

        for i,(images,masks) in enumerate(train_dataset):
            loss, out_ = train(images,masks,model_,opt)
            total_loss += loss
            train_IoU.update_state(masks,tf.round(out_))

            if i%100 == 0:
                print(f'loss in epoch {epoch_ + 1} in iter {i}: {loss}')
            print(f'loss in epoch {epoch_ + 1} : {total_loss/(i+1)}')
            print(f'mIoU of train set in epoch {epoch + 1}: {train_IoU.result().numpy()}')

        val_IoU.reset_state()
        val_recall.reset_state()
        val_precision.reset_state()
        for test_images, test_masks in test_dataset:
            # gray_test_images = rgb2gray(test_images)
            test_out_= model_(test_images)
            val_IoU.update_state(test_masks,tf.round(test_out_))
            val_recall.update_state(test_masks,tf.round(test_out_))
            val_precision.update_state(test_masks,tf.round(test_out_))

        print(f'mIoU of validation set in epoch {epoch + 1}: {val_IoU.result().numpy()}')
        print(f'Recall of validation set in epoch {epoch + 1}: {val_recall.result().numpy()}')
        print(f'Precision of validation set in epoch {epoch + 1}: {val_precision.result().numpy()}')
        print(f'F1 score of validation set in epoch {epoch + 1}: {2*(val_recall.result().numpy()*val_precision.result().numpy())/(val_recall.result().numpy()+val_precision.result().numpy()+1e-7)}')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model for image processing.')
    parser.add_argument('--train_images_path', type=str, default='../datasets/GC_Places/train/inpainted',
                        help='Path to training images')
    parser.add_argument('--train_masks_path', type=str, default='../datasets/GC_Places/train/mask',
                        help='Path to training masks')
    parser.add_argument('--test_images_path', type=str, default='../datasets/GC_Places/test/inpainted/',
                        help='Path to test images')
    parser.add_argument('--test_masks_path', type=str, default='../datasets/GC_Places/test/mask/',
                        help='Path to test masks')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and testing')
    parser.add_argument('--stddev', type=float, default=0.00, help='Standard deviation for Gaussian Noise layer')
    parser.add_argument('--version', type=str, default='SE_proposed_Places_GC', help='Model version')
    parser.add_argument('--total_epoch', type=int, default=10, help='Total number of epochs')
    parser.add_argument('--initial_epoch', type=int, default=0, help='Initial epoch to start training from')
    parser.add_argument('--continue_', type=bool, default=True, help='Flag to continue training from a checkpoint')

    args = parser.parse_args()
    main(args)
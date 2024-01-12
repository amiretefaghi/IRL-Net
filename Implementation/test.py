import tensorflow as tf
import numpy as np
from PIL import Image
from networks_p2 import SE_Proposed_model
import os
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
#   print(len(img_paths))

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

def main(args):
    # Load test dataset
    test_dataset = Create_dataset(images_path=[args.test_images_path],
                                  masks_path=[args.test_masks_path],
                                  batch_size=args.batch_size)

    # Load model
    model_ = SE_Proposed_model(input_shape=(256, 256, 3))
    model_.load_weights(args.model_weights_path)

    # Make sure the output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Process each batch in the test dataset
    for i, (test_images, test_masks) in enumerate(test_dataset):
        # Get model output
        test_out_ = model_(test_images)

        # Convert model output to images and save
        for j, img in enumerate(test_out_):
            img_array = (img.numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_array)
            img_pil.save(os.path.join(args.output_path, f"output_{i * args.batch_size + j}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a model for image processing and save output.')
    parser.add_argument('--test_images_path', type=str, required=True, help='Path to test images')
    parser.add_argument('--test_masks_path', type=str, required=True, help='Path to test masks')
    parser.add_argument('--model_weights_path', type=str, required=True, help='Path to the model weights')
    parser.add_argument('--output_path', type=str, default='./output', help='Path to save output images')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for testing')

    args = parser.parse_args()
    main(args)

"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record
  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'upper-body clothes':
        return 1
    elif row_label == 'lower-body clothes':
        return 2
    elif row_label == 'full-body clothes':
        return 3
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(example, path_root):
    # import image
    f_image = Image.open(path_root + example["image_name"])

    # get width and height of image
    width, height = f_image.size

    # crop image randomly around bouding box within a 0.15 * bbox extra range
    if FLAGS.evaluation_status != "test":

        left = example['x_1'] - round((random.random() * 0.15 + 0.05) * (example['x_2'] - example['x_1']))
        top = example['y_1'] - round((random.random() * 0.15 + 0.05) * (example['y_2'] - example['y_1']))
        right = example['x_2'] + round((random.random() * 0.15 + 0.05) * (example['x_2'] - example['x_1']))
        bottom = example['y_2'] + round((random.random() * 0.15 + 0.05) * (example['y_2'] - example['y_1']))

        if left < 0: left = 0
        if right >= width: right = width
        if top < 0: top = 0
        if bottom >= height: bottom = height

        f_image = f_image.crop((left, top, right, bottom))
        _width, _height = width, height
        width, height = f_image.size

    # read image as bytes string
    encoded_image_data = io.BytesIO()
    f_image.save(encoded_image_data, format='jpeg')
    encoded_image_data = encoded_image_data.getvalue()

    filename = example["image_name"]  # Filename of the image. Empty if image is not from file
    filename = filename.encode()

    image_format = 'jpeg'.encode()  # b'jpeg' or b'png'

    if FLAGS.evaluation_status != "test":
        xmins = [(example['x_1'] - left) / width]  # List of normalized left x coordinates in bounding box (1 per box)
        xmaxs = [(example['x_2'] - left) / width]  # List of normalized right x coordinates in bounding box (1 per box)
        ymins = [(example['y_1'] - top) / height]  # List of normalized top y coordinates in bounding box (1 per box)
        ymaxs = [(example['y_2'] - top) / height]  # List of normalized bottom y coordinates in bounding box (1 per box)
    else:
        xmins = [example['x_1'] / width]  # List of normalized left x coordinates in bounding box (1 per box)
        xmaxs = [example['x_2'] / width]  # List of normalized right x coordinates in bounding box (1 per box)
        ymins = [example['y_1'] / height]  # List of normalized top y coordinates in bounding box (1 per box)
        ymaxs = [example['y_2'] / height]  # List of normalized bottom y coordinates in bounding box (1 per box)

    assert (xmins[0] >= 0.) and (xmaxs[0] < 1.01) and (ymins[0] >= 0.) and (ymaxs[0] < 1.01), \
        (example, _width, _height, width, height, left, right, top, bottom, xmins, xmaxs, ymins, ymaxs)

    if width < 50 or height < 50 \
        or (xmaxs[0] - xmins[0]) / (ymaxs[0] - ymins[0]) < 0.2 \
        or (xmaxs[0] - xmins[0]) / (ymaxs[0] - ymins[0]) > 5.:
        return None

    if FLAGS.categories == 'broad':
        classes_text = [LABEL_DICT[example['category_type']].encode()]  # List of string class name of bounding box (1 per box)
        classes = [example['category_type']]  # List of integer class id of bounding box (1 per box)
    elif FLAGS.categories == 'fine':
        classes_text = [example['category_name'].encode()]  # List of string class name of bounding box (1 per box)
        classes = [example['category_label']]  # List of integer class id of bounding box (1 per box)
    else:
        raise (ValueError("Incorrect value for flag categories. Must be 'broad' or 'fine'."))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()



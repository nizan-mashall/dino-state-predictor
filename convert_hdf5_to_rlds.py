import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os

input_path = '/code/data/demo_with_images.hdf5'
output_path = '/code/data/rlds_dataset/lift_robosuite/1.0.0'
os.makedirs(output_path, exist_ok=True)

f = h5py.File(input_path, 'r')
demos = list(f['data'].keys())
print(f"Converting {len(demos)} demos...")

def generate_episodes():
    for demo_key in demos:
        demo = f['data'][demo_key]
        actions = demo['actions'][:]
        images = demo['obs']['robot0_eye_in_hand_image'][:]
        T = len(actions)
        
        steps = []
        for t in range(T):
            image_tensor = tf.constant(images[t], dtype=tf.uint8)
            step = {
                'observation': {
                    'image': image_tensor,
                },
                'action': tf.constant(actions[t], dtype=tf.float32),
                'language_instruction': tf.constant('pick up the red cube', dtype=tf.string),
                'is_first': tf.constant(t == 0, dtype=tf.bool),
                'is_last': tf.constant(t == T-1, dtype=tf.bool),
                'is_terminal': tf.constant(t == T-1, dtype=tf.bool),
            }
            steps.append(step)
        yield {'steps': steps}

# Write as TFRecord
writer = tf.io.TFRecordWriter(f'{output_path}/lift_robosuite-train.tfrecord-00000-of-00001')

for i, demo_key in enumerate(demos):
    demo = f['data'][demo_key]
    actions = demo['actions'][:]
    images = demo['obs']['robot0_eye_in_hand_image'][:]
    T = len(actions)
    
    steps = []
    for t in range(T):
        image_encoded = tf.image.encode_jpeg(
            tf.constant(images[t], dtype=tf.uint8)
        ).numpy()
        
        step_feature = {
            'observation/image': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image_encoded])
            ),
            'action': tf.train.Feature(
                float_list=tf.train.FloatList(value=actions[t].tolist())
            ),
            'language_instruction': tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=['pick up the red cube'.encode()]
                )
            ),
            'is_first': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(t == 0)])
            ),
            'is_last': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(t == T-1)])
            ),
            'is_terminal': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(t == T-1)])
            ),
        }
        steps.append(tf.train.Example(
            features=tf.train.Features(feature=step_feature)
        ).SerializeToString())
    
    # Write episode
    episode_feature = {
        'steps': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=steps)
        ),
    }
    example = tf.train.Example(
        features=tf.train.Features(feature=episode_feature)
    )
    writer.write(example.SerializeToString())
    
    if (i+1) % 10 == 0:
        print(f"Converted {i+1}/{len(demos)} demos")

writer.close()
f.close()
print("Done!")

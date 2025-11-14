import tensorflow as tf
from pipeline.constants import IMG_SIZE

def preprocessing_fn(inputs):
    image = inputs['image']
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return {'image': image, 'label': inputs['label']}

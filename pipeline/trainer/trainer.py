import tensorflow as tf
from pipeline.trainer.model import build_model
from pipeline.constants import IMG_SIZE

# -----------------------------
# FIXED parse_fn  (IMPORTANT)
# -----------------------------
def parse_fn(example):
    feature_spec = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    ex = tf.io.parse_single_example(example, feature_spec)

    # Decode image safely
    img = tf.io.decode_image(
        ex['image/encoded'],
        channels=3,
        expand_animations=False
    )

    # Give TensorFlow a static shape (fixes your error)
    img.set_shape([None, None, 3])

    # Resize image
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))

    img = tf.cast(img, tf.float32) / 255.0
    label = tf.cast(ex['label'], tf.float32)

    return img, label


# -----------------------------
# LOAD DATASET
# -----------------------------
def load_dataset(path, batch=16, shuffle=False):
    ds = tf.data.TFRecordDataset([path])
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        ds = ds.shuffle(1000)
    
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


# -----------------------------
# TRAINING LOOP
# -----------------------------
def train():
    print("📥 Loading TFRecords...")

    train_ds = load_dataset("tfdataset/train.tfrecord", shuffle=True)
    val_ds = load_dataset("tfdataset/val.tfrecord")

    print("🧠 Building model...")
    model = build_model()

    print("🚀 Starting training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5
    )

    print("💾 Saving model...")
    model.save("serving_model")

    print("🎉 Model saved to: serving_model/")


# -----------------------------
# RUN TRAINING
# -----------------------------
if __name__ == "__main__":
    train()

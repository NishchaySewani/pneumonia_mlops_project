import os
import tensorflow as tf
import glob

CLASS_MAP = {"NORMAL": 0, "PNEUMONIA": 1}

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image_string, label):
    feature = {
        'image/encoded': _bytes_feature(image_string),
        'label': _int64_feature(label)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def write_tfrecords(data_dir, out_file):
    print(f"\n🔍 Reading images from: {data_dir}")

    count = 0
    with tf.io.TFRecordWriter(out_file) as writer:
        for class_name in CLASS_MAP.keys():

            folder = os.path.join(data_dir, class_name)
            print(f"\n📁 Checking folder: {folder}")

            image_files = glob.glob(folder + "/*.jpeg") + \
                          glob.glob(folder + "/*.jpg") + \
                          glob.glob(folder + "/*.png")

            print(f"Found {len(image_files)} images in {class_name}")

            for img_path in image_files:
                with open(img_path, 'rb') as f:
                    img_bytes = f.read()

                label = CLASS_MAP[class_name]
                example = image_example(img_bytes, label)
                writer.write(example.SerializeToString())
                count += 1

                if count <= 5:
                    print(f"✔ Processed: {img_path}")

    print(f"\n✅ TFRecord saved: {out_file}")
    print(f"📦 Total images written: {count}\n")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    write_tfrecords(args.data, args.out)

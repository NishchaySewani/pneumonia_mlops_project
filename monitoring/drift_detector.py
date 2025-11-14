import numpy as np
import cv2
import glob

def compute_stats(image_paths):
    means = []
    stds = []
    for p in image_paths:
        img = cv2.imread(p)
        img = cv2.resize(img, (224,224))
        arr = img.astype('float32')/255.0
        means.append(arr.mean())
        stds.append(arr.std())
    return np.mean(means), np.mean(stds)

train_imgs = glob.glob("data/train/NORMAL/*") + glob.glob("data/train/PNEUMONIA/*")
test_imgs  = glob.glob("data/test/NORMAL/*") + glob.glob("data/test/PNEUMONIA/*")

train_mean, train_std = compute_stats(train_imgs)
test_mean, test_std = compute_stats(test_imgs)

print("Train mean/std:", train_mean, train_std)
print("Test mean/std:", test_mean, test_std)

if abs(test_mean - train_mean) > 0.05:
    print("\n⚠️ ALERT: Pixel drift detected!")
else:
    print("\n✅ No drift detected.")

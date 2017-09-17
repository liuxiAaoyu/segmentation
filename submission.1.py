## Modified Version from https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37523
import pandas as pd
import numpy as np
import cv2

## Import required info from your training script
from train_unet_general import INPUT_SHAPE, batch_size, model
from skimage.transform import resize

## Used to save time
from multiprocessing import Pool

import time
import gc

## Configure with number of CPUs you have or the number of processes to spin ##
CPUs = 48

## Tune it; used in generator
batch_size = batch_size + 6

## Mask properties
WIDTH_ORIG = 1918
HEIGHT_ORIG = 1280

## More Tuning
MASK_THRESHOLD = 0.6

## Submission data
df_test = pd.read_csv('input/sample_submission.csv')
print('sample_submission.csv shape:: ', df_test.shape)
print('sample_submission.csv columns:: ', df_test.columns.values.tolist())
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

## load it up
model = load_model('weights/best_model.hdf5')


## will be used in making submission
names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))


## https://www.kaggle.com/hackerpoet/even-faster-run-length-encoder
def run_length_encode(img):
    img = cv2.resize(img, (WIDTH_ORIG, HEIGHT_ORIG))
    flat_img = img.flatten()
    flat_img[0] = 0
    flat_img[-1] = 0
    flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix

    encoding = ''
    for idx in range(len(starts_ix)):
        encoding += '%d %d ' % (starts_ix[idx], lengths[idx])
    return encoding.strip()


rles = []

##  Split cz you can't keep all the images in memory at once
test_splits = 59  # Split test set (number of splits must be multiple of 59
ids_test_splits = np.split(ids_test, indices_or_sections=test_splits)
split_count = 0


## predict and collect rles here on splits
for ids_test_split in ids_test_splits:

    split_count += 1
    hm_samples_here = len(ids_test_split)
    
    ## generator on the small split we did earlier; batch variable used here
    def test_generator():
        while True:
            for start in range(0, len(ids_test_split), batch_size):
                x_batch = []
                end = min(start + batch_size, hm_samples_here)
                ids_test_split_batch = ids_test_split[start:end]
                for id in ids_test_split_batch.values:
                    img = cv2.imread('input/test/{}.jpg'.format(id))
                    img = cv2.resize(img, INPUT_SHAPE)
                    x_batch.append(img)
                x_batch = np.array(x_batch, np.float32) / 255
                yield x_batch

    print("Predicting on {} samples (split {}/{})".format(len(ids_test_split),
                                                          split_count, test_splits))
    ## Predictions
    preds = model.predict_generator(generator=test_generator(),
                                    steps=np.ceil(
                                        float(len(ids_test_split)) / float(batch_size)),
                                    max_queue_size=10, use_multiprocessing=True, verbose=1)

    print("Prediction of {} samples done. Now Generating RLE masks...".format(
        hm_samples_here))
    
    ## lets do rle computation in parallel
    start = time.clock()
    pool = Pool(CPUs)
    split_rle = pool.map(run_length_encode, preds)
    rles = rles + split_rle
    del split_rle
    del preds
    gc.collect()

    print(len(rles))
    pool.close()
    pool.join()
    del pool

    print(time.clock() - start)

print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submissions/submission.csv.gz', index=False, compression='gzip')







Aknowledgements

The base script uses the methods created by Sam Stainsby

https://www.kaggle.com/stainsby/fast-tested-rle/notebook

import csv
from scipy import ndimage
import numpy as np
import os
import time
from multiprocessing import Process, Queue

# Training images folder
TRAIN_IMG = "../input/train_masks"

# Time decorator
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r (%r, %r) %2.2f sec' % (method.__name__, args, kw, te-ts))
        return result
    return timed

# Create some helper functions
def get_time_left(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

# List of all images in the folder
def get_all_images(folder_path):
    # Get all the files
    features = sorted(os.listdir(folder_path))
    features_path =[]
    for iF in features:
        features_path.append(os.path.join(folder_path, iF))

    return features_path, [i_feature.split('.')[0] for i_feature in features]


list_of_images = get_all_images(TRAIN_IMG)
print(list_of_images[1][1:4])

['00087a6bd4dc_02_mask', '00087a6bd4dc_03_mask', '00087a6bd4dc_04_mask']

Next I define the actual functions that handle converting a mask to the RLE string.

def load_mask_image(mask_path):
    mask_image = ndimage.imread(mask_path, mode="L")
    return mask_image

def rle_encode(mask_image):
    pixels = mask_image.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs.tolist()

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def mask_to_row(mask_path):
    mask_name = "%s.jpg" % os.path.basename(mask_path).split('.')[0]
    return [mask_name, rle_to_string(rle_encode(load_mask_image(mask_path)))]

    Now for the part that creates the queue and workers. In essence it simply creates N workers that convert the images, while a single worker has access to the csv files and writes to it.

@timeit
def create_submission(csv_file, inference_folder, num_workers=2, image_queue=4, dry_run=False):
    # Create file and writer, if a dry run is specified, we dont write anything
    if dry_run:
        writer_fcn = lambda x: x
    else:
        open_csv = open(csv_file, 'w')
        writer = csv.writer(open_csv, delimiter=',')
        writer_fcn = lambda x: writer.writerow(x)
        
    # Write the header
    writer_fcn(["img", "rle_mask"])
    
    # Wrapper for writing
    def writer_wrap(queue):
        while True:
            # Get stuff from queue
            x = queue.get(timeout=1)
            if x is None:
                break
            writer_fcn(x)
        return

    # wrapper for creating
    def rle_wrap(queues):
        while True:
            path = queues[0].get(timeout=1)
            if path is None:
                break
            if path == -1:
                queues[1].put(None)
                break
            this_str = mask_to_row(path)
            queues[1].put(this_str)
        return

    # Define the rle queue
    rle_queue = Queue(image_queue)
    # Allow a little bit more to be passed to the writer queue
    writer_queue = Queue(image_queue*2)
    
    # Define and start our workers
    rle_workers = num_workers
    rle_consumer = [Process(target=rle_wrap, args=([rle_queue, writer_queue],)) for _ in range(rle_workers)]
    csv_worker = Process(target=writer_wrap, args=(writer_queue,))
    [_p.start() for _p in rle_consumer]
    csv_worker.start()

    # Fetch all images
    paths, names = get_all_images(inference_folder)

    # Now run through all the images
    sum_time = 0
    n_images = len(paths)
    for i, iMask in enumerate(paths):
        start_time = time.time()
        rle_queue.put(iMask)
        run_time = time.time() - start_time
        sum_time += run_time
        mean_time = sum_time / (i + 1)
        eta_time = mean_time * (n_images - i - 1)
        print("%d/%d: ETA: %s, AVE: %dms" % (i, n_images, get_time_left(eta_time), int(mean_time*1000)))
        
    # Poison pill
    for _ in range(num_workers-1):
        rle_queue.put(None)
    # Last worker will kill the writer 
    rle_queue.put(-1)
    
    # And join them
    for thread in rle_consumer:
        thread.join()
    csv_worker.join()

create_submission("", TRAIN_IMG, dry_run=True)


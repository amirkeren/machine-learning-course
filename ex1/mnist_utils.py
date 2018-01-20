import os


def prepare_mnist(data_path):
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    from mnist import MNIST

    _download_extract(data_path, 'train-labels-idx1-ubyte.gz', 28881)
    _download_extract(data_path, 'train-images-idx3-ubyte.gz', 9912422)
    _download_extract(data_path, 't10k-labels-idx1-ubyte.gz', 4542)
    _download_extract(data_path, 't10k-images-idx3-ubyte.gz', 1648877)
    print('Finished downloading and extracting MNIST, loading data...')
    mndata = MNIST(data_path)
    mndata.load_training()
    mndata.load_testing()
    train_images = [np.reshape(x, (len(mndata.train_images[0]), 1)).astype(float) / 255 for x in mndata.train_images]
    train_labels = OneHotEncoder(sparse=False).fit_transform(
        np.asarray(mndata.train_labels).reshape(len(mndata.train_labels), 1))
    train_labels = [np.reshape(x, (-1, 1)) for x in train_labels]
    test_images = [np.reshape(x, (len(mndata.test_images[0]), 1)).astype(float) / 255 for x in mndata.test_images]
    test_labels = mndata.test_labels
    print('MNIST ready')
    return [(train_images, train_labels), (test_images, test_labels)]


def _download_extract(data_path, file_name, expected_bytes):
    import shutil

    url = 'http://yann.lecun.com/exdb/mnist/' + file_name
    extract_path = os.path.join(data_path, file_name.split('.')[0])
    save_path = os.path.join(data_path, file_name)
    if os.path.exists(extract_path):
        print('Found {} Data'.format(file_name))
        return
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    _download(url, save_path)
    file_stat = os.stat(data_path + '/' + file_name)
    assert file_stat.st_size == expected_bytes, \
        '{} file is corrupted, expected size is {} but got {}. Remove the file and try again.' \
            .format(save_path, expected_bytes, file_stat.st_size)
    try:
        _ungzip(save_path, extract_path)
    except Exception as err:
        shutil.rmtree(extract_path)
        raise err


def _download(url, save_path):
    import requests
    from tqdm import tqdm

    print('downloading ' + url.split('/')[-1])
    r = requests.get(url, stream=True)
    total_length = int(r.headers.get('content-length', 0))
    with open(save_path, 'wb') as f:
        for data in tqdm(r.iter_content(1), total=total_length, unit='B', unit_scale=True):
            f.write(data)


def _ungzip(save_path, extract_path):
    import gzip

    in_f = gzip.GzipFile(save_path, 'rb')
    s = in_f.read()
    in_f.close()
    out_f = file(extract_path, 'wb')
    out_f.write(s)
    out_f.close()
    os.remove(save_path)

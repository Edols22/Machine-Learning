from semi_em import *


import gzip
def read_images(filename):
    with gzip.open(filename, 'rb') as f:
        # 跳过文件头部分
        f.read(16)

        # 读取图像信息
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        images = data.reshape(-1, 28 * 28)  # 将图像数据转换为二维形式
    
    return images

def read_labels(filename):
    with gzip.open(filename, 'rb') as f:
        # 跳过文件头部分
        f.read(8)

        # 读取标签信息
        buf = f.read()
        labels = np.frombuffer(buf, dtype=np.uint8)
    
    return labels


n_labeled_samples_list = [10,20]
n_components = 10
PCA_vector = 50
# 读取图像和标签数据
X = read_images('train-images-idx3-ubyte')
y = read_labels('train-labels-idx1-ubyte')

semi_gmm = semi_gmm_em(X,y)

for n_labeled_samples in n_labeled_samples_list:
    mean_accuracy, variance = semi_gmm.cross_validation(PCA_vector, n_labeled_samples, n_components ,semi_create = 1, semi_signal = 11)


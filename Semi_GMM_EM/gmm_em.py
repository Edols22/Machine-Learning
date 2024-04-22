# -*- coding:utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.datasets import fetch_openml
from scipy.stats import multivariate_normal
import scipy
import gzip

Different_signal = 11

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


class GaussianDistribution:
    def __init__(self, mean, covariance,epsilon = 1e-6):
        """
        初始化高斯分布对象
        
        参数：
        mean：均值，一个n维向量
        covariance：协方差矩阵，一个n x n的矩阵
        """
        self.mean = mean
        self.covariance = covariance
        self.epsilon = epsilon

    def pdf(self, x):
        """
        计算n维数据点x在该高斯分布下的概率密度函数
        
        参数：
        x：一个n维向量

        返回值：
        prob_density：x在高斯分布下的概率密度值
        """
        n = len(x)
        # 计算指数部分的值
        exponent = -0.5 * np.dot(np.dot((x - self.mean).T, scipy.linalg.pinv(self.covariance)), (x - self.mean))
        # 计算概率密度值
        det_covariance = np.linalg.det(self.covariance)
        if det_covariance <= 0:
            det_covariance = self.epsilon  # 将行列式为0的情况替换为一个较小的常数
        prob_density = (1 / ((2 * np.pi) ** (n / 2) * np.sqrt(det_covariance))) * np.exp(exponent)
        return prob_density


def m_step(x,y,l,a,miu,sigma,r):
    N = len(x)
    k = len(np.unique(y)) -1
    #m_step
    miu_new = [np.empty_like(x[0])*0.0 for i in range(k)]
    sigma_new = [np.zeros((len(x[0]), len(x[0]))) for i in range(k)]
    a_new = [0 for i in range(k)]
    
    r_new = [[0 for _ in range(k)] for __ in range(N)]
    for i in range(k):
        miu_down = l[i]
        for j in range(N):
            if y[j] == Different_signal:
                miu_down += r[j][i]
                miu_new[i] += r[j][i] * x[j]
                #sigma_new[i] += r[j][i] * np.outer((x[j]-miu[i]),((x[j] - miu[i]).T))

            elif y[j] == i:
                miu_new[y[j]] += x[j]
                #sigma_new[y[j]] += np.outer((x[j]-miu[y[j]]),((x[j] - miu[y[j]]).T))
        miu_new[i] /= miu_down

        for j in range(N):
            if y[j] == Different_signal:
                sigma_new[i] += r[j][i] * np.outer((x[j] - miu_new[i]),((x[j] - miu_new[i]).T))
            elif y[j] == i:
                sigma_new[y[j]] += np.outer((x[j]-miu_new[y[j]]),((x[j] - miu_new[y[j]]).T))

        sigma_new[i] /= miu_down
        a_new[i] = miu_down / N

    for i in range(k):
        sigma_new[i] = np.diag(np.diag(sigma_new[i]))
    guassion_new = [multivariate_normal(mean=miu_new[_], cov=sigma_new[_]) for _ in range(k)]
    #guassion_new = [GaussianDistribution(mean=miu_new[_], covariance=sigma_new[_]) for _ in range(k)]
    for j in range(N):
        sum_pdf_new = 0
        for i in range(k):
            sum_pdf_new += a_new[i] * guassion_new[i].pdf(x[j])
        for i in range(k):
            r_new[j][i] = a_new[i] * guassion_new[i].pdf(x[j]) / sum_pdf_new
    return a_new,miu_new,sigma_new,r_new




def em_gmm(x,y):
    N = len(x)
    k = len(np.unique(y)) -1
    #e_step
    a = [0 for i in range(k)]
    l = [0 for i in range(k)]
    miu = [np.empty_like(x[0])*0 for i in range(k)] 
 
    M = 0
    for j in range(N):
        if y[j] != Different_signal:
            l[y[j]] += 1
            miu[y[j]] += x[j]
            M += 1


    for i in range(k):
        miu[i] /= l[i]
 
    for i in range(k):
        a[i] = l[i] / M

    sigma = [np.zeros((len(x[0]), len(x[0]))) for i in range(k)]

    for i in range(N):
        if y[i] != Different_signal:
            z = y[i]
            sigma[z] += np.outer((x[i] - miu[z]),(x[i] - miu[z]).T) / l[z]
            
    for i in range(k):
        sigma[i] = np.diag(np.diag(sigma[i]))

    guassion = [multivariate_normal(mean=miu[i], cov=sigma[i]) for i in range(k)]   
    #guassion = [GaussianDistribution(mean=miu[i], covariance=sigma[i]) for i in range(k)]
    r = [[0 for i in range(k)] for _ in range(N)]
    for j in range(N):
        sum_pdf = 0
        for i in range(k):
            sum_pdf += a[i] * guassion[i].pdf(x[j])
        for i in range(k):
            r[j][i] = a[i] * guassion[i].pdf(x[j]) / sum_pdf
    #m_step
    a_new,miu_new,sigma_new,r_new = m_step(x,y,l,a,miu,sigma,r)
    # for i in range(40):
    #     a_new,miu_new,sigma_new,r_new = m_step(x,y,l,a_new,miu_new,sigma_new,r_new)
    return a_new,miu_new,sigma_new,r_new


class GaussianMixtureModel:
    def __init__(self, means, covariances, weights):
        self.means = means
        self.covariances = covariances
        self.weights = weights

    def posterior_probability(self, x):
        # 计算每个类别的后验概率
        posteriors = []
        for mean, covariance, weight in zip(self.means, self.covariances, self.weights):
            # 计算每个高斯分布的概率密度
            gmm_temp = multivariate_normal(mean=mean, cov=covariance)
            probability = gmm_temp.pdf(x)
            # 将概率乘以权重得到后验概率
            posteriors.append(probability * weight)
        # 将后验概率归一化，以确保它们的总和为1
        posteriors = np.array(posteriors) / np.sum(posteriors)
        return posteriors

    def predict(self, x):
        # 预测分类
        y = []
        for each in x:
            posteriors = self.posterior_probability(each)
            # 选择具有最大后验概率的类别作为预测结果
            predicted_class = np.argmax(posteriors)
            y.append(predicted_class)
        return y



def cross_validation(X, y, k, n_labeled_samples, n_components, n_splits=5, n_repeats=5):
    skf = StratifiedKFold(n_splits=n_repeats, shuffle=True, random_state=42)
    scores = []
    i = 0
    times = 0
    
    pca = PCA(n_components=k)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # 在训练集中随机选择有标签数据
        labeled_indices = []
        for label in np.unique(y_train):
            indices = np.where(y_train == label)[0]
            labeled_indices += np.random.choice(indices, size=n_labeled_samples, replace=False).tolist()

        # 获取有标签数据和无标签数据
        X_labeled = X_train[labeled_indices]
        y_labeled = y_train[labeled_indices]

        before = []
        for i in range(5):
            before.append(y_labeled[i])

        for i in range(5):
            y_labeled[i] = Different_signal


        # 执行EM+Semi-GMM算法
        X_labeled = pca.fit_transform(X_labeled)
        N = len(X_labeled)
        a,miu,sigma,r = em_gmm(X_labeled,y_labeled)
        y_pred = []
        for i in range(5):
            y_labeled[i] = before[i]
        for i in range(N):
            y_pred.append(np.argmax(r[i]))
        accuracy = accuracy_score(y_labeled, y_pred)
        print(f'训练集：{accuracy}')

        # 在测试集上进行预测并计算准确率
        gmm = GaussianMixtureModel(miu,sigma,a) 


        X_test_reduced = pca.transform(X_test)
        y_pred = gmm.predict(X_test_reduced)

        accuracy = accuracy_score(y_test, y_pred)
        scores.append(accuracy)
        times +=1
        print(f'第{times}次：{accuracy}')

    scores = np.array(scores)
    mean_scores = scores.mean()
    variance_scores = scores.var()

    return mean_scores, variance_scores

n_labeled_samples_list = [10,20]
n_components = 10
PCA_vector = 50
# 读取图像和标签数据
X = read_images('train-images-idx3-ubyte')
y = read_labels('train-labels-idx1-ubyte')


for n_labeled_samples in n_labeled_samples_list:
    mean_accuracy, variance = cross_validation(X, y, PCA_vector, n_labeled_samples, n_components)
    print(f"Number of labeled samples: {n_labeled_samples}")
    print(f"Mean accuracy: {mean_accuracy}")
    print(f"Variance: {variance}")
    print("-----------------------------")

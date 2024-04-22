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

# 
#  函数名: GaussianDistribution
#  描述  : 对象GaussianDistribution,用于计算给定多个n维高斯函数下，x值的预测函数
#  输入  : 初始化为高斯函数参数，计算输入数据
#  输出  : 数据对应的预测类型
#  备注  : 对于非正定对称的协方差矩阵貌似有用
#  
class GaussianDistribution:

# 
#  函数名: /
#  描述  : 初始化高斯分布对象
#  输入  :         
#         mean：均值，一个n维向量
#         covariance：协方差矩阵，一个n x n的矩阵
#  输出  : /
#  备注  : 
#  
    def __init__(self, mean, covariance,epsilon = 1e-6):

        self.mean = mean
        self.covariance = covariance
        self.epsilon = epsilon
# 
#  函数名: GaussianDistribution.pdf
#  描述  : 计算n维数据点x在该高斯分布下的概率密度函数
#  输入  : x：一个n维向量
#  输出  : prob_density：x在高斯分布下的概率密度值
#  备注  : 奇异值分解 + 小数替代
#  
    def pdf(self, x):
        n = len(x)
        # 计算指数部分的值
        exponent = -0.5 * np.dot(np.dot((x - self.mean).T, scipy.linalg.pinv(self.covariance)), (x - self.mean))
        # 计算概率密度值
        det_covariance = np.linalg.det(self.covariance)
        if det_covariance <= 0:
            det_covariance = self.epsilon  # 将行列式为0的情况替换为一个较小的常数
        prob_density = (1 / ((2 * np.pi) ** (n / 2) * np.sqrt(det_covariance))) * np.exp(exponent)
        return prob_density

# 
#  函数名: GaussianDistribution
#  描述  : 对象GaussianDistribution,用于计算给定多个n维高斯函数下，x值的预测函数
#  输入  : 初始化为高斯函数参数，计算输入数据
#  输出  : 数据对应的预测类型
#  备注  : 只对于对称正定矩阵有用
#  
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


# 
#  函数名: semi_gmm_em
#  描述  : 半监督GMM+EM
#  输入  : /
#  输出  : /
#  备注  : 五次五折校验 + 自设定半监督
#  
class semi_gmm_em:
# 
#  函数名: /
#  描述  : 初始化
#  输入  : 数据,标签,是否开启半监督
#  输出  : /
#  备注  : /
#  
    def __init__(self,x,y,semi_on = True):
        self.x = x
        self.y = y
        self.semi_signal = None
        self.semi_on = semi_on
        
# 
#  函数名: m_step
#  描述  : 更新参数
#  输入  : 数据，标签，类别个数，模型权重，初始均值，协方差矩阵（列表），数据权重
#  输出  : 更新后的模型权重，初始均值，协方差矩阵（列表），数据权重
#  备注  : /
#  
    def m_step(self,x,y,l,a,miu,sigma,r):
        N = len(x)
        k = len(np.unique(y)) -1
        if (self.semi_on == False):
            k = k + 1 
        #m_step
        miu_new = [np.empty_like(x[0])*0.0 for i in range(k)]
        sigma_new = [np.zeros((len(x[0]), len(x[0]))) for i in range(k)]
        a_new = [0 for i in range(k)]
        
        r_new = [[0 for _ in range(k)] for __ in range(N)]
        for i in range(k):
            miu_down = l[i]
            for j in range(N):
                if y[j] == self.semi_signal:
                    miu_down += r[j][i]
                    miu_new[i] += r[j][i] * x[j]
                    #sigma_new[i] += r[j][i] * np.outer((x[j]-miu[i]),((x[j] - miu[i]).T))

                elif y[j] == i:
                    miu_new[y[j]] += x[j]
                    #sigma_new[y[j]] += np.outer((x[j]-miu[y[j]]),((x[j] - miu[y[j]]).T))
            miu_new[i] /= miu_down

            for j in range(N):
                if y[j] == self.semi_signal:
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



# 
#  函数名: em_gmm
#  描述  : 初始化参数并进行更新
#  输入  : 数据，标签
#  输出  : 更新后的模型权重，初始均值，协方差矩阵（列表），数据权重
#  备注  : 只进行一次闭式解
#  
    def em_gmm(self,x,y):
        N = len(x)
        k = len(np.unique(y)) -1
        #e_step
        a = [0 for i in range(k)]
        l = [0 for i in range(k)]
        miu = [np.empty_like(x[0])*0.0 for i in range(k)] 
        M = 0
        for j in range(N):
            if y[j] != self.semi_signal:
                l[y[j]] += 1
                miu[y[j]] += x[j]
                M += 1


        for i in range(k):
            miu[i] /= l[i]
    
        for i in range(k):
            a[i] = l[i] / M

        sigma = [np.zeros((len(x[0]), len(x[0]))) for i in range(k)]

        for i in range(N):
            if y[i] != self.semi_signal:
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
        a_new,miu_new,sigma_new,r_new = self.m_step(x,y,l,a,miu,sigma,r)
        # for i in range(40):
        #     a_new,miu_new,sigma_new,r_new = m_step(x,y,l,a_new,miu_new,sigma_new,r_new)
        return a_new,miu_new,sigma_new,r_new

# 
#  函数  : em_gmm
#  描述  : 五次五折交叉验证
#  输入  : PCA维数，每种标签个数，五折，五次，semi_create：是否创造无标签数据，无标签数据的标签设定为semi_signal
#  输出  : 每折的准确率和均值方差
#  备注  : 若创造无标签数据，则必须给出无标签数据的设定标签，不能与原有标签重复，且创造的无标签数据占总数据1/3
#  
    def cross_validation(self, k, n_labeled_samples, n_splits=5, n_repeats=5, semi_create = False, semi_signal = None):
        skf = StratifiedKFold(n_splits=n_repeats, shuffle=True, random_state=42)
        scores = []
        i = 0
        times = 0
        if(semi_signal != None):
            self.semi_signal = semi_signal
        else:
            if (semi_create == True):
                print("False, please give the semi_signal")
        
        pca = PCA(n_components=k)
        for train_index, test_index in skf.split(self.x, self.y):
            X_train, X_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            # 在训练集中随机选择有标签数据
            labeled_indices = []
            for label in np.unique(y_train):
                indices = np.where(y_train == label)[0]
                labeled_indices += np.random.choice(indices, size=n_labeled_samples, replace=False).tolist()

            # 获取有标签数据和无标签数据
            X_labeled = X_train[labeled_indices]
            y_labeled = y_train[labeled_indices]

            before = []
            nolabel_indices = []
            if (semi_create  == 1):

                for label in np.unique(y_labeled):
                    indices = np.where(y_labeled == label)[0]
                    nolabel_indices += np.random.choice(indices, size=n_labeled_samples // 3, replace=False).tolist()


                for j in nolabel_indices:
                    before.append(y_labeled[j])

                for j in nolabel_indices:
                    y_labeled[j] = self.semi_signal



            # 执行EM+Semi-GMM算法
            X_labeled = pca.fit_transform(X_labeled)
            N = len(X_labeled)
            a,miu,sigma,r = self.em_gmm(X_labeled,y_labeled)
            y_pred = []

            if (semi_create  == 1):
                for j in range(len(nolabel_indices)):
                    y_labeled[nolabel_indices[j]] = before[j]

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
            print(f'测试集第{times}次：{accuracy}')

        scores = np.array(scores)
        mean_scores = scores.mean()
        variance_scores = scores.var()
        print(f"Number of labeled samples: {n_labeled_samples}")
        print(f"Mean accuracy: {mean_scores}")
        print(f"Variance: {variance_scores}")
        print("-----------------------------")
        return mean_scores, variance_scores

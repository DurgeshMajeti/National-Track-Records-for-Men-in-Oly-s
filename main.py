import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import library as lib
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

##Reading Data
mvd = pd.read_csv("data.csv",index_col=None)
data = np.matrix(mvd.values)
data = data.T

##Checking normality
cov = np.matrix(mvd.cov())
cov -= cov%0.001
normal_corr = []
fig,axs = plt.subplots(3,3)
normal_corr.append(lib.qq_plot(data[0],axs[0,0],'100m Q-Q plot'))
normal_corr.append(lib.qq_plot(data[1],axs[0,1],'200m Q-Q plot'))
normal_corr.append(lib.qq_plot(data[2],axs[0,2],'400m Q-Q plot'))
normal_corr.append(lib.qq_plot(data[3],axs[1,0],'800m Q-Q plot'))
normal_corr.append(lib.qq_plot(data[4],axs[1,2],'1500m Q-Q plot'))
normal_corr.append(lib.qq_plot(data[5],axs[2,0],'5000m Q-Q plot'))
normal_corr.append(lib.qq_plot(data[6],axs[2,1],'10000m Q-Q plot'))
normal_corr.append(lib.qq_plot(data[7],axs[2,2],'Marathon Q-Q plot'))
outliers = lib.chi2_plot(data[:8],cov,axs[1,1],'Chi Square Plot')
print("Outlier countries : ",data[8,outliers[0]],data[8,outliers[1]],data[8,outliers[2]])
plt.tight_layout()
plt.show()
print("Correlation coefficients with Q:",normal_corr,"\n")

##Obtainthe sample correlation matrix R for given data, and determine its eigen values and eigenvectors
R = lib.corr_matrix(data[:8],data[:8])
print("Correlation Matrix : ",R,'\n')
eigvals,eigvects = lib.eig_val(R)
print("Eigen values and Eigen vectors of R : ")
print(eigvals)
print(eigvects)

##Determine the the number k of principal components for the standardized variables. Obtain the k components
fig,axs = plt.subplots(1,1)
k_comp = lib.scree(eigvals,axs)
plt.show()
print("\nNumber of principle components are : ",k_comp,"\n")

avg = lib.mean(data[:8])
norm_data = lib.standardize(data[:8],avg,cov)
p_data = lib.gen_pcs(norm_data[:8],eigvects[:k_comp])
print("P Data : ",p_data)

##Prepare a table showing the correlations of the standardized variables with the k principal components
pca_x_cor = lib.corr_matrix(p_data,norm_data)
print("Correlation between PCA and Standardized data : \n",pca_x_cor,"\n")

##Rank the nations based on their score on the fist principal component. Find India’s rank
pca1 = np.array(p_data[:,0])
contry = np.array(data[8].T)
pca1_contry = {}
for i in range(len(pca1)):
    pca1_contry[pca1[i,0]] = contry[i,0]
pca1 = np.sort(pca1[:,0])
for i in range(len(pca1)):
    print(i+1,":",pca1_contry[pca1[i]])

##Perform graphical analysis on k principal components
fig,axs = plt.subplots(2,2)
axs[0,0].scatter(np.array(p_data.T[0]),np.array(p_data.T[1]))
axs[0,0].set_title("Scatter Plot")
pca1_q_cor = lib.qq_plot(p_data[:,0].T,axs[0,1],"PCA 1 Q-Q Plot")
pca2_q_cor = lib.qq_plot(p_data[:,1].T,axs[1,0],"PCA 2 Q-Q Plot")
pca_avg = lib.mean(p_data)
pca_cov = lib.cov(p_data,pca_avg)
pca_outliers = lib.chi2_plot(p_data.T,pca_cov,axs[1,1],'Chi Square Plot')
print("Correlation between PCA and Quantiles : ",pca1_q_cor,pca2_q_cor)
print("Outlier countries : ",data[8,outliers[0]],data[8,outliers[1]])
plt.show()

#Find the multilinear regression model for the first seven varibles x 1 , . . . , x 7 (independent) and x 8 as response varibale

reg_x = np.matrix((np.append(np.array(data[:7]),np.ones(55)).reshape(8,55)).T)
reg_y = np.matrix(np.array(data[7]).reshape(55))
xtx = reg_x.T@reg_x
xtxi = (np.matrix(xtx,dtype=float)).I
xty = reg_x.T*reg_y.T
b = xtxi*xty
b -= b%0.01
b = np.array(b)
print("Y =",b[0],"x1+",b[1],"x2+",b[2],"x3+",b[3],"x4+",b[4],"x5+",b[5],"x6+",b[6],"x7+",b[7])

#data convert
data[0] = 100/data[0]
data[1] = 200/data[1]
data[2] = 400/data[2]
data[3] = 800/((data[3]/1)*60+(data[3]%1))
data[4] = 1500/((data[4]/1)*60+(data[4]%1))
data[5] = 5000/((data[5]/1)*60+(data[5]%1))
data[6] = 10000/((data[6]/1)*60+(data[6]%1))
data[7] = 42000/((data[7]/1)*60+(data[7]%1))
data[:8] -= data[:8]%0.001
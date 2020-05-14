import numpy as np
import scipy.stats as stats

def mean(data):
    mean = []
    for i in range(len(data)):
        mean.append(round(data[i].mean(),2))
    return mean

def correlation(x,y):
    x=np.array(x)
    if (x.shape[0] == 1):
        x = x.T
    y=np.array(y)
    if(y.shape[0] == 1):
        y = y.T
    X=np.stack((x,y))
    mu = mean(X)
    cov = round(np.sum((x-mu[0])*(y-mu[1]))/(len(x)-1),3)
    x_var = round(np.sum((x-mu[0])*(x-mu[0]))/(len(x)-1),3)
    y_var = round(np.sum((y-mu[1])*(y-mu[1]))/(len(x)-1),3)
    cor = cov/(np.sqrt(x_var*y_var))
    cor -= cor%0.001
    return(cor)

def qq_plot(data,ax,title):
    x = np.copy(data)
    x = np.sort(x[0])
    q = np.zeros(len(x))
    for i in range(len(x)):
        q[i] = (stats.norm.ppf((i+0.5)/len(x)))
    ax.scatter(q,x,s=5,color='green')
    ax.set_xlabel("Theoratical Quantiles",size=8)
    ax.set_ylabel("Experimental Quantiles",size=8)
    ax.set_title(title)
    return(correlation(x,q))

def statistical_dist(data,cov,avg):
    dat = data.T
    dist = np.zeros(len(dat))
    for i in range(len(dat)):
        dist[i] = float((dat[i]-avg).T@cov.I@(dat[i]-avg))
    dist -= dist%0.001
    return(dist)

def chi2_plot(data,cov,ax,title):
    matrix = np.copy(data)
    avg = mean(data)
    dist_unsort = statistical_dist(matrix,cov,avg)
    dist = np.sort(dist_unsort)
    q = np.zeros(len(dist))
    for i in range(len(q)):
        q[i] = stats.chi2.ppf((i+0.5)/len(q),8)
    ax.scatter(q,dist,s=5,color='orchid')
    ax.set_title(title)
    dist_unsort = list(dist_unsort)
    return(dist_unsort.index(dist[-1]),dist_unsort.index(dist[-2]),dist_unsort.index(dist[-3]))

def eig_val(matrix):
    eigvals,vects = np.linalg.eig(matrix)
    eigvals -= eigvals%0.001
    vects -= vects % 0.001
    return(eigvals,vects.T)

def corr_matrix(data1,data2):
    if(data1.shape[0] != data2.shape[0]):
        data1 = data1.T
    data2 = np.matrix(data2)
    cor = np.matrix(np.ndarray((len(data1),len(data2))))
    for i in range(len(data1)):
        for j in range(len(data2)):
            cor[i,j] = correlation(data1[i],data2[j])
    return(cor)

def scree(data,ax):
    sortd = np.sort(data)
    sotd = []
    for i in range(len(sortd)-1,-1,-1):
        sotd.append(sortd[i])
    y = [i+1 for i in range(len(data))]
    ax.scatter(y,sotd)
    ax.plot(y,sotd)
    data_sum = np.sum(data)
    accept = 0
    i = sotd[0]
    while((i/data_sum)<0.9):
        accept+=1
        i = i+sotd[accept]
    return(accept+1)

def gen_pcs(data,eigv):
    dat = np.matrix(data.T)
    eig_v = np.matrix(eigv)
    p_data = np.matrix(np.ndarray((len(dat),len(eig_v))))
    for i in range(len(eig_v)):
        p_data[:,i] = dat*eig_v[i].T
    return (p_data)

def standardize(data,avg,cov):
    stand = np.copy(data)
    avg = np.array(avg)
    for i in range(len(data)):
        for j in range(data.shape[1]):
            stand[i,j] = round((data[i,j]-avg[i])/np.sqrt(cov[i,i]),3)
    return(stand)

def cov(data,mu):#######returns covariance matrix
    x = np.array(data[:,0])
    y = np.array(data[:,1])
    d1 = x-mu[0]
    d2 = y-mu[1]
    s11 = np.sum(d1**2)/41
    s22 = np.sum(d2**2)/41
    s12 = np.sum(d1*d2)/41
    return(np.matrix([[s11,s12],[s12,s22]]))
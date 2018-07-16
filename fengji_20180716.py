
# coding: utf-8

# In[162]:


import pandas as pd

#添加转化率特征
all=pd.read_csv("train_labels.csv")
all.columns=["fengjiid","haohuai"]
print all.fengjiid[0]

for j in [0]:
    print '表%s'%j
    fengji=pd.read_csv('train\\'+all.fengjiid[j])
    #print fengji.head(4)
    del fengji['采样序号']
    #print fengji.head(4)
    fengji_median=pd.DataFrame(fengji.median()).T
    
    #print fengji_mean.columns
    for lieming in fengji_median.columns:
        fengji_median.rename(columns={lieming:(lieming+'_median')}, inplace = True)
    #print fengji_mean.columns
    #print type(all.fengjiid[j])
    #fengji_mean[u'编号']=all.fengjiid[j]
    
    fengji_median.insert(0,'fengjiid',all.fengjiid[j])
    
    fengji_quantile25=pd.DataFrame(fengji.quantile(.25)).T
    fengji_quantile75=pd.DataFrame(fengji.quantile(.75)).T
    for lieming in fengji_quantile25.columns:
        fengji_quantile25.rename(columns={lieming:(lieming+'_quantile25')}, inplace = True)
    for lieming in fengji_quantile75.columns:
        fengji_quantile75.rename(columns={lieming:(lieming+'_quantile75')}, inplace = True)
    fengji_quantile25.insert(0,'fengjiid',all.fengjiid[j])
    fengji_quantile75.insert(0,'fengjiid',all.fengjiid[j])
    all_fengji = fengji_median.merge((fengji_quantile25), on=['fengjiid'], how="left")
    all_fengji = all_fengji.merge((fengji_quantile75), on=['fengjiid'], how="left")

for j in range(1,10000):
    print '表%s'%j
    fengji=pd.read_csv('train\\'+all.fengjiid[j])
    #print fengji.head(4)
    del fengji['采样序号']
    #print fengji.head(4)
    if fengji.empty:
        print '表%s为空'%j
    else:
        fengji_median=pd.DataFrame(fengji.median()).T
        fengji_quantile25=pd.DataFrame(fengji.quantile(.25)).T
        fengji_quantile75=pd.DataFrame(fengji.quantile(.75)).T
        for lieming in fengji_median.columns:
            fengji_median.rename(columns={lieming:(lieming+'_median')}, inplace = True)
        for lieming in fengji_quantile25.columns:
            fengji_quantile25.rename(columns={lieming:(lieming+'_quantile25')}, inplace = True)
        for lieming in fengji_quantile75.columns:
            fengji_quantile75.rename(columns={lieming:(lieming+'_quantile75')}, inplace = True)
        fengji_quantile25.insert(0,'fengjiid',all.fengjiid[j])
        fengji_quantile75.insert(0,'fengjiid',all.fengjiid[j])
        fengji_median.insert(0,'fengjiid',all.fengjiid[j])
        all_fengji_demo = fengji_median.merge(fengji_quantile25, on=['fengjiid'], how="left")
        all_fengji_demo = all_fengji_demo.merge(fengji_quantile75, on=['fengjiid'], how="left")
        all_fengji=all_fengji.append(all_fengji_demo)
    
# all = all.merge(fengji_mean, on=['fengjiid'], how="left")
# print all.head(2)
all_fengji.to_csv("all_result_v1_0715.csv",index=None)


# In[11]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data=pd.read_csv("all_result_v2_0715.csv")

x_prime = data.iloc[0:,1:len(data.columns)-1] #去掉y列，所以len-1
y = data.iloc[0:,len(data.columns)-1:]
x_prime_train, x_prime_test, y_train, y_test = train_test_split(x_prime, y, train_size=0.7, random_state=0)
feature_pairs=[]

for p in range(len(data.columns)-1):
    # 准备数据
    x_train = x_prime_train.iloc[0:,range(p)+range(p+1,len(data.columns)-2)]#去掉y列之外再去掉一列特征，所以len-2
    x_test = x_prime_test.iloc[0:,range(p)+range(p+1,len(data.columns)-2)]

    # 决策树学习
    model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)
    model.fit(x_train, y_train)

    # 训练集上的预测结果
    y_train_pred = model.predict(x_train)
    acc_train = accuracy_score(y_train, y_train_pred)
    y_test_pred = model.predict(x_test)
    acc_test = accuracy_score(y_test, y_test_pred)
    if acc_train > 0.85 and acc_test > 0.7:
        print '不要特征：', data.columns[p]
        print '\t训练集准确率: %.4f%%' % (100*acc_train)
        print '\t测试集准确率: %.4f%%\n' % (100*acc_test)


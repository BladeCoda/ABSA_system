# -*- coding: utf-8 -*-
import numpy as np
import contextProcessing
import word2vecProcessing
from sklearn.metrics import classification_report

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

def getFeaturesFromContext(aspectContext,W2V):
    w2v_feature=[]
    for word in aspectContext.context.split(' '):
        try:
            w2v_feature.append(W2V[word.lower()])
        except:
            w2v_feature.append([0 for i in range(W2V.vector_size)])
            #print('not find :%s'%word.lower())
    w2v_feature=np.array(w2v_feature).mean(axis=0)
    
    dep_feature=[]
    for dep in aspectContext.dep_context:
        for word in dep:
            try:
                dep_feature.append(W2V[word.lower()])
            except:
                dep_feature.append([0 for i in range(W2V.vector_size)])
                #print('not find :%s'%word.lower())
    dep_feature=np.array(dep_feature).mean(axis=0)
    
    return np.concatenate((w2v_feature,dep_feature)).tolist()
    
def getInfoFromList(aspectContextList,W2V):
    features=[getFeaturesFromContext(ac,W2V) for ac in aspectContextList]
    pols=[ac.pol for ac in aspectContextList]
    return features,pols
    
def getFeaturesAndPolsFromFile(filepath,d_type='re',per=0.8):
    if d_type=='re':
        d_name='Restaurants'
    else:
        d_name='LapTops'
    train_data,test_data=contextProcessing.splitContextFile(filepath,per)
    print('获取特征与情感分类中。。。。。')
    W2V=word2vecProcessing.loadForWord('model/%s.w2v'%d_name)
    
    trainX,trainY=getInfoFromList(train_data,W2V)
    testX,testY=getInfoFromList(test_data,W2V)
    print('获取完成')
    
    trainX=np.array(trainX)
    trainY=np.array(trainY)
    testX=np.array(testX)
    testY=np.array(testY)
    
    return trainX,trainY,testX,testY
    
def trainClassifier(trainX,trainY,classifier='SVM'):  
    if classifier=='SVM':
        print('使用SVM进行分类器训练')
        clf=LinearSVC()
        clf=clf.fit(trainX,trainY)
        print('训练完成')
    else:
        print('使用逻辑斯蒂')
        lr=LogisticRegression()
        lr=lr.fit(trainX,trainY)#用训练数据来拟合模型
        print('训练完成')
        clf=lr  
        
    return clf
    
def predict(testX,testY,clf):
    print('开始预测')
    true_result=clf.predict(testX)
    pre_result=testY
    
    print('分类报告: \n')
    print(classification_report(true_result, pre_result,digits=4))
    
    clf.score(testX,testY)
    
def examByML(d_type='re',classifier='SVM',per=0.8):
    if d_type=='re':
        filepath='contextFiles/re_train.cox'
    else:
        filepath='contextFiles/lp_train.cox'
        
    trainX,trainY,testX,testY=getFeaturesAndPolsFromFile(filepath,d_type,per)
    clf=trainClassifier(trainX,trainY,classifier)
    predict(testX,testY,clf)
    
if __name__=='__main__':
    examByML('re','SVM',0.8)
    
    
    
    
        
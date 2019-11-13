# -*- coding: utf-8 -*-
from preProcessing import *
from word2vecProcessing import trainingEmbedding, createCluster
from exam_CRF import evaluate
from contextProcessing import createAllForPol
from polClassification_ML import examByML

if __name__ == '__main__':
    print('#### Step1. Preprocessing')
    # 对原始数据做预处理
    #transformJSONFiles(d_type='re')
    #transformJSONFiles(d_type='lp')
    #createAllDependence()
    #depend_list=loadDependenceInformation('dependences/re_train.dep')
    
    print('#### Step1. Word2Vec and Clutering')
    # 训练词向量与聚类
    '''trainingEmbedding(300,'re',True)
    createCluster(100,'re')
    trainingEmbedding(300,'lp',True)
    createCluster(200,'lp')'''
    
    print('#### Step3 evaluate for Aspect Term Extraction')
    #all_terms,all_offsets,origin_text_test,true_offsets=evaluate(False,'re')
    #all_terms,all_offsets,origin_text_test,true_offsets=evaluate(False,'lp')
    
    print('#### Step4 context Processing')
    '''createAllForPol(d_type='re',context=5)
    createAllForPol(d_type='lp',context=5)'''
    
    print('#### Step5 evaluate for Aspect Term Classification')
    #examByML('re','SVM',0.8)
    examByML('lp','SVM',0.8)
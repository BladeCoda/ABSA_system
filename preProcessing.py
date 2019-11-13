# -*- coding: utf-8 -*-
import pandas
import json
import entity
import xml.etree.ElementTree as ET
import sys
import nltk
from nltk.stem import SnowballStemmer
from nltk.parse.stanford import StanfordDependencyParser
import nltk.data  
import pickle

wn=nltk.WordNetLemmatizer()
sp=SnowballStemmer('english')

#验证文件是不是一个符合规范的XML,返回所有实例与所有的AspectCategory
def validateXML(filename):
    #解析XML，找到所有的句子
    elements=ET.parse(filename).getroot().findall('sentence')
    aspects=[]
    #遍历所有的句子
    for e in elements:
        #获取每个实例中所有的aspectTerm
        for ats in e.findall('aspectTerms'):
            if ats is not None:
                for a in ats.findall('aspectTerm'):
                    aspects.append(entity.AspectTerm('','',[]).create(a).term)
    return elements,aspects
    
#从文件中加载语料实体
def loadXML(filename):
    try:
        elements,aspects=validateXML(filename)
        print('XML合法，一共有%d个句子，%d个AspectTerm,%d个不同的AspectTerm'
              %(len(elements),len(aspects),len(list(set(aspects)))))
    except:
        print("XML不合法:", sys.exc_info()[0])#不规范的文件
        raise
    
    corpus=entity.Corpus(elements)
    return corpus  
  
#获取BIO的全部信息
def createBIOClass(instances,d_type):
    bio_entity=entity.BIO_Entity(instances,d_type)
    bio_entity.createBIOTags()
    bio_entity.createPOSTags()
    bio_entity.createLemm()
    
    bio_entity.createW2VCluster()
    
    return bio_entity
    
def cutFileForBIO(filename,threshold=0.8,shuffle=False,d_type='re'):
    corpus=loadXML(filename)
    train_corpus,test_corpus=corpus.split(threshold, shuffle)
    print('------切割原始数据集完成，开始构造特征------')
    
    bio_train=createBIOClass(train_corpus,d_type)
    print('-----训练BIO构造完成，开始构造测试BIO-------')
    bio_test=createBIOClass(test_corpus,d_type)
    print('测试BIO构造完成')
    
    print('-----获取训练集的特征和标记-------')
    train_X,train_Y=bio_train.getFeaturesAndLabels()
    print('-----获取测试集的特征和标记-------')
    test_X,test_Y=bio_test.getFeaturesAndLabels()
    
    true_offset=[]
    
    for i in range(len(bio_test.instances)):
        offset=[a.offset for a in bio_test.instances[i].aspect_terms]
        true_offset.append(offset)
        
    origin_text=bio_test.texts
    
    return train_X,train_Y,test_X,test_Y,true_offset,origin_text
    
#读取额外的数据并转化为CSV文件
def transformJSONFiles(d_type='re',all_text=False,text_num=200000):
    if d_type=='re':
        filepath='data/extra/yelp/yelp_academic_dataset_review.json'
        outpath='data/extra/yelp/Restaurants_Raw.csv'
        text_item='text'
    else:
        filepath='data/extra/amzon/Electronics_5.json'
        outpath='data/extra/amzon/LapTops_Raw.csv'
        text_item='reviewText'
        
    print('开始加载JSON并获取其文本.....')

    review_list=[]
    if d_type=='re':
        with open(filepath,'r') as f:
            items_list=f.readlines()
            #是否获取所有的文本
            if all_text==False:
                items_list=items_list[:text_num]
            #一个个解析
            for item in items_list:
                json_dict=json.loads(item)
                review_list.append(' '.join(nltk.word_tokenize(json_dict[text_item])))
    else:
        count = 0
        with open(filepath,'r') as f:
            items_list=f.readlines()
            #一个个解析
            for item in items_list:
                json_dict=json.loads(item)
                words=nltk.word_tokenize(json_dict[text_item])
                words1=[word.lower() for word in words]
                if 'notebook' in words1 or 'laptop' in words1:
                    review_list.append(' '.join(words))
                    count += 1
                    if all_text==False and count > text_num:
                        break
            
    print('评价文本加载完毕，转化为CSV文件')
    output=pandas.DataFrame({'text':review_list})
    output.to_csv(outpath,index=False)
    print('转化完成！！')
    
def createDependenceInformation(inputfile,outputfile):
    corpus=loadXML(inputfile)
    texts=corpus.texts
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sent_num=[]
    all_sents=[]
    print('分句开始')
    for text in texts:
        #对文本分句
        sents=tokenizer.tokenize(text)  
        sents=[nltk.word_tokenize(sent) for sent in sents]
        all_sents.extend(sents)
        sent_num.append(len(sents))
    print('解析开始')
    eng_parser = StanfordDependencyParser(r"stanford parser/stanford-parser.jar",
                                          r"stanford parser/stanford-parser-3.6.0-models.jar")
    res_list=list(eng_parser.parse_sents(all_sents))
    res_list=[list(i) for i in res_list]
    depends=[]
    #遍历每组关联
    for item in res_list:
        depend=[]
        for row in item[0].triples():
            depend.append(row)
        depends.append(depend)
    print('解析完成,开始切分')
    index=0
    depend_list=[]
    for num in sent_num:
       depend_list.append(depends[index:index+num])
       index+=num
       
    print('切分完成，开始保存')
    with open(outputfile,'wb') as f:
        pickle.dump(depend_list,f)
    print('完成。')
        
def loadDependenceInformation(filepath):
    print('载入依赖关系列表：%s'%filepath)
    with open(filepath,'rb') as f:
        depend_list=pickle.load(f)
    return depend_list
    
def createAllDependence():
    print('获取餐厅数据的句法依赖')
    createDependenceInformation('data/origin/Restaurants_Train_v2.xml','dependences/re_train.dep')
    createDependenceInformation('data/origin/Restaurants_Test_Data_phaseB.xml','dependences/re_test.dep')
    
    print('获取笔记本数据的句法依赖')
    createDependenceInformation('data/origin/LapTops_Train_v2.xml','dependences/lp_train.dep')
    createDependenceInformation('data/origin/LapTops_Test_Data_phaseB.xml','dependences/lp_test.dep')
    
    
if __name__=='__main__': 
    print('preProcessing')
    #transformJSONFiles(d_type='re')
    #transformJSONFiles(d_type='lp')
    
    #createDependenceInformation('../../data/ABSA_2014_origin/Restaurants_Train_v2.xml','dependences/re_train.dep')
    #depend_list=loadDependenceInformation('dependences/re_train.dep')
    
    #createAllDependence()
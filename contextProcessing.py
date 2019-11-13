# -*- coding: utf-8 -*-
#用于上下文获取
#获取句法特征信息
import nltk
import re
import pickle
import preProcessing
import entity
import random

#定义AspectContex类
class AspectContext:
    def __init__(self,t_words=[],pol='',context=[],lemm_contex=[],
                 wp_context=[],dep_context=[],context_num=5):
        self.t_words=t_words
        self.pol=pol
        self.context=context
        self.lemm_context=lemm_contex
        self.wp_context=wp_context
        self.dep_context=dep_context
        self.isvalid=True
        self.context_num=context_num
        
    def createBasic(self,term,pol,context_num):
        self.t_words=[re.sub(r"[^A-Za-z0-9]", "", t_word) for t_word in nltk.word_tokenize(term)]
        self.pol=pol
        self.context_num=context_num
        
    def createContext(self,words):
        words_t=[re.sub(r"[^A-Za-z0-9]", "", w) for w in words]
        index=getIndexForTerm(words_t,self.t_words)
        if index==-1:
            print('AspectTerm匹配失败！未找到AspectTerm：%s'%(' '.join(self.t_words)))
            print('当前文本：%s'%(' '.join(words_t)))
            self.isvalid=False
        else :
            self.context=getContextForPol(words,index,len(self.t_words),self.context_num)
        
    def createDepContext(self,deps):
        #当前AT无效跳过即可
        if self.isvalid==False:
            return        
        dict_l={}
        dict_r={}
        dep_c_list=[]
        #统计依存关系
        for dep in deps:
            for triple in dep: 
                if triple[0][0] not in dict_l:
                    dict_l[triple[0][0]]=[triple[2][0]]
                else:
                    if triple[2][0] not in dict_l[triple[0][0]]:
                        dict_l[triple[0][0]].append(triple[2][0])
                        
                if triple[2][0] not in dict_r:
                    dict_r[triple[2][0]]=[triple[0][0]]
                else:
                    if triple[0][0] not in dict_r[triple[2][0]]:
                        dict_r[triple[2][0]].append(triple[0][0])
                        
        #遍历上下文，获取每个单词的上下文
        for word in self.context.split(' '):
            if word not in dict_l: 
                d_l=[]
            else: 
                d_l=dict_l[word]
            if word not in dict_r: 
                d_r=[]
            else: 
                d_r=dict_r[word]

            d_all=list(set(d_l+d_r))
            dep_c_list.append(d_all)
        
        self.dep_context=dep_c_list

def isRightPosition(words,t_words,pos):
    flag=True
    i=pos
    for t_word in t_words:
        if words[i]!=t_word or i==len(words):
            flag=False
            break
        i+=1
    return flag

def getIndexForTerm(words,t_words):
    index=-1
    for i in range(len(words)):
        if isRightPosition(words,t_words,i)==True:
            index=i
            break
    return index
    
#获取Aspect Term的上下文
def getContextForPol(words,index,wlen,context):
    re=[]
    for i in range(index-context,index):
        if i>=0:
            re.append(words[i])
    for i in range(index,index+wlen):
        re.append(words[i])
    for i in range(index+wlen,index+wlen+context):
        if i<len(words):
            re.append(words[i])
    return ' '.join(re)
    
def createFeatureFileForPol(inputpath,deppath,outpath,context,d_type):  
    print('加载数据')
    corpus=preProcessing.loadXML(inputpath)
    dep_list=preProcessing.loadDependenceInformation(deppath)
    instances=corpus.corpus
    
    aspectTermList=[]
    texts_list=[]
    aspectContextList=[]
    
    bio_entity=entity.BIO_Entity(instances,d_type)
    bio_entity.createPOSTags()
    bio_entity.createLemm()
    
    print('获取基础信息')
    for i in range(len(bio_entity.texts)):
        texts_list.append(bio_entity.texts[i])
        aspectTermList.append(bio_entity.instances[i].aspect_terms)
    
        
    for i in range(len(texts_list)):
        for term in aspectTermList[i]:
            aspectContext=AspectContext()
            aspectContext.createBasic(term.term,term.pol,context)
            aspectContext.createContext(texts_list[i].split(' '))
            aspectContext.createDepContext(dep_list[i])
            if aspectContext.isvalid==True:
                aspectContextList.append(aspectContext)
                
    print('切分完成，开始保存')
    with open(outpath,'wb') as f:
        pickle.dump(aspectContextList,f)
    print('完成。')
    
#读取保存有上下文关系的pkl
def loadContextInformation(filepath):
    print('载入上下文关系列表：%s'%filepath)
    with open(filepath,'rb') as f:
        aspectContextList=pickle.load(f)
    return aspectContextList

def createAllForPol(d_type='re',context=5):
    if d_type=='re':
        inputpath='data/origin/Restaurants_Train_v2.xml'
        deppath='dependences/re_train.dep'
        outpath='contextFiles/re_train.cox'
    else:
        inputpath='data/origin/Laptops_Train_v2.xml'
        deppath='dependences/lp_train.dep'
        outpath='contextFiles/lp_train.cox'
        
    createFeatureFileForPol(inputpath,deppath,outpath,context,d_type) 
    
def splitContextFile(filepath,per=0.8):
    data_all=loadContextInformation(filepath)
    print('读取完成，开始切分')
    random.seed(77)
    random.shuffle(data_all)
    train_num=int(len(data_all)*per)
    data_train=data_all[:train_num]
    data_test=data_all[train_num:]
    print('切分完成')
    return data_train,data_test
        
if __name__=='__main__':
    print('context processing')
    createAllForPol(d_type='re',context=5)
    createAllForPol(d_type='lp',context=5)
     
    #aspectContextList=loadContextInformation('contextFiles/re_train.cox')
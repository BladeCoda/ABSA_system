# -*- coding: utf-8 -*-
import nltk
from itertools import chain
from collections import Counter
import preProcessing
import pycrfsuite
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

#将文本转化为CRFsuite可以识别的格式
def word2features(sent, i,window_size=2):
    word = sent[i]['word']
    postag = sent[i]['pos']
    lemm=sent[i]['lemm']

    w2v_c=sent[i]['w2v_c']
    w2v_c_c=sent[i]['w2v_c_c']

    amod_l=sent[i]['amod_l']
    nsubj_r=sent[i]['nsubj_r']
    dobj_r=sent[i]['dobj_r']

    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-2:]=' + word[-2:],
        'word[-3:]=' + word[-3:],
        'word.istitle=%s' % word.istitle(),
        'postag=' + postag,
        'lemm='+lemm,
        
        #以下为依存关系特征        
        'amod_l=%s'%amod_l,
        'nsubj_r=%s'%nsubj_r,
        'dobj_r=%s'%dobj_r,   
        
        #以下为embedding特征
        'w2v_c=%d'%w2v_c,
        'w2v_c_c=%d'%w2v_c_c,

    ]
    
    for j in range(1,window_size+1):
        if i-j>=0:
            word1 = sent[i-j]['word']
            postag1 = sent[i-j]['pos']
            w2v_c1=sent[i-j]['w2v_c']

            features.extend([
                '-%d:word.lower=%s'%(j,word1.lower()),
                '-%d:postag=%s'%(j,postag1),
                '-%d:word.istitle=%s'%(j,word1.istitle()),
                '-%d:w2v_c=%s'%(j,w2v_c1),
            ])
            
        if i+j<=len(sent)-1:
            word1 = sent[i+j]['word']
            postag1 = sent[i+j]['pos']
            w2v_c1=sent[i+j]['w2v_c']

            features.extend([
                '+%d:word.lower=%s'%(j,word1.lower()),
                '+%d:postag=%s'%(j,postag1),
                '+%d:word.istitle=%s'%(j,word1.istitle()),
                '+%d:w2v_c=%s'%(j,w2v_c1),
            ])
    
    if i==0:
        features.append('BOS')
    if i==len(sent)-1:
        features.append('EOS')
                
    return features
    
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
            
def crfFormat_X(train_X,test_X):
    print('开始讲特征转化为CRF格式')
    train_X = [sent2features(s) for s in train_X]
    test_X = [sent2features(s) for s in test_X]
    print('CRF格式转化完成')
    return train_X,test_X
    
def train_CRF(train_X,train_Y):
    trainer=pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(train_X, train_Y):
        trainer.append(xseq, yseq)
    #crf参数设置   
    trainer.set_params({
        #'c1': 1.0,   # coefficient for L1 penalty
        #'c2': 1e-3,  # coefficient for L2 penalty
        #'max_iterations': 75,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    
    trainer.params()
    print('开始训练')
    trainer.train('model/temp.crfsuite')
    print('训练完成')
    trainer.logparser.last_iteration
    print(len(trainer.logparser.iterations), trainer.logparser.iterations[-1])
    
def tag_CRF(test_X):
    tagger = pycrfsuite.Tagger()
    tagger.open('model/temp.crfsuite')
    print('模型读取完成')
    predict_Y=[tagger.tag(xseq) for xseq in test_X]
    print('标注完成')
    return predict_Y,tagger
    
def report_CRF(y_true, y_pred):
    lb=LabelBinarizer()
    
    #chain.from_iterable用于将字符串转化为链表'aaa'->['a','a','a']
    y_true_combined=lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pre_combined=lb.transform(list(chain.from_iterable(y_pred)))
    
    tagset = set(lb.classes_) - {'O'}#不评估'O'标签
    tagset=list(tagset)
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}                
    
    return classification_report(
        y_true_combined,
        y_pre_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
        digits=4
    )
    
def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))
        
def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr)) 
        
#通过bio_tag来获取Aspect Term
def getTermsFromYSeq(yseq,text):
    terms=[]
    flag=0
    term=''
    words=nltk.word_tokenize(text)
    for i in range(len(yseq)):
        if flag==0:
            if yseq[i]=='O':
                continue
            elif yseq[i]=='I':
                print('%s :出现错误转换 O->I')
                continue
            elif yseq[i]=='B':
                term+=words[i]
                flag=1
        elif flag==1:
            if yseq[i]=='O':
                terms.append(term)
                term=''
                flag=0
            elif yseq[i]=='I':
                term+=' '
                term+=words[i]
                flag=2
                continue
            elif yseq[i]=='B':
                terms.append(term)
                term=''
        elif flag==2:
            if yseq[i]=='O':
                terms.append(term)
                term=''
                flag=0
            elif yseq[i]=='I':
                term+=' '
                term+=words[i]
                continue
            elif yseq[i]=='B':
                terms.append(term)
                term=''
                flag=1
    return terms   
    
def getOffestFromText(terms,text):
    offsets=[]
    for term in terms:
        try:
            t_from=text.index(term)
            t_to=t_from+len(term)
            offsets.append({'from':str(t_from),'to':str(t_to)})
        except:
            print(text)
            print("一个AspectTerm匹配失败：%s"%term) 
    return offsets
    
def semEvalValidate(pred_offsets,true_offsets, b=1):
        common, relevant, retrieved = 0., 0., 0.
        #遍历每个句子
        for i in range(len(true_offsets)):
            #正确的aspect term集合
            cor = true_offsets[i]
            #预测的aspect term集合
            pre = pred_offsets[i]
            #正确预测的aspect term个数
            common += len([a for a in pre if a in cor])
            #预测的aspect term总个数
            retrieved += len(pre)
            #实际的aspect term个数
            relevant += len(cor)
        p = common / retrieved if retrieved > 0 else 0.
        r = common / relevant
        f1 = (1 + (b ** 2)) * p * r / ((p * b ** 2) + r) if p > 0 and r > 0 else 0.
        print('P = %f -- R = %f -- F1 = %f (#correct: %d, #retrieved: %d, #relevant: %d)' %
              (p,r,f1,common,retrieved,relevant))
    
def evaluate(detail=False,d_type='re'): 
    if d_type=='re':
        d_name='Restaurants'
    else:
        d_name='LapTops'
    
    print('加载并处理训练数据集')
    train_corpus=preProcessing.loadXML('data/origin/%s_Train_v2.xml'%d_name)
    train_bio=preProcessing.createBIOClass(train_corpus.corpus,d_type)
    dep_path='dependences/%s_train.dep'%d_type
    train_bio.createDependenceFeature(dep_path)
    train_X,train_Y=train_bio.getFeaturesAndLabels()
    
    print('加载并处理测试数据集')
    test_corpus=preProcessing.loadXML('data/origin/%s_Test_Data_phaseB.xml'%d_name)
    test_bio=preProcessing.createBIOClass(test_corpus.corpus,d_type)
    dep_path='dependences/%s_test.dep'%d_type
    test_bio.createDependenceFeature(dep_path)
    test_X,test_Y=test_bio.getFeaturesAndLabels()
    
    true_offsets=[] 
    for i in range(len(test_bio.instances)):
        offset=[a.offset for a in test_bio.instances[i].aspect_terms]
        true_offsets.append(offset)      
    origin_text_test=test_bio.origin_texts
      
    train_X,test_X=crfFormat_X(train_X,test_X)
    train_CRF(train_X,train_Y)
    predict_Y,tagger=tag_CRF(test_X)
    report=report_CRF(test_Y,predict_Y)
    print('\n--------结果报告如下(BIO基准)---------')
    print(report)
    
    if detail==True:
        print('\n--------其他关键信息(BIO基准)---------')
        info=tagger.info()  
        print("可能性最高的状态转移:")
        print_transitions(Counter(info.transitions).most_common(10))
        print("\n可能性最低的状态转移:")
        print_transitions(Counter(info.transitions).most_common()[-10:])
        print("\n最强的特征关联:")
        print_state_features(Counter(info.state_features).most_common(10))
        print("\n最弱的特征关联:")
        print_state_features(Counter(info.state_features).most_common()[-10:])
    
    all_terms=[]
    for i in range(len(origin_text_test)):
        all_terms.append(getTermsFromYSeq(predict_Y[i],origin_text_test[i]))
    all_offsets=[]
    for i in range(len(origin_text_test)):
        all_offsets.append(getOffestFromText(all_terms[i],origin_text_test[i]))
        
    print('\n--------SemEval基准报告如下---------')
    semEvalValidate(all_offsets,true_offsets, b=1)
    
    return all_terms,all_offsets,origin_text_test,true_offsets
    
if __name__=='__main__':
    all_terms,all_offsets,origin_text_test,true_offsets=evaluate(False,'re')
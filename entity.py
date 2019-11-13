# -*- coding: utf-8 -*-

try:
    import sys, random, copy,nltk,re,word2vecProcessing,preProcessing
    from xml.sax.saxutils import escape
    from nltk.corpus import wordnet
    from nltk.stem import SnowballStemmer
    from nltk.tag import StanfordNERTagger
    from nltk.parse.stanford import StanfordDependencyParser

except:
    sys.exit('某些包丢失了?')
    
wn=nltk.WordNetLemmatizer()
sp=SnowballStemmer('english')

#定义Aspect Category类
class AspectCategory:
    def __init__(self,term='',pol=''):
        self.term=term
        self.pol=pol
        
    def create(self, element):
        self.term = element.attrib['category']
        if 'polarity' in element.attrib:
            self.pol = element.attrib['polarity']
        return self

    def update(self, term='', pol=''):
        self.term = term
        self.pol = pol
        
#定义AspectTerm类
class AspectTerm:
    def __init__(self,term='',pol='',offset=''):
        self.term=term
        self.pol=pol
        self.offset=offset
        
    def create(self, element):
        self.term = element.attrib['term']
        if 'polarity' in element.attrib:
            self.pol = element.attrib['polarity']
        self.offset = {'from': str(element.attrib['from']), 'to': str(element.attrib['to'])}
        return self

    def update(self, term='', pol='',offset=''):
        self.term = term
        self.pol = pol   
        self.offset=offset
        
#定义Instance类（一个文本就是一个实例）
class Instance:
    def __init__(self,element):
        #从element中获取需要的信息
        self.text = element.find('text').text
        self.id = element.get('id')
        self.aspect_terms = [AspectTerm('', '', offset={'from': '', 'to': ''}).create(e) for es in
                             element.findall('aspectTerms') for e in es if
                             es is not None]
        self.aspect_categories = [AspectCategory(term='', pol='').create(e) for es in element.findall('aspectCategories')
                                  for e in es if
                                  es is not None]
                                  
    #获取句子中所有的AspectTerm
    def getAspectTerms(self):
        return [a.term for a in self.aspect_terms]
                
    #获取句子中所有的AspectCategory
    def getAspectCategory(self):
        return [a.term for a in self.aspect_categories]
                
    #添加Apect Term
    def addAspectTerm(self, term, pol='', offset={'from': '', 'to': ''}):
        a = AspectTerm(term, pol, offset)
        self.aspect_terms.append(a)

    #添加Aspect Category
    def addAspectCategory(self, term, pol=''):
        c = AspectCategory(term, pol)
        self.aspect_categories.append(c)
        
#统计词典里的词频
def fd(counts):
    d = {}
    for i in counts: 
        d[i] = d[i] + 1 if i in d else 1
    return d

#定义一个函数，讲词频从大到小排序，返回id
freq_rank = lambda d: sorted(d, key=d.get, reverse=True)

#简单地处理下原始文本
#escape函数用于替换掉&, <, 和 > 这几个特性字符
fix = lambda text: escape(text).replace('\"', '&quot;')
        
#语料类
class Corpus:
    def __init__(self, elements):
        #讲每个elements打包成Instance，corpus就是一个Instance的list
        self.corpus = [Instance(e) for e in elements]
        self.size = len(self.corpus)
        #获取所有的AspectTerm
        self.aspect_terms_fd = fd([a for i in self.corpus for a in i.getAspectTerms()])
        #对语料中的单词进行词频排序（从大到小）
        self.top_aspect_terms = freq_rank(self.aspect_terms_fd)
        #获取文本集
        self.texts = [t.text for t in self.corpus]

    #输出一些主要的信息
    def echo(self):
        print('%d instances\n%d distinct aspect terms' % (len(self.corpus), len(self.top_aspect_terms)))
        print('Top aspect terms: %s' % (', '.join(self.top_aspect_terms[:10])))

    #清除语料案例中所有的AspectTerm，主要用于测试
    def clean_tags(self):
        for i in range(len(self.corpus)):
            self.corpus[i].aspect_terms = []

    #按照一定的比例切割训练测试集合
    def split(self, threshold=0.8, shuffle=False):
        '''Split to train/test, based on a threshold. Turn on shuffling for randomizing the elements beforehand.'''
        #拷贝对象（深拷贝）
        clone = copy.deepcopy(self.corpus)
        #随机打乱顺序
        if shuffle: 
            random.shuffle(clone)
        train = clone[:int(threshold * self.size)]
        test = clone[int(threshold * self.size):]
        return train, test

    #讲Corpus对象写成一个文件，用于提交和后续的操作
    def write_out(self, filename, instances, short=True):
        with open(filename, 'w') as o:
            o.write('<sentences>\n')
            #遍历每个实例
            for i in instances:
                o.write('\t<sentence id="%s">\n' % (i.id))
                o.write('\t\t<text>%s</text>\n' % fix(i.text))
                o.write('\t\t<aspectTerms>\n')
                if not short:
                    #遍历AspectTerm然后添加
                    for a in i.aspect_terms:
                        o.write('\t\t\t<aspectTerm term="%s" polarity="%s" from="%s" to="%s"/>\n' % (
                            fix(a.term), a.pol, a.offset['from'], a.offset['to']))
                o.write('\t\t</aspectTerms>\n')
                o.write('\t\t<aspectCategories>\n')
                if not short:
                    #遍历添加AspectCategory
                    for c in i.aspect_categories:
                        o.write('\t\t\t<aspectCategory category="%s" polarity="%s"/>\n' % (fix(c.term), c.pol))
                o.write('\t\t</aspectCategories>\n')
                o.write('\t</sentence>\n')
            o.write('</sentences>')
    
def pos_process(text):
    #分词
    words=text.split(' ')
    poss=[w[1] for w in nltk.pos_tag(words)]
    return poss
    
def lemm_process(text,poss):
    #分词
    words=text.split(' ')
    lemms=[]
    for i in range(len(words)):
        if poss[i][0]=='V':
            lemm=wn.lemmatize(words[i],pos='v')
        else:
            lemm=wn.lemmatize(words[i])
        lemm=sp.stem(lemm)
        lemms.append(lemm)
    return lemms
    
def w2v_cluster_processing(text,dict_w2v,max_c):
    words=text.split(' ')
    t_cluster=[]
    for word in words:
        try:
            t_cluster.append(dict_w2v[word.lower()])
        except:
            t_cluster.append(max_c+1)
            #print('not find Word:%s'%word.lower())
    return t_cluster
  
#解析文本的句法特征
def dep_processing(text,deps):

    amod_l=set([])#amod:形容词关系
    nsubj_r=set([])
    dobj_r=set([])#dobj:直接宾语

    #统计依存关系
    for dep in deps:
        for triple in dep: 
            if triple[1]=='amod':
                if triple[0][0] not in amod_l:
                    amod_l.add(triple[0][0])
                    
            elif triple[1]=='dobj':
                if triple[2][0] not in dobj_r:
                    dobj_r.add(triple[2][0])
                
            elif triple[1]=='nsubj':
                if triple[2][0] not in nsubj_r:
                    nsubj_r.add(triple[2][0])
                    
    t_amod_l=[]
    t_nsubj_r=[]
    t_dobj_r=[]

    for word in text.split(' '):
        if word in amod_l:
            t_amod_l.append(True)
        else:  
            t_amod_l.append(False)
        if word in dobj_r:
            t_dobj_r.append(True)
        else:   
            t_dobj_r.append(False)
        if word in nsubj_r:
            t_nsubj_r.append(True)
        else:
            t_nsubj_r.append(False)

    return t_amod_l,t_nsubj_r,t_dobj_r
    
def isRightPosition(words,t_words,pos):
    flag=True
    i=pos
    for t_word in t_words:
        if words[i]!=t_word or i==len(words) :
            flag=False
            break
        i+=1
    return flag
            
#BIO类
class BIO_Entity():
    def __init__(self, instances,d_type):
        #dtype为数据的类型，有're'(代表餐厅)，'lp'(代表笔记本)，其他的输入会报错
        self.d_type=d_type
        #输入为Instance的列表
        self.instances = instances
        self.size = len(self.instances)
        self.origin_texts=[t.text for t in self.instances]
        self.texts = [' '.join(nltk.word_tokenize(t.text)) for t in self.instances]

        self.bio_tags=[]
        self.pos_tags=[]
        self.word_pos=[]
        self.lemm_tags=[]

        
        self.amod_l=[]
        self.nsubj_r=[]
        self.dobj_r=[]
        
        self.w2v_cluster=[]
        self.w2v_cluster_c=[]

    def createBIOTags(self):
        print('开始标记BIO')
        bios=[]

        #遍历每个句子
        for instance in self.instances:
            terms=instance.getAspectTerms()
            text=instance.text
            
            words=nltk.word_tokenize(text) 
            words=[re.sub(r"[^A-Za-z0-9]", "", w) for w in words]
            bio=['O' for word in words]
            #遍历每一个AspectTerm
            for term in terms:
                t_words=nltk.word_tokenize(term)
                t_words=[re.sub(r"[^A-Za-z0-9]", "", w) for w in t_words]
                try:  
                    cur=words.index(t_words[0])
                    #继续查找
                    if isRightPosition(words,t_words,cur)==False:
                        cur=words.index(t_words[0],cur+1)
                    bio[cur]='B'
                    for i in range(1,len(t_words)):
                        bio[cur+i]='I'
                except:
                    print('查找ApsectTerm失败,跳过此AspectTerm！！！')
                    print('当前文本：%s'%text)
                    print('无法找到的AspectTerm：%s'%term)
            bios.append(bio)

        print('标记完成')
        self.bio_tags=bios
        
    def createPOSTags(self):
        print('开始标记词性')
        t_pos_tags=[pos_process(text) for text in self.texts]
        print('标记完成')
        self.pos_tags=t_pos_tags   
        
    def createLemm(self):
        print('开始词根标记')
        all_lemms=[]
        for i in range(len(self.texts)):
            all_lemms.append(lemm_process(self.texts[i],self.pos_tags[i]))
        print('标记完成')
        self.lemm_tags=all_lemms
        
    def createW2VCluster(self):
        print('开始标记W2V类别')
        dict_W2V=word2vecProcessing.loadDict('cluster/%s_w2v.pkl'%self.d_type)
        max_c=max([item[1] for item in dict_W2V.items()])
        cluster=[w2v_cluster_processing(text,dict_W2V,max_c) for text in self.texts]
        print('W2V类别标记完成')
        self.w2v_cluster=cluster
        
        print('开始标记W2V类别_CBOW')
        dict_W2V=word2vecProcessing.loadDict('cluster/%s_w2v_c.pkl'%self.d_type)
        max_c=max([item[1] for item in dict_W2V.items()])
        cluster=[w2v_cluster_processing(text,dict_W2V,max_c) for text in self.texts]
        print('W2V类别标记完成_CBOW')
        self.w2v_cluster_c=cluster
        
    def createDependenceFeature(self,dep_path):
        print('标记依存信息')
        depend_list=preProcessing.loadDependenceInformation(dep_path)

        dep_amod_l=[]
        dep_nsubj_r=[]
        dep_dobj_r=[]
        
        for i in range(len(self.texts)):
            t_amod_l,t_nsubj_r,t_dobj_r=dep_processing(self.texts[i],depend_list[i])
            
            dep_amod_l.append(t_amod_l)
            dep_nsubj_r.append(t_nsubj_r)
            dep_dobj_r.append(t_dobj_r)

        self.amod_l=dep_amod_l
        self.nsubj_r=dep_nsubj_r
        self.dobj_r=dep_dobj_r
        
    def getFeaturesAndLabels(self):
        print('开始获取特征')
        features=[]
        for i in range(self.size):
            feature=[]
            text=self.texts[i]
            text=text.split(' ')
            for j in range(len(text)):
                feature.append({'word':text[j],'pos':self.pos_tags[i][j],
                                'lemm':self.lemm_tags[i][j],
                                'w2v_c':self.w2v_cluster[i][j],
                                'w2v_c_c':self.w2v_cluster_c[i][j],

                                'amod_l':self.amod_l[i][j],
                                'nsubj_r':self.nsubj_r[i][j],
                                'dobj_r':self.dobj_r[i][j],
                                })
            features.append(feature)
            
            if not (len(text)==len(self.pos_tags[i]) and len(self.pos_tags[i])==len(self.bio_tags[i])):
                print('一个特征匹配失败')
            
        print('特征提取完成，返回特征和标记')
        return features,self.bio_tags
                
    #清除语料案例中所有的AspectTerm，主要用于测试
    def clean_tags(self):
        for i in range(len(self.corpus)):
            self.corpus[i].aspect_terms = []
        
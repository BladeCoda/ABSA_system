# ABSA_system
一个基础且完整的细粒度情感分析（ABSA，Aspect based sentiment analysis）案例，基于semeval-2014 ABSA课题
--------------------------------------

数据与工具
----------------------
SemEval 2014 ABSA 竞赛数据：https://pan.baidu.com/s/18fsF8Bx4yZxpw9gZy2u9Fg
原始数据地址：http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools

yelp 餐厅评论语料：https://pan.baidu.com/s/12h7xCFgnlxZX2CWSlmhlBg
原始数据地址：https://www.yelp.com/dataset

amzon电子产品评论语料：https://pan.baidu.com/s/1emRDSdwWYNZ2RePboYq85w
原始数据地址： http://jmcauley.ucsd.edu/data/amazon/

stanford parser网盘备份(本文版本)：https://pan.baidu.com/s/17xLGWRqHLvA827jpaQmhEQ
原始下载地址（最新）：https://nlp.stanford.edu/software/lex-parser.shtml

注意项
---------------------
1. 在运行代码前，你需要安装如下重要的python库（pip安装即可）
gensim, nltk, python-crfsuite，（其他sklearn，pandas什么的不列了，安个anaconda即可）
2. 其中python-crfsuite为crf模型依赖库，导入用import pycrfsuite。
3. nltk安好后并不能直接使用，需要在python里执行nltk.download()下载模型和语料，这是必须的，否则无法进行词性标注等任务。
4. 代码有用到stanford parser，nltk可以调用其接口，但是工具需要自行下载（链接见顶上），
工程目录下有个空的stanford parser目录，将下载的stanford parser内容全部解压到该目录即可。

* boot.py为程序流程的入口

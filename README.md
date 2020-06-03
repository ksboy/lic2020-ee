# lic2020-ee

## 任务简介
事件抽取可以分成两步：
1. 触发词词提取/事件类型：通常用NER的方式识别触发词，同时也判断出事件类型
2. 事件元素抽取：提取与触发词相关的事件元素

一般来说，触发词不存在重叠问题，但是可能存在多个相同类型的触发词（事件）；同一事件中事件元素可能存在重叠问题。

## 比赛思路

### baseline
pipeline方式：NER方式提取所有的触发词，NER方式提取所有的事件元素，然后根据event_schema匹配触发词和事件元素。

实验结果：  
dev-trigger: f1=0.86427 precision=0.85655 recall=0.87921  
dev-role:    f1=0.62321 precision=0.58821 recall=0.67439  
test:        f1=0.808   precision=0.836 recall=0.782  

### 事件类型分类
由于此次比赛只关注事件类型，并且同一文本中相同类型的事件合并不会影响评测结果；对于任务1，并不需要提取触发词。因此除了NER方式，还可以采用**多标签分类**的方法判断事件类型（采用sigmoid激活函数）。

验证集上的实验结果表明，NER和多标签分类对于事件类型分类结果差不多。
|                      | Precision-macro          | Recall-macro             | F1-macro                 |
| -------------------- | ------------------ | ------------------ | ------------------ |
| NER                  | 0.9418052256532067 | 0.9571514785757392 | 0.9494163424124513 |
| multi-label-classify | 0.9548229548229549 | 0.943874471937236  | 0.9493171471927163 |
| multi-label-classify+ trick | 0.9515445184736523 | 0.9480989740494871 | 0.9498186215235792 |

这里有一个小的trick：当多标签分类结果若为空时，则选择score最大的标签，带来了微小的提升。

我们还尝试了重采样 WeightedRandomSampler，但是 recall 很低。最终没有采用。  
macro_F1 = 0.930539826349566  
micro_f1 = 0.928857823783912  
precision = 1.0  
recall = 0.8823529411764706  


模型融合时：我们也测试了两种方法，logits平均和labels投票，结果相差不大。

基于多标签分类的模型融合结果：  
5-merge-labels: {'precision': 0.9555555555555556, 'recall': 0.96016898008449, 'f1': 0.9578567128236003}  
5-merge-logits: {'precision': 0.9577804583835947, 'recall': 0.9583584791792396, 'f1': 0.9580693815987934}  


### 事件元素提取
事件元素提取有两种思路：
1. 提取触发词相关的事件元素
2. 先提取所有事件元素，然后把事件元素和触发词对应起来

#### 第一种思路
这里采用 start-end 标注方式，为了解决重叠问题。

有几种特征输入的方式：
1. 加入触发词特征：更改触发词对应的 segemnt-id，这里主要参考 PLMEE[2] 。
2. 加入触发词特征：将触发词的 embedding 平均后，和 句子的 embedding 相加，这里主要参考 CASREL [1]。
3. 1 和 2 相结合。
4. 在2的基础上，加入event-type特征。
5. 在3的基础上，加入event-type特征。

验证集上的实验结果，我们只做了3和4：
|                      | Precision          | Recall             | F1                 |
| -------------------- | ------------------ | ------------------ | ------------------ |
| 3                    | 0.7114285714285714 | 0.7410714285714286 | 0.7259475218658891|
| 4                    | 0.7352192362093353 | 0.7031926406926406 | 0.7188493984234545 |

3 在test1上的结果：f1=0.789 precision=0.828 recall=0.754 

recall很低，最后放弃了第一种思路。

#### 第二种思路

我们测试了 BIO 标注和 start-end 标注方式，发现 start-end 标注方式更优。

验证集上的实验结果：
|                      | Precision-span          | Recall-span             | F1-sapn                 |
| -------------------- | ------------------ | ------------------ | ------------------ |
| BIO                  | 0.6721880844242586 | 0.7517179563788468 | 0.7097320169252468 |
| start-end            | 0.7690982194141298 | 0.7147050974112623 | 0.7409046894452898 |
| start-end + 5-merge  | 0.8174037089871612 | 0.7751623376623377 | 0.7957228162755173 |

值得一提的是，模型融合能够带来较大的提升。

### 最终方案

采用pipeline方式：多标签分类得到事件类型，采用 start-end 标注方式提取所有的事件元素，然后根据event_schema匹配触发词和事件元素。

test1结果：precision=0.847 recall=0.842	f1=0.844  
test1排名：28  
test2结果：  
test2排名:20  

## 参考文献

[1] Wei Z, Su J, Wang Y, et al. A Novel Hierarchical Binary Tagging Framework for Joint Extraction of Entities and Relations[J]. arXiv: Computation and Language, 2019.  
[2] Yang S, Feng D, Qiao L, et al. Exploring Pre-trained Language Models for Event Extraction and Generation[C]. meeting of the association for computational linguistics, 2019: 5284-5294.

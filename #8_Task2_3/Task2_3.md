

GitHub地址：https://github.com/f1shungry/Jotang-recruit

## 查阅GNN和GCN有关资料

学习过程中的一些记录

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E5%9B%BE%E7%89%8720221007140514.jpg)

![QQ图片20221007140552](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E5%9B%BE%E7%89%8720221007140552.jpg)



不得不说招新题中提供的参考资料 3 [知乎文章-讲解GCN](https://zhuanlan.zhihu.com/p/71200936)真的太好理解了(T T)



## 论文的中心



- 过平滑 (over-smoothing): 图神经网络经过多层的传播后，各个结点的特征变得趋于一致， 难以分辨
- DropEdge机制: 在每个epoch开始时，以一个固定的概率,，随机去掉一些原图中的边，把新得到的图的邻接矩阵将用于后续的计算
- TABLE2 的实验结果说明:
  - DropEdge 对提升GCN的效果是明显的
  - 随着GCN深度的增加，效果难以有提升， 甚至对某些种类的GCN来说，准确率明显下降。 在使用某种数据集和某种GCN模型的特定情况下， 效果有一些提升， 但是作用微乎其微(不到1%)。 个人感觉，除非必须， 没有必要使用层数很多的GCN
  - 与Original GCN相比， 采用了DropEdge后， 对某些模型来说， 随GCN深度增加， 准确率的下降更平缓一些 

## 理解cora数据集



数据集打开是乱码，光靠想也想不出‘y’ , ‘ty’ , ‘ally’ , ‘x’ , ‘tx’ , ‘allx’ , ‘graph’分别对应些什么，所以去网上寻找答案



> cora读取的文件说明：
>
> ind.cora.x => 训练实例的特征向量，是scipy.sparse.csr.csr_matrix类对象，shape:(140, 1433)，由于ind.cora.x的数据包含于 allx 中，所以实验中没有读取 x
>
> ind.cora.tx => 测试实例的特征向量,shape:(1000, 1433)
>
> ind.cora.allx => 有标签的+无无标签训练实例的特征向量，是ind.dataset_str.x的超集，shape:(1708, 1433)
>
> 实验中的完整的特征向量是有（allx,tx）拼接而成，（2708,1433），在实际训练是整体训练，只有当要计算损失值和精确度时才用掩码从（allx,tx）截取相应的输出
>
> ind.cora.y => 训练实例的 标签，独热编码，numpy.ndarray类的实例，是numpy.ndarray对象，shape：(140, 7)
>
> ind.cora.ty => 测试实例的标签，独热编码，numpy.ndarray类的实例,shape:(1000, 7)
>
> ind.cora.ally => 对应于ind.dataset_str.allx的标签，独热编码,shape:(1708, 7)
>
> 同样是（ally,ty）拼接
>
> ind.cora.graph => 图数据，collections.defaultdict类的实例，格式为 {index：[index_of_neighbor_nodes]}
>
> ind.cora.test.index => 测试实例的id，（1000，）



## 复现论文实验



写在前面：复现的实验我前后做了两版

- 第一版的特点：**没有设随机种子，多层GCN手动搭建**
- 第二版的特点：**设置了随机种子，用循环搭建多层GCN**

> 最终提交的是第二版



### 论文实验的一些小细节

##### 我们可以模仿论文的实验，隐藏层神经元数取128，优化器用Adam optimizer,Epochs=400

> **Implementations.** Different from § 5.1, the parameters of all models are trainable and initialized with the method proposed by [1], and the ReLU function is added.We implement all models on all datasets with the depth*d* *∈ {*2*,* 4*,* 8*,* 16*,* 32*,* 64*}* and the **hidden dimension 128**. 
>
> 
>
> We adopt the **Adam optimizer** for model training.
>
> We fix the number of training epoch to **400** for all datasets.



##### 打印出 ``validation accuracy`` :  验证集上的准确率

- 在第一版实验中我犯了一个错误。论文中提到 `` report the case giving the best accuracy on validation set of each benchmark`` ，而我读到这句话时没有结合上下文理解，误以为是取的是多次训练结果中最大的validation accuracy，但原意应该是：``实验中采用了若干个不同类型的GCN，为了让实验结果更加有效，对每种GCN模型，我们在含有不同超参数的实验环境中选择准确率最高的实验环境进行后续的准确率比较``

  下图是关于这个信息的 完整的语境

  ![image-20221011124645312](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221011124645312.png)
  
  第一版实验中，对于每个不同的模型，我选择记录**训练20次的结果中，最大的validation accuracy**。现在才发现误解了原文的意思，原实验并没有这样做。这种做法有失偏颇
  
  
  
- 还有一点也是我刚开始忽略的：**固定随机种子**

  ![image-20221011125647396](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221011125647396.png)
  
  

神经网络初始化的时候，权重矩阵中的值是随机生成的（前提是合理的、方便向前传递传播和向后传播的值）。正是由于不同的初始化值，每一次的训练结果有可能不一样。为了保证实验的公平性，实验采用了固定随机种子的方法，这样每次初始化获得的值都是相同的。

于是我在第二版实验中对这个问题进行了改进

但是遇到了问题，明明固定了随机种子，每次训练出来的结果还是有差异。我上网查了原因，有人提到加上这两行代码应该可以解决



![image-20221011171951556](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221011171951556.png)



但是这个办法对我来说并没有看到明显的作用。

训练多次以后发现，有些时候训练结果差异太大。再次Google，加上这一行代码之后结果就固定了！

```
torch.manual_seed(SEED) #为CPU设置随机种子
```

我最开始没有加这一行，因为我想我用的是GPU，就不用为CPU设置随机种子了。事实证明CPU也生成了随机数的！是我考虑不周



- 我们在说到神经网络的层数时，通常指具有计算能力的层的数量，因此输入层不算在内。即：**层数 = 隐藏层 + 输出层**

  但是对TABLE2的实验，我确实不知道它的2,4,8,32,64层是单纯指隐藏层还是包括了输出层

  我做的第一版实验是探究的2,4,8,16,32层 **隐藏层** 准确率的不同

  第二版实验进行了改良。把想要的隐藏层层数赋给NUM_HID，层数就不再限制在{2,4,8,16,32}中。但是层数太多了好像不太行。我改到63层时就是下图的效果了
  
  
  
  ![image-20221012083908750](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221012083908750.png)

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221012083814.png)

loss太大超出表示范围



##### GPU的设置

进行这个实验的GPU环境是我之前断断续续搭好的。用的是电脑自带的显卡（不咋好的显卡），CUDA11.6，pytorch 1.12.0

![image-20221012103128061](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221012103128061.png)

前前后后安装了三个版本的CUDA，最终能适配上的是11.6

我在做NLP的时候也试过用colab，但是当时使用colab的过程让我很挫败。第一次使用过后，第二天又要重新在上面安装必要的库，而且告诉我GPU使用时间已到限制。

所以当我发现自带的显卡运行这个实验速度还不错时，选择了用电脑自带的显卡训练模型



#### Original GCN：探究层数增加时，准确率的变化

我是在题目要求给的这个代码框架的基础上，进行相应的修改，做的实验

![image-20221012093619349](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221012093619349.png)

这个框架默认GCN with bias。考虑到论文中TABLE2的数据表明，GCN with bias的准确率会比GCN高一点点，我就沿用到了GCN with bias

![image-20221012094142654](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221012094142654.png)



现在要实现不同层数的GCN。做第一版实验时，我没有找到好的构建多层GCN的方法，感觉可以使用循环，但具体该怎么做我又想不出；在网上搜索分别以中英文的形式搜索了多次也找不到我想要的结果（感觉我利用搜索引擎的能力真的有待提高 T T）。因此，不同层数的GCN全都是我手动打的，非常的麻烦。

（下图为每一层GCN都手动构建的部分截图）

![image-20221011171617688](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221011171617688.png)



后来从别人的做法中（也就是下文会提到的paper ``DROPEDGE: TOWARDS DEEP GRAPH CONVOLUTIONAL NETWORKS ON NODE CLASSIFICATION`` ）找到了一种巧妙地利用循环的构建方法



![image-20221011173209686](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221011173209686.png)



于是将其 ~~剽窃~~ 利用起来，变成了我的代码



![image-20221011182059676](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221011182059676.png)



第二版实验就是这样做的。于是乎我们可以随时根据需要，在下图所示处修改NUM_HID（隐藏层的数量）的值，就可以灵活又便捷地比较层数渐增时准确率的变化了



![image-20221011190102953](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221011190102953.png)



下表是我**第一版实验的结果**

| 层数（隐藏层） | 准确率（20次训练结果中的最大值） |
| -------------- | -------------------------------- |
| 2              | 0.8020000457763672               |
| 4              | 0.7460000514984131               |
| 8              | 0.3160000145435333               |
| 16             | 0.34800001978874207              |
| 32             | 0.1420000046491623               |

可以看到， **随着层数的增加，准确率基本呈降低的走势**  ~~除了16层这个叛徒~~



**第二版实验结果**

| 层数（隐藏层） | 准确率(整理记录时手动保留一位小数) |
| -------------- | ---------------------------------- |
| 1              | 72.6%                              |
| 2              | 79.2%                              |
| 3              | 76.6%                              |
| 4              | 73.2%                              |
| 5              | 38.6%                              |
| 6              | 19.4%                              |
| 7              | 15.6%                              |
| 8              | 7.6%                               |
| 9              | 12.2%                              |
| 10             | 7.4%                               |
| 11             | 11.0%                              |
| 12             | 10.6%                              |
| 13             | 27.4%                              |
| 14             | 14.4%                              |
| 15             | 11.4%                              |
| 16             | 11.4%                              |
| 17             | 11.4%                              |
| 18             | 15.6%                              |
| 19             | 10.2%                              |
| 20             | 12.2%                              |
| 21             | 15.6%                              |

为什么我选的层数是一串连续的自然数，而不再是{2,4,8,16,32}这样离散的自然数呢？结合没有固定随机种子时，多次训练的结果，我认为固定了随机种子以后，对某些层数来说是有利的，也许固定的随机种子对它来说正好能得到较好的准确率；但对另一些层数来说，固定的随机种子可能很不巧撞上了软肋，得到的准确率属于很低的那一批。所以我想通过连续的自然数来获取一个变化的趋势，这样如果出现了极端情况（准确率过高或者过低）就可以认为这个数据是不太有效的



然而这个实验结果，实在是令人难过。1-4层还算正常，只是从4层跨越到5层，怎么一下就从天上到地面了？从5层再到6层，怎么就从地面到地底了？百分之十几这个准确率……我还不如靠猜呢……

这个结果和第一版实验差太多了，第一版16层都有34%呢！我不理解（哽咽）

虽然能看出 **随层数增加，准确率降低** ，但百分之十几的准确率，真的有参考价值吗？我还不如靠猜呢！不算是一个有效的实验吧 T T





#### GCN with DropEdge：探究层数增加时，准确率的变化

DropEdge的原理知道了，但是该如何实现呢？我自己是写不出来的，还是需要~~抄作业~~参考别人的做法

- 这篇论文的源码我没有找到，但是发现了一篇叫做``DROPEDGE: TOWARDS DEEP GRAPH CONVOLUTIONAL NETWORKS ON NODE CLASSIFICATION`` 的论文，这两篇论文的作者完全一致，内容也极其相似，从发表时间来看，``Tackling Over-Smoothing for General Graph Convolutional Networks``应该是其前身

- 我从GitHub上下载了``DROPEDGE: TOWARDS DEEP GRAPH CONVOLUTIONAL NETWORKS ON NODE CLASSIFICATION``的源码

  ![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221010200158.png)
  
  在sample.py中找到了DropEdge的实现代码
  
  ```python
  def randomedge_sampler(self, percent, normalization, cuda):
          """
          Randomly drop edge and preserve percent% edges.
          """
          "Opt here"
          if percent >= 1.0:
              return self.stub_sampler(normalization, cuda)
          
          nnz = self.train_adj.nnz
          perm = np.random.permutation(nnz)
          preserve_nnz = int(nnz*percent)
          perm = perm[:preserve_nnz]
          r_adj = sp.coo_matrix((self.train_adj.data[perm],
                                 (self.train_adj.row[perm],
                                  self.train_adj.col[perm])),
                                shape=self.train_adj.shape)
          #_preprocess_adj和_preprocess_fea也来自sample.py
          r_adj = self._preprocess_adj(normalization, r_adj, cuda)
          fea = self._preprocess_fea(self.train_features, cuda)
          return r_adj, fea
  ```
  
  在train_new.py中可以看到，作为DropEdge中 **保留边** 概率的是``args.sampling_percent``![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221009121439.png)

​				``args.sampling_percent`` 默认值是1.0																																																											![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221009115828.png)

​				运行脚本中设置为0.7	

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221009115809.png)

那就试着采用0.7这个值



##### 理解DropEdge的实现代码

###### 一些查阅

- sp.coo_matrix：scipy.sparse中矩阵的一种形式

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221009163732.png)

由上图所示，``r_adj = sp.coo_matrix((self.train_adj.data[perm],
                               (self.train_adj.row[perm],
                                self.train_adj.col[perm])),
                              shape=self.train_adj.shape)``可以把对应的数据写到矩阵对应的位置



- sp.vstack ( )

> ```python
> features = sp.vstack((allx, tx)).tolil()
> 
> #vsatck把矩阵堆起来.这里就是把allx和x上下堆叠起来
> #tolil-----Convert this matrix to List of Lists format.(链表？)
> ```
>
> 
>
> ![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221009171549.png)



- np.concatenate：把np按axis的值叠起来。axis=0上下叠，axis=1左右叠，axis=None连接成一维数组

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate((a, b), axis=0)
array([[1, 2],
       [3, 4],
       [5, 6]])
np.concatenate((a, b.T), axis=1)
array([[1, 2, 5],
       [3, 4, 6]])
np.concatenate((a, b), axis=None)
array([1, 2, 3, 4, 5, 6])
```



- np.arange(a): 生成{1,2,3,4....,a}的一维数组
- np.reshape: 重构数组的形状（按照参数列表的参数重构）
- np.argmax: 按某种方式找出np数组中值最大的数，返回其下标。具体什么方式要看axis的值

```python
a = np.arange(6).reshape(2,3) + 10
a
array([[10, 11, 12],
       [13, 14, 15]])
np.argmax(a)
5
np.argmax(a, axis=0)
array([1, 1, 1])
np.argmax(a, axis=1)
array([2, 2])
```





###### 理清代码结构

我们的主要目标是函数randomedge_sampler,所以先解决这个函数中不能理解的部分

- self.train_adj是什么？是怎么来的？它的nnz，data，row，col又是什么？
- 函数 _preprocess_adj 和 _preprocess_fea做了些什么？
- normalization做了些什么？

只看函数的话，我没办法解决这些问题，需要先去看看这个程序的 “主函数”

找到train.new.py，看到这个程序中跟NLP一样使用了argparse模块管理参数（截取一部分）

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221010213007.png)



class Sampler用于加载数据

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221010213112.png)

class Sample是sample.py的主体部分，我们需要理解的函数randomedge_sampler、_preprocess_adj 和 _preprocess_fea都是Sample中的函数

创建了class Sampler的实例``sample`` 。我们再看看参数列表中的三个实参

![image-20221010214631866](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221010214631866.png)

![image-20221010214654182](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221010214654182.png)



看看Sample的__ init __部分

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221010214849.png)

有了sample这个实例以后，我们就知道，第一个问题中的``self.train_adj``就是上图中的self.train_adj。继续追根溯源

去找data_loader函数（utils.py中）

![image-20221010215107293](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221010215107293.png)

train.adj = adj，adj来自load_citation函数

再去找load_citation函数(utils.py中)

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221010215207.png)

可以看到，在这个函数中，完成了从cora数据集中加载数据

这个函数乍一看有点头疼，但其实很简单。

``with open (...'rb') as f``是以读二进制文件的方式打开文件，路径就在括号中

路径不是全部手动敲出来的，利用了**os.path.join函数** 和 每个文件路径最后的小尾巴组成的**列表**把各个文件对应的路径组合起来，再利用循环一个个打开加载。加载用的是**pickle.load(f,encoding='latin1')**，文件的一种加载方式，这里不深究。这样加载数据过程就很简洁

（接load_citation)

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221010215952.png)

nx.from_dict_of_list是什么呢？nx就是库networkx，可以处理图数据。nx.from_dict_of_list可以把图转化成nx类型的图

nx.adjacency_matrix可以从nx类型的图中获取邻接矩阵（adjacency）

再看下一行``adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)``

这个奇怪的计算是在干什么？就是把adj转化成一个对称矩阵。这样这个邻接矩阵就是一个有向图的邻接矩阵

然后load_citation又对 adj 和 feature 进行了一些操作

![image-20221010220609275](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221010220609275.png)

去找preprocess_citation函数

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221010220634.png)

然后又要去找fetch_normalization函数（OMG）



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221010220809.png)

根据参数normlization值的不同，会对adj进行不同的操作。比如normalization==AugNormAdj，就会把adj（现在简称A）转换为A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2。其中D是A的度矩阵

yep，这不就是我们学到的GCN的公式吗！

![image-20221010221228552](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221010221228552.png)

再比如，normlization==NoNorm，其实什么都不做，只是把adj转化成sp.coo_matrix 类型

![image-20221010234035579](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221010234035579.png)



（计算A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2的函数）

![QQ截图20221010220833](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221010220833.png)

这个时候我们就能发现，normalization就是对所给的矩阵进行一些变换操作，以便后续的计算。第三个问题解决。



对feature的操作，我也不太懂。。。。为什么要这么做呢。。（最终没有采用这个做法）

![QQ截图20221010220853](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221010220853.png)

到这里load_citaiton差不多就结束了。现在我们就能~~痛苦地~~分析出来**self.train.adj**的来龙去脉：

- 加载数据，获得图
- 从图中得到邻接矩阵，把它转化为对称矩阵，进行一个normlization（根据传进data_loader的参数"NoNorm",这里的normlization其实什么都没有做，只是转化了一下adj的类型）

查了一下资料，了解了第一个问题中的nnz，data，row，col都是sp.coo_matrix中的参数。至此第一个问题解决

那么randomedge_sampler这一段（如下图）代码的作用就是：

**把self.train_adj中的边存到nnz中；将所有边随机到生成一个序列中；nnz*percent 是要保留的边的个数；在刚刚生成的序列中，取前  ``要保留的边的个数`` 条边，用这些边的信息构成新的邻接矩阵**

![image-20221011205058303](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221011205058303.png)

接下来，_preprocess_adj 和 _preprocess_fea 同 _preprocess_citation一样， **对邻接矩阵** 和特征矩阵 **进行变换操作** 

![image-20221011205839223](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221011205839223.png)

（下图为_preprocess_adj 和 _preprocess_fea）

![image-20221011210053513](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221011210053513.png)

到这里我们就知道DropEdge代码的实现过程了



然后观察一下训练部分：
![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221010213642.png)

看到了for循环，对sample（Sample的实例）函数randomedge_sampler的调用，以及train函数的调用。于是我们知道了：DropEdge的步骤在循环中的train环节之前



##### 实现DropEdge

现在可以在所给的框架基础上加入DropEdge机制了。我想要把randomedge_sampler的核心照搬过去，因为自己写的话会debug到怀疑人生……

然后要阅读原框架代码，思考怎么把“照搬”的东西和原来的衔接好。考虑数据的类型是否一致？若不一致又该如何处理？

下面是我阅读过程中的一些记录

![image-20221012100537863](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221012100537863.png)

![image-20221012102043998](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221012102043998.png)

![image-20221012102104396](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221012102104396.png)

接下来就是一个折磨code + debug过程……（略）



##### 一点难题

有一点要注意：计算准确率的时候，要采用DropEdge吗？

我刚开始没有意识到这个问题，直接把DrpoEdge后的邻接矩阵用于准确率的计算，然后发现每次训练的准确率差异可能超过20%，有时候可以很高，高过OriginalGCN；有时候却低的令人发指很不稳定！我思考了很久为什么会出现这种情况，突然想起计算准确率之前我也DropEdge了！于是做出改动：将未处理过的原邻接矩阵用于计算准确率。在这种情况下，准确率稳定了，但是！达不到之前的高度了

私心是希望我加入DropEdge以后，准确率可以有提升的……而我目前做出来的实验，加入DropEdge以后准确率并没有明显的提升。于是我纠结，计算准确率的时候，到底该怎么做？是选择不稳定的高爆发，还是稳定的小渣渣？

最终重新读了论文，找到了第一次阅读时忽略的一句话（下图高亮部分）

![image-20221012095725160](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221012095725160.png)



强调一遍： **计算准确率时，不使用DropEdge**

现在只能接受我的DropEdge不太成功的事实了……



##### 实验结果

和Original GCN（第二版实验）比较，实验结果表格如下

（准确率转化为百分数，保留小数点后一位）



| 层数（隐藏层） | DropEdge准确率 | Original GCN准确率 |
| -------------- | -------------- | ------------------ |
| 1              | 66.0%          | 72.6%              |
| 2              | 75.6%          | 79.2%              |
| 3              | 69.0%          | 76.6%              |
| 4              | 46.6%          | 73.2%              |
| 5              | 12.4%          | 38.6%              |
| 6              | 12.2%          | 19.4%              |
| 7              | 8.4%           | 15.6%              |
| 8 *            | 19.4%          | 7.6%               |
| 9              | 8.2%           | 12.2%              |
| 10 *           | 16.0%          | 7.4%               |
| 11 *           | 14.0%          | 11.0%              |
| 12             | 5.2%           | 10.6%              |
| 13             | 11.4%          | 27.4%              |
| 14             | 5.4%           | 14.4%              |
| 15 *           | 14.0%          | 11.4%              |
| 16 *           | 16.0%          | 11.4%              |
| 17 *           | 15.0%          | 11.4%              |
| 18             | 14.0%          | 15.6%              |
| 19 *           | 15.4%          | 10.2%              |
| 20 *           | 15.6%          | 12.2%              |
| 21             | 13.2%          | 15.6%              |

可以看到，只有在{8，10，11，15，16，17，19，20}层中，采用DropEdge的准确率比Original GCN高（我用星号在层数旁边标注出来了）

时间关系，只能到这里了，an unsuccessful experiment...



## 最后的最后

现在是10.12晚上22:47，招新已接近尾声。从焦糖发布招新题的那天起，我就开启了每天不是上课就是做题的状态，持续到今日。很感谢焦糖能给我们这样一个机会，这一个月里我学到了太多太多；感谢每一位出题人，我深知出题比做题难上好几倍。不管这一个月的结局如何，都祝愿焦糖越来越好！




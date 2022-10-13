GitHub地址：https://github.com/f1shungry/Jotang-recruit

# Task2_1

该文件包括以下内容：

- 以运行train.py为目标的环境配置和代码修改过程
- train.py 复现的截图，训练结果评价指标
- 以运行gui.py为目标的修改过程
- gui.py界面识别表情的截图
- 对题目第5点要求的回答（看懂train.py的源码并回答——它是如何载入数据，如何设计trainer，采用怎样的网络结构？）



## train.py的运行过程

首先给jupyter notebook换环境

我为这个项目创建了一个叫cvi的虚拟环境，并装好了tensorflow和numpy

在这个环境中启用jupyter notebook

![image-20220923170306231](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20220923170306231.png)

再执行conda install nb_conda 的命令

然而下载好nb_conda插件后，出现了报错：

![QQ截图20220920144103](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220920144103.png)

上网查了以后，应该需要把上图最后一排路径的文件删除

![QQ截图20220920145642](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220920145642.png)

但是删除上图的文件以后又出现了报错：

![QQ截图20220920150352](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220920150352.png)

我没注意仔细观察这两次报错的不同，瞄了一眼以为最后一排的路径是一模一样的……确认刚刚的文件已经删除后，又上网查了很久，最后发现这两个路径虽然长得像但是其实是完全不同的，两个路径下的文件都需要删除。



完成删除操作后重新执行一次conda install nb_conda，再进入jupyter notebook 就可以切换到为这个项目新建的虚拟环境了。

试着运行train代码

![QQ截图20220921184819](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220921184819.png)

不出意料地报错了。

我上网搜索了一下，决定把当前2.0.0版本的tensorflow降到1.13.1试试。降版本后出现如下错误：

![image-20220923170406803](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20220923170406803.png)

上网搜索，有人说是numpy和tensorflow版本不兼容。我的numpy是1.19.2，二者不兼容，于是我下载了和1.13.1的tensorflow兼容的numpy1.16.0

然后报错如下：

![image-20220923170602668](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20220923170602668.png)

为了解决这个错误我查了很久。先是搜索No module named 'keras.emotion_models'，出来的搜索结果都是关于No module named 'keras.models'，解决方法我也都尝试了一下，都失败了。我又搜索keras.emotion_models，想知道这个包到底是什么，但是搜索结果中找不到一点关于这个包的信息，只能找到keras.models。这很难不让人怀疑，keras.emotion_models这个包真的存在吗？源代码是不是有问题？我进入这个项目的网站，在评论区中发现了有人遇到了和我一样的问题。

![image-20220923170532768](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20220923170532768.png)

于是我把代码中from keras.emotion_models import Sequential改成了from keras.models import Sequential，接下来报错如下：

![image-20220923170642277](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20220923170642277.png)

又对代码做如下修改：

![image-20220923170656889](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20220923170656889.png)

报错如下：

![image-20220923170717910](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20220923170717910.png)

修改代码中的路径，使之与我电脑中文件存储情况相同

![QQ截图20220922190034](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220922190034.png)

然后报错：

![image-20220923170740099](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20220923170740099.png)

下载pillow包

![image-20220923170802561](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20220923170802561.png)

按下shift+Enter键运行……跑起来了！！！！

![image-20220923170820934](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20220923170820934.png)

（先不管这个warning……

（训练好慢T T

热泪盈眶！！！！！！

![QQ截图20220922220935](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220922220935.png)



训练了一晚上，早上起来发现还是有错

![image-20220923171008244](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20220923171008244.png)



---------------------------------------------------------分割线----------------------------------------------------------------------

由于我自己的电脑暑假的时候坏了拿去维修了，这之前的做的所有事情都是在家里的旧电脑上做的。现在（9月25日）电脑拿回来了，继续之前的工作。

先把环境复刻到我的电脑上（略）

针对上一张图中的报错，做下图①处修改;下图中明显可以看出②处原代码需要做修改，要和本机的文件路径保持一致

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220925222027.png)

50个epoch耗时实在太久了，我自己的电脑比之前用的电脑快一些，但还是要跑两个半小时，现在改成了一个epoch试试

一个epoch的训练很快就结束了，接着弹出了一个video的窗口，并出现了我的大脸！

惊喜！

用手机从网上找了一张图片测试效果。当然因为只训练了一个epoch，可以说没啥准确率



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220925222923.png)



## train.py复现截图

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220927172842.png)



## gui.py的运行

同train.py，首先对这里进行修改，与本机文件路径一致



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220927165559.png)



试着运行，提示“model.h5”不存在（忘记截图了），暂时不清楚model.h5是什么东西。这个时候已经是深夜了，我打算第二天再解决这个问题，睡前用手机搜索了一下model.h5文件，大致浏览了一下这样一篇帖子，大致了解了h5文件可以存放模型的数据，但潜意识以为h5文件要从别的地方下载。



>ML/DL中常见的模型文件(.h5、.keras)简介及其使用方法
>一、.h5文件
>可使用model.save(filepath)函数，将Keras模型和权重保存在一个HDF5文件中，h5文件将包含：
>
>模型的结构，以便重构该模型
>模型的权重
>训练配置（损失函数，优化器等）
>优化器的状态，以便于从上次训练中断的地方开始
>(1)、模型的保存和载入
>
>```python
>model_path = 'model.h5'
>model.save(model_path )                 '保存模型'
>
>from keras.models import load_model
>model = load_model(model_path )         '载入模型'
>
>
>model_weights_path = 'model_weights.h5'
>model.save_weights(model_weights_path )   '保存模型的权重'
>model.load_weights(model_weights_path )   '载入模型的权重'
>
>'如果你需要加载权重到不同的网络结构（有些层一样）中，例如fine-tune或transfer-learning，你可以通过层名字来加载模型'
>model.load_weights('model_weights.h5', by_name=True)
>```
>
>1、常见的h5文件下载
>
>resnet50_coco_best_v2.1.0.h5模型文件
>下载地址：https://github.com/fizyr/keras-retinanet/releases
>
> 
>
>————————————————
>版权声明：本文为CSDN博主「一个处女座的程序猿」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
>原文链接：https://blog.csdn.net/qq_41185868/article/details/97669527



第二天早上，发现gui代码所在的文件夹里多了emotion_model.h5文件



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220926100435.png)



很疑惑它是从哪里来的，并感觉它和原代码的"model.h5"有紧密的联系

于是试着作如下修改



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220926100656.png)



运行，提示‘’logo.png‘’不存在（忘记截图了）

我又搜寻了一遍该项目网站上提供的所有相关资源，没有找到logo.png。浏览项目网站，推断这个‘’logo.png‘’应该是gui界面的最上方的一张小小的图片。于是我把它截了下来，命名为"logo.png"，和gui代码放到同一个文件夹。



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE(2).png)

项目网站的评论中也有相关解释：这个logo.png是这家公司的logo

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220926133356.png)

再次尝试运行，这次没有报错了



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220926082739.png)



没有出现弹窗，但是电脑摄像头旁边的灯开始闪烁了

等待了几分钟还是没有出现弹窗。观察了一下代码，我开始思考是不是最后几行的缩进有问题？然后把删除了最后几行的缩进再次运行。

这时我突然发现任务栏多了一个以前没见过的图标，点开发现



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220927214714.png) 



是他！就是他！

但是不知道这个界面是什么时候出现的，它没有一下子弹出来，我也没有注意观察任务栏的情况。

我把重启了jupyter notebook，再次尝试运行，折腾了几次以后，这个界面可以正常的弹出来了，但是卡的像PPT，而且不管我做什么表情，都识别为surprised。

这个gui界面是如何和之前train.py中训练的网络联系起来的呢？我不由得联想到，昨天晚上训练了一个epoch的模型同样基本上只能将表情识别为surprised。再想想从天而降的emotion_model.h5……

我再次打开昨天晚上看到的那篇帖子，仔细阅读了其中的代码



> ML/DL中常见的模型文件(.h5、.keras)简介及其使用方法
> 一、.h5文件
> 可使用model.save(filepath)函数，将Keras模型和权重保存在一个HDF5文件中，h5文件将包含：
>
> 模型的结构，以便重构该模型
> 模型的权重
> 训练配置（损失函数，优化器等）
> 优化器的状态，以便于从上次训练中断的地方开始
> (1)、模型的保存和载入
>
> ```python
> model_path = 'model.h5'
> model.save(model_path )                 '保存模型'
>  
> from keras.models import load_model
> model = load_model(model_path )         '载入模型'
>  
>  
> model_weights_path = 'model_weights.h5'
> model.save_weights(model_weights_path )   '保存模型的权重'
> model.load_weights(model_weights_path )   '载入模型的权重'
>  
> '如果你需要加载权重到不同的网络结构（有些层一样）中，例如fine-tune或transfer-learning，你可以通过层名字来加载模型'
> model.load_weights('model_weights.h5', by_name=True)
> ```
>
> 
>
> ————————————————
> 版权声明：本文为CSDN博主「一个处女座的程序猿」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
> 原文链接：https://blog.csdn.net/qq_41185868/article/details/97669527



再去查看train.py的代码



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220926102137.png)



这一句将训练的模型的权重保存到同一个文件夹的emotion_model.h5文件中



gui的这句代码则是加载emotion_model.h5文件中的权重保存到gui的模型中



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220926100656.png)

这样gui和train就产生联系了。



（第二天我重新完整地训练了train.py的模型。所以现在emotion_model.h5已经保存了最精确的权重信息）



最难的问题， **GUI界面** 不能总是成功显示。

进行了上述的纠错以后，我运行gui.py时，GUI界面曾有一次正常弹出，并且能够对我的表情做出一些识别；但是我将其关闭后几分钟，再次运行时就不能显示了，只留下摄像头旁边的灯不断闪烁。

我用了一天多的时间寻找原因：

- 学习了tkinter写gui界面的一些基本用法，进行了一些模仿实验

```python
import tkinter as tk
root = tk.Tk()
root.mainloop()
#只写了最基本的写法。有这几行代码是足够弹出一个窗口的，但是gui.py弹不出窗口，为什么？
```



- 给原代码加print（）函数用于测试哪些代码可能有问题，哪些代码没问题

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220927094320.png)



经测试，show.vid函数中的for循环（如下图）经常不能得到执行



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220927171057.png)

为什么？o(╥﹏╥)o

- 翻阅项目网站的评论，看到有人说去掉这两行有作用。我尝试了，有一次去掉以后，GUI界面确实成功出现了。但后来又跑不出来了，合理怀疑是巧合（怒

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220926133356.png)

- 在我不断尝试的过程中,经常会interrupt the kernel 来终止没有本次失败的运行,再重启内核重新运行或者重启jupyter notebook。估计有90%的尝试中，当我interrupt the kernel，任务栏就会立刻弹出羽毛的图标（就是GUI界面），点击羽毛图标，看到的摄像头拍到的画面是静止的（毕竟已经种植运行了）。有的时候，重启后运行就可以成功显示GUI界面，但是过一段时间又失败



GUI界面的触发机制至今让我困惑。我觉得tkinter本就有些不稳定，该界面的显示可能极其依赖电脑状态（？）我最终对gui.py也没有做出什么大的改动，可能只能做到这里了



## gui.py识别表情截图

下面是我趁着GUI界面龙体尚佳时，截下来的表情识别例子………

（好丑啊哈哈哈不要笑我）

![QQ截图20220927135128](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220927135128.png)

![QQ截图20220927135230](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220927135230.png)

![QQ截图20220927135239](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220927135239.png)

![QQ截图20220927135311](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220927135311.png)

![QQ截图20220927135330](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220927135330.png)

![QQ截图20220927135431](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220927135431.png)

![QQ截图20220927135732](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220927135732.png)









## 对题目第5点要求的回答：train.py的设计



#### 如何载入数据：

利用ImageDataGenerator先对数据进行预处理：

- rescale(1./255) : 对图片的每个像素值均乘上1/255放缩因子，把像素值放缩到0和1之间有利于模型的收敛



- flow_from_directory函数获得数据的路径（directory），设置图像大小（target_size），把经过像素值缩放的图像转为向量



#### 如何设计trainer

- 损失函数选择交叉熵函数

- 选择Adam优化器
- 采用dropout防止神经网络过拟合



#### 采用的网络结构

- 采用神经网络中特殊的一种：卷积神经网络，以降低要处理的数据量。

- 网络结构为：

卷积层 --> 卷积层 --> 最大池化层 -->  卷积层 --> 最大池化层 --> 卷积层 ---->完全连接层 -->完全连接层

 




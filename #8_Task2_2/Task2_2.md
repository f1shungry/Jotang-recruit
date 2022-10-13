GitHub地址：https://github.com/f1shungry/Jotang-recruit



## 基本要求



#### 学习json文件

> json是JavaScript Object Notation的简写，翻译过来就是js对象简谱，简单点来说就是一种轻量级的数据交换格式。从结构上看，json文件中的最终可以分解成三种类型：
>
> 第一种类型是标量scalar，也就是一个单独的字符串string或数字numbers，比如“成都”这个单独的词。
>
> 第二种类型是序列sequence，也就是若干个相关的数据按照一定顺序并列在一起，又叫做数组array，或者列表list，比如“成都，重庆”。
>
> 第三种类型是映射mapping，也就是一个名/值name/value，即数据有一个名称，还有一个与之相对应的值，这又称作散列hash或字典dictionary，比如“蓉城：成都”。
>
> 
>
> 1.并列的数据之间用逗号(,)分隔
>
> 2.映射用冒号(:)表示
>
> 3.并列数据的集合(数组)用方括号([])表示
>
> 4.映射的集合(对象)用大括号({})表示
>
> 以上四条规则，就是json格式的所有内容。

接下来就是要读懂train.json文件。

分析后发现，train.json的内容是一个列表，列表中的元素是一个个字典。字典中的键有pre_text, post_test, filename, ......qa等等，而每个键对应的值又可能是一个字典或列表。比如qa对应的值是一个字典，这个字典包括很多键，其中一个就是我们需要的program。

下面进行一些测试：

（下图为json文件的一种处理方式。这种方式比较适合该题的处理要求）

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221001173403.png)





我在jupyter notebook中写了一段测试代码，发生了错误

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221001173411.png)

搜索以后得知，是train.json文件过大，超出了jupyter notebook  的内存限制

**解决办法**：

在启动jupyter notebook时，像下图这样操作（具体加多少0看感觉）

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220929140433.png)



这样运行以后结果如下（很长，只截了一部分）

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221001174035.png)

这就是train.json文件中所有的program内容组成的列表。

接下来的关键问题是，**要把program转换成concat_op**: 例如divide(100, 100), divide(3.8, #0)转换成divide0_divide1

感觉到十分棘手。先去搜索了一些Python字符串的常用处理，比如find(), index()可以用来搜索所需的str是否在字符串中。但始终感觉设计起来很复杂很困难。根据提示决定从源码中获取灵感



#### 阅读源码

费了很大劲才从源码中找到可以用于转换的函数。

首先阅读prepro.py, 里面有一些类和函数是从config.py和finqa.utils.py中引入的，所以这三个文件都需要阅读

- 对不知道的函数和模块进行搜索

  - config.py中 set_args()----涉及**argparse模块**

    ​		![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221001181551.png)

    ​	![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221001190310.png)

    > argparse 模块是 Python 内置的一个用于命令项选项与参数解析的模块，argparse 模块可以让人轻松编写用户友好的[命令行](https://so.csdn.net/so/search?q=命令行&spm=1001.2101.3001.7020)接口。通过在程序中定义好我们需要的参数，然后 argparse 将会从 sys.argv 解析出这些参数。
    >
    > 三个步骤：
    >
    > - 创建一个解析器——创建 ArgumentParser() 对象
    > - 添加参数——调用 add_argument() 方法添加参数
    > - 解析参数——使用 parse_args() 解析添加的参数

    这样管理参数很方便，在用sh脚本运行代码前也可以对参数进行修改，

    

  - prepro.py中**tqdm模块**

    ![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221001191657.png)

    

    > tqdm是一个方便且易于扩展的Python进度条，可以在python执行长循环时在命令行界面实时地显示一个进度提示信息，包括执行进度、处理速度等信息，且可在一定程度上进行定制。

    

  - prepro.py中**strip函数和replace函数**

    ![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221001191631.png)

    strip()将删除字符串开头和结尾的空格，并返回前后不带空格的相同字符串

    replace()把参数列表中的第一个字符串替换为第二个（图中的效果相当于把“· · · · · ·”和“* * * * * * ”删去）

    

  - prepro.py中**split函数**

    ![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221001192322.png)

    同样用于处理字符串，把参数列表中单引号之间的字符作为分割符，对字符串进行分割，返回分割后的若干字符串的列表。

    

  - prepro.py中**enumerate函数**

    ![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221001192849.png)

    > enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    >
    > 例如：
    >
    > ```python
    > >>> seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    > >>> list(enumerate(seasons))
    > [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
    > ```
    >

    

  - prepro.py中tokenize函数

    ![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221002142501.png)

    这个函数是在finqa_utils.py中定义的。于是前去浏览finqa_utils.py, 这时候感觉全篇都在出现tokenize，tokenizer等单词，而我确实不知道它到底是个什么东西。

    上网学习以后，明白了tokenizer是NLP中经常会使用到的分词器，因为原始文本需要处理成数值型字符才能够被计算机处理。

    我们可以看到prepro.py中，首先import了一些bert模型的库

    ![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221002143156.png)

    

    在prepro.py的最后，可以发现源码中多次出现的tokenizer其实是基于BertTokenizer的一些类和函数。Bert模型之后再研究。这里先懂个大概

    

    ![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221002143222.png)

    

  - finqa_utils.py中**re.compile&re.UNICODE**![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221002112054.png)

  ​				是正则表达式的一些处理

  

  - finqa_utils.py中的**extend函数**

    ​	![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221002112656.png)

  ​		用于列表的处理，在这里是把tokenize_fn(token)列表的内容加到token列表的末尾

  ​		

  - finqa_utils.py中的**program_tokenization()**&**get_program_op_args()**

    - 两个函数的名字和形式参数的名字让我有一种``这就是我需要的函数!``的感觉。但是仔细阅读以后我不太能分析出来它的作用效果

      ![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221002113604.png)

      ![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221002113631.png)

    - 首先测试一下program_tokenization函数

      original_program用的divide(100, 100), divide(3.8, #0)的例子，最终的效果是**把original_program分割开来，放到列表中**。

      ![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221002114505.png)

    - 然后测试get_program_op_args()

      ![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221002115917.png)

      输出结果如下：

      ![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221002115923.png)

      get_program_op_args的返回类型是一个元组。

      惊喜地发现，元组中的那个字典里的两个键组合起来不就是我们要的concat_prog吗？！

      

    - 尝试对刚刚提到键进行组合

      ![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221002120531.png)

      ![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221002120701.png)

      但是发生了错误。试着把下标修改为1，才发现这个元组中其实只有True和字典两个元素。

      

    - 换一个示例，将键组合起来

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221002121309.png)

现在可以确认，finqa_utils.py中的**program_tokenization()**&**get_program_op_args()**就是我们可以用来将program转换为concat_prog的工具


​    

- 拉通阅读源码

  

  - 可以发现prepro.py的主体是类CustomDataLoader，这个类中函数很多。如果没有明晰代码的结构很容易晕头转向。明确了CustomDataLoader的主体地位后阅读起来就容易很多了

  

  

#### 项目各模块作用说明

##### finqa_utils.py和preprocess.py负责数据预处理。

- preprocess.py中先对一些数据进行整理

  下图对context进行标准化处理（？不知道如何表达，就是去掉了context中杂乱的符号：最后的空格、星号串和省略号），把原question和context进行拼接，再把拼接结果分割

![image-20221013165019922](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221013165019922.png)

接着处理concat_prog，最后把处理好的数据存到MathQAExample类中，一个个MathQAExample构成examples列表

![image-20221013165353469](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221013165353469.png)

- convert_examples_to_features函数中，利用bert的分词器，把处理好的数据转化为便于电脑处理的特征
- data_loader函数则把特征转化为张量，打包生成输入数据



##### trainer.py负责训练、优化、评估模型



##### 主函数main.py

先加载模型所需的config和分词器tokenizer

![image-20221013170931644](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221013170931644.png)

然后载入数据

![image-20221013170944502](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221013170944502.png)

接着训练

![image-20221013171011971](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221013171011971.png)

如果根据args参数中的要求，需要评估，就调用评估效果的函数

![image-20221013171235241](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221013171235241.png)



#### 写statistics.py

本来是想import finqa_utils.py中的函数，但是运行起来一直报错，因为涉及到了config.py中的argparse模块。![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221002135324.png)



所以我选择把finqa_utils.py中的program_tokenization()ge和t_program_op_args()源码复制到statistics.py 里面



- 测试过程中发现了代码中的很多错误

  先来看看我的第一版代码

  ```python
  def get_concat_prog():
      with open(filename) as f:
          input_data = json.load(f)
      concat_prog_list = []
      concat_prog = ''
      for entry in input_data:
          original_program = entry['qa']['program']
          m = program_tokenization(original_program)
          n = get_program_op_args(m)
          for key in n[1].keys():
              concat_prog = concat_prog + key + '_'
          	concat_prog_list.append(concat_prog)
      return concat_prog_list
  
  result = get_concat_prog()
  print(result)
  
  ```

  

  - 错误1：concat_prog在每次内层循环执行完后都需要清空为'', 否则输出结果中连个逗号都找不到……

  - ![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221001165458.png)

    ​	太长了……这里只截了一小部分。可以发现，输出结果实在太长了，就是因为concat_prog没有清空过，一直在叠加，越叠越多

    

  - 错误2：组合键的过程中多了一个小尾巴

    ​	仔细看这个输出结果，是不是多了个下划线？

    ​	![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221002121309(1).png)

    ​	我还以为自己写的很对，其实错误百出……

    ​	**解决方法**：把concat_pro_list.append(concat_prog)改为concat_pro_list.append(**concat_prog.strip('_')**)

    > 使用**strip()**方法删除最后一个字符
    >
    > Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
    >
    > strip()方法语法：str.strip([chars]);
    >
    > 参数：chars – 移除字符串头尾指定的字符序列。

    

  - 错误3：concat_pro_list.append()的位置？

    该函数的位置应该紧跟在内层循环之外，而不是内层循环中。否则还没组合好的，concat_prog的残次品也会被加入到列表中！

  

- 最终版代码：

  ```python
  def get_concat_prog():
      with open (filename) as f:
         input_data = json.load(f)
      concat_prog_list = []
      concat_prog = ''
      for entry in input_data:
          original_program = entry['qa']['program']
          m = program_tokenization(original_program)
          n = get_program_op_args(m)
          concat_prog = ''
          for key in n[1].keys():
              concat_prog = concat_prog + key + '_'
          concat_prog = concat_prog.strip("_")
          concat_prog_list.append(concat_prog)
  
      return concat_prog_list
  result = get_concat_prog()
  print(result)
  ```

  输出结果（截取部分）：

  ![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221002141759.png)

  

```或许再分析一下源码```



## 进阶要求



#### Pycharm无法运行程序

我一直都是用的jupyter notebook运行python代码，前几天想试一试Pycharm。第一天都用的好好的，结果第二天就出问题了

run按键是灰的

​	debug出现如下提示

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221002134322.png)

​						全网都找不到和我一样的情况

​						重启过了，刚打开加载完成后可以run ，但按下按键后就没有反应了，并且按键变灰



#### Jupyter notebook出现错误-----500 : Internal Server Error

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221004163240.png)



```py
conda install nbconvert==5.4.1
```

解决问题



### 安装apex

cmd进入torch环境

cd到**setup.py**所在目录下，然后运行如下命令：（C:\Users\86199\NLP-Part2\apex-master）

```python
python setup.py install --cpp_ext --cuda_ext
如果此条命令运行不通的话，使用 pip3 install -v --no-cache-dir ./ 和 pip install -v --no-cache-dir ./
```

但是没有反应也



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221004152929.png)

搞错了，没进入虚拟环境, 重来（恼）

看到别人说使用apex要求cuda版本和pytorch相适应

按官网的说法，用如下命令安装了cuda=11.6(虽然我电脑上还有之前装的11.7和10.0(担忧))和pytorch=1.12.0

> conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge





#### 安装过程问题一

##### 报错：subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.

尝试的解决方法：

> E:\Anaconda\envs\torch\Lib\site-packages\torch\utils\cpp_extension.py文件中
>
> `['ninja','-v']`改成`['ninja','--v']` 或者`['ninja','--version']`。
>



#### 安装过程问题二

对问题一尝试上面的解决方法后，又出现了新的报错

```
LINK : fatal error LNK1181: 无法打开输入文件“C:\Users\86199\NLP-Part2\apex-master\build\temp.win-amd64-cpython-37\Release\csrc\multi_tensor_axpby_kernel.obj”
error: command 'C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.29.30133\\bin\\HostX86\\x64\\link.exe' failed with exit code 1181
```



太恼火了, 尝试另一种安装方法



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005114903.png)

出现报错：

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005113826.png)



解决方法：输入命令```pip install -r requirements.txt https://pypi.tuna.tsinghua.edu.cn/simple```

出现报错：

![QQ截图20221005113834](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005113834.png)

重新输入命令：pip install -r requirements.txt https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

又出现报错

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005114154.png)

后来发现，我好像……把顺序搞错了，应该是

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn 包名
```



于是输入

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn requirements.txt
```



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005114802.png)



这里又错了，应该在requirements.txt前加上-r，表示下载requirements.txt文件中的包，即像下方这样输入

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn -r requirements.txt
```



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005114717.png)

这一步OK了

下一步

```python
python setup.py install
```

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005115056.png)

应该是成功了！



### 用train.sh脚本训练模型

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005115758.png)

打开Git运行train.sh 脚本，没有反应

然后发现忘了进入虚拟环境

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005120031.png)

进入虚拟环境后再执行，报错说model.py的某一行（如下图）缩进有问题

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005120019.png)



把上图中的文字注释掉，再执行

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005120415.png)

网络不好? 试着打开代理再执行

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005132000.png)

哦, 看来代理更开不得

上网搜索解决办法

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005140315.png)



决定先把模型下载到本地

Git中执行如下命令

> git lfs install
>
> git clone https://huggingface.co/bert-base-uncased

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005132547.png)



可以看到下载到文件夹里了

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005132951.png)



但运行train.sh还是出错

报错的一些关键信息：

> 
> OSError: [Errno 0] Error
> 
>  (Caused by ProxyError('Cannot connect to proxy.', OSError(0, 'Error')))
> 
>  raise ProxyError(e, request=request)
> requests.exceptions.ProxyError: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: (Caused by ProxyError('Cannot connect to proxy.', OSError(0, 'Error')))
>  
> 
>  f"Can't load config for '{pretrained_model_name_or_path}'. If you were trying to load it from "
> OSError: Can't load config for 'BertConfig {
>  "architectures": [
> "BertForMaskedLM"
>  ],
> "attention_probs_dropout_prob": 0.1,
>  "classifier_dropout": null,
> "gradient_checkpointing": false,
>  "hidden_act": "gelu",
> "hidden_dropout_prob": 0.1,
>  "hidden_size": 768,
> "initializer_range": 0.02,
>"intermediate_size": 3072,
> "layer_norm_eps": 1e-12,
>"max_position_embeddings": 512,
> "model_type": "bert",
> "num_attention_heads": 12,
>  "num_hidden_layers": 12,
> "pad_token_id": 0,
>  "position_embedding_type": "absolute",
> "transformers_version": "4.18.0",
>  "type_vocab_size": 2,
> "use_cache": true,
>"vocab_size": 30522
> }
>'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'BertConfig {
> "architectures": [
> "BertForMaskedLM"
>  ],
> "attention_probs_dropout_prob": 0.1,
>  "classifier_dropout": null,
> "gradient_checkpointing": false,
>  "hidden_act": "gelu",
> "hidden_dropout_prob": 0.1,
>  "hidden_size": 768,
> "initializer_range": 0.02,
>  "intermediate_size": 3072,
> "layer_norm_eps": 1e-12,
>  "max_position_embeddings": 512,
> "model_type": "bert",
>  "num_attention_heads": 12,
> "num_hidden_layers": 12,
>  "pad_token_id": 0,
> "position_embedding_type": "absolute",
>"transformers_version": "4.18.0",
> "type_vocab_size": 2,
>"use_cache": true,
> "vocab_size": 30522
> }
>  ' is the correct path to a directory containing a config.json file



我以为是路径有问题，于是对main.py做如下修改

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005133722.png)



还是报相同的错误



突然发现好像没下载完？！

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005134704.png)

上图最后一行是过了很久才跳出来的。我就说怎么光标一直没回到输入命令的状态

……

好吧，下载好以后还是报一样的错



采用这种方式

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005161203.png)



依然报错，报错的一些关键信息：
>"Connection error, and we cannot find the requested files in the cached path."
>ValueError: Connection error, and we cannot find the requested files in the cached path. Please try again or make sure your Internet connection is on.
>
>We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it in the cached "
>OSError: We couldn't connect to 'https://huggingface.co' to load this model, couldn't find it in the cached files and it looks like BertConfig {
>"architectures": [
> "BertForMaskedLM"
>],
>"attention_probs_dropout_prob": 0.1,
>"classifier_dropout": null,
>"gradient_checkpointing": false,
>"hidden_act": "gelu",
>"hidden_dropout_prob": 0.1,
>"hidden_size": 768,
>"initializer_range": 0.02,
>"intermediate_size": 3072,
>"layer_norm_eps": 1e-12,
>"max_position_embeddings": 512,
>"model_type": "bert",
>"num_attention_heads": 12,
>"num_hidden_layers": 12,
>"pad_token_id": 0,
>"position_embedding_type": "absolute",
>"transformers_version": "4.18.0",
>"type_vocab_size": 2,
>"use_cache": true,
>"vocab_size": 30522
>}
>is not the path to a directory containing a {configuration_file} file.
>Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.



学习了一下别人的写法

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005173353.png)

对model.py做如下修改:

```python
#self.bert = BertModel.from_pretrained(config)
self.bert = BertModel(config)
```



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005162722.png)

报错:

> ValueError: num_samples should be a positive integer value, but got num_samples=0



> 可能的原因：传入的Dataset中的len(self.data_info)==0，即传入该dataloader的dataset里没有数据
>
> 解决方法：
>
> \1. 检查dataset中的路径，路径不对，读取不到数据。
>
> \2. 检查Dataset的__len__()函数为何输出为零



问题应该是没读取到数据，可能是dataset的路径可能不正确……尝试写出具体路径, 但结果没有差别



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005191135.png)



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005205832.png)



继续找办法解决

反复阅读源码以后，发现下图中框出来的部分很奇怪，为什么要把CustomDataLoader这个类传进去?

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005210009.png)

于是创建CustomDataLoader的一个实例,命名为data,把data传进去

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005210040.png)

再运行main.py,报错如下

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005205932.png)



这里处理了很久。 上网搜索后首先怀疑是重名导致的， 但是检查后排除了这个可能性

...一头雾水找不到原因... 

陷入沉思： 实例化以后， not callable；不实例化， 没数据。 那我到底选哪个啊? (?)

在反复检查代码后突然有了新发现！

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005210850.png)

既然我已经创建了CustomDataLoader的实例,，对应形参load_and_cache_examples传了进去，那为什么还要给self.load_and_cache_examples传参数呢? !

于是果断修改如下:

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005211259.png)

成功解决了困扰我几个小时的问题!



新的问题出现了:

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005211210.png)

这个时候就要去检查我自己写的网络了T T

concat_prog_ids不应该作为参数？那我只输入 input_ids 和 input_mask试试

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005221559.png)

以及模仿一下别人的网络

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005221857.png)



结果如下

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005221606.png)



self.model(input[input_ids],input[input_mask])这么写是错的，那我就把后两行都注释掉试试

![QQ截图20221005222015](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005222015.png)

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005221924.png)

> RuntimeError: CUDA out of memory. Tried to allocate 12.00 MiB (GPU 0; 2.00 GiB total capacity; 1.63 GiB already allocated; 0 bytes free; 1.67 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

CUDA out of memory. 真的要笑死了



model改回来

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005222209.png)

还是CUDA out of memory



求助搜索引擎，看到大家说可以尝试 **换小一点的数据集** ， **把batch size改小** ，**加一行代码清理CUDA缓存**



先换一个小很多的数据集

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005232422.png)

还是CUDA out of memory





![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005232706.png)

per_gpu_train_batch_size和 per_gpu_eval_batch_size 从8 改成1



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221005232840.png)

train_batch_size eval_batch_size 从4改成1

还是CUDA out of memory

而且我输入 `nvidia-smi`，显示当前没有占用GPU的应用程序

……

加一行代码 ``torch.cuda.empty_cache()`` 清空缓存，一样没有





夜已深了，决定第二天再折腾

结果第二天开机再跑, 还是out of memory 

我的gpu拿去干嘛了啊!



### 尝试colab

自带显卡伤了我的心……于是尝试一下colab

第一次使用，分配了T4（这个真的转了好久才转出来啊！应该分配显卡用了很久吧

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006150202.png)

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006153519.png)



网上说的转载google drive的方法对我来说似乎不起作用，我的文件夹里看不到我上传的文件

我点了下图中左上角``文件``两个字下方的第三个"装载云端硬盘"按键后, 我上传的文件才加载出来

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006162831.png)



试着运行train.sh

![QQ截图20221006163300](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006163300.png)



从左边的文件列表中获取了正确的路径, 重新尝试

![QQ截图20221006163338](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006163338.png)



main.py的路径不正确. 在train.sh 中,把原文件路径修改为在google drive中的路径

![QQ截图20221006163519](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006163519.png)

再运行, 报错pytorch_transformers不存在, ``pip install pytorch_transformers``安装该包

![QQ截图20221006165827](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006165827.png)

报错apex不存在

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006170009.png)



忍不住打了个寒颤。之前安装apex的过程让我心有余悸...

尝试用以下命令``python /content/drive/MyDrive/NLP/apex-master/setup.py install`` 安装

![image-20221006170236388](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20221006170236388.png)

看到最后一句``Finished processing dependencies for apex``，误以为很顺利, 实则还是``No module named 'apex'``



从StackOverflow找到一个colab安装apex的方法

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006175435.png)

很顺利地安装好了, 新建一个单元格执行import apex 也没问题

但是被告知``cannot import name 'amp' from 'apex' ``

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006175536.png)



怀疑是cuda版本和torch不对应

查看pytorch 版本----> 1.12.1+cu113

cuda版本---->11.2

应该不是对应的, 因为我之前的conda环境安装的pytoch=1.12.0, cuda=11.6

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006183525.png)

想尝试上图的torch=1.10.1+cu111

执行如下命令

```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```

报错:

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006183630.png)

> ```
> ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
> torchtext 0.13.1 requires torch==1.12.1, but you have torch 1.10.1+cu111 which is incompatible.
> Successfully installed torch-1.10.1+cu111 torchaudio-0.10.1+rocm4.1 torchvision-0.11.2+cu111
> WARNING: The following packages were previously imported in this runtime:
>   [torch]
> You must restart the runtime in order to use newly installed versions.
> ```

卸载torchtext

```
pip uninstall torchtext
```

再安装pytorch

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006183852.png)

但是查看版本还是显示之前的

![QQ截图20221006184130](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006184130.png)

然而我又卸不掉自带的版本

```
pip uninstall torch
pip uninstall torch==1.12.1
pip uninstall torch==1.12.1+cu113
```

执行以上三条命令都显示

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006184807.png)







最后发现，原来是因为我没有重启colab.....

重启后再重新下载我需要的版本, 然后查看版本, 已经调整过来了

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006190215.png)



##### 重新安装apex

import amp 失败后, 我一顿折腾没有解决, 阴差阳错把已经安装的apex用pip uninstall卸掉了. 现在文件夹中有一堆apex的残留文件, 而左边的文件列表中不能用鼠标删除非空的文件夹

于是采用linux命令的方式删除文件夹, 以便从头重新安装apex

例如要删除content中的"dist"文件夹

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006192145.png)

执行命令

![QQ截图20221006192150](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006192150.png)

刷新后dist文件夹就被删除了

![QQ截图20221006192205](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006192205.png)



重复安装apex的操作, 然后运行train.sh

![QQ截图20221006195243](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006195243.png)



config.py中做如下修改



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006195225.png)



再运行, 报错说没有model_name_or_path这个变量（而用Git运行的时候这里没有报错（可能是鉴于args.config的值本身就存在），colab要更严格一些）



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006195638.png)



添加这个变量

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006195919.png)



再运行, 报错

> The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache. This is a one-time only operation. You can interrupt this and resume the migration later on by calling `transformers.utils.move_cache()`. Moving 0 files to the new cache system



截取了一点报错信息:

> The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache. This is a one-time only operation. You can interrupt this and resume the migration later on by calling `transformers.utils.move_cache()`. Moving 0 files to the new cache 



Google到一个解决方案：

在train.sh python一行前, 加入下图框起来的一行![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006203134.png)



还是不行



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221006202454.png)



真的不知道怎么解决了， 明明从huggingface下载的模型就在本地...





### 再次回到Git

做了两天Graph以后，再次回到这里，重新尝试用Git运行train.sh

浏览了config.py和train.sh后，突然发现：

出现CUDA out of memory的时候，我不是试过修改过batch_size吗！怎么还是原来的值啊！

所以，真实情况是，我当时昏头昏脑的，只是截了个修改之前的图（记录以防止遗忘），并忘记了我要把batch_size改小这件事？！

然后我就赶紧把所有的batch_size改成1，再次尝试

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221008110540.png)

没有再出现CUDA out of memory了

要被自己蠢哭了

好吧，抖擞精神，接着改新出炉的错误，慢慢完善模型

（忍不住感叹，这个报错真是怎么看怎么可爱呢！）

查看BertModel forward函数源码的返回类型

```python
def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ): ...
#返回部分
if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
```

return的类型叫**BaseModelOutputWithPoolingAndCrossAttentions**，和报错是对应的

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221008112856.png)

对model.py 的forward函数作如下修改

```python
#pooled_output= self.bert(input_ids,input_mask)
pooled_output= self.bert(input_ids,input_mask)[0]#取输出的第一层
```



新的报错：

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221008113234.png)



加一个print语句，查看loss的类型

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221008113931.png)



类型是tensor

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221008114040.png)



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221008140639.png)

![QQ截图20221008140718](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221008140718.png)



说实话，这个训练的过程我不太明白，之前一直在“投机取巧”，现在时间不多了，可能只能走到这里了



#### 终于真正理解了** **input **的含义

我们从下图可以看到，inputs是一个字典，有5个键

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221008182715.png)

之前查了资料，认为input作为一个字典，前面加两个星号就是解压。

现在才知道字典前加两个星号是把字典的值拿出来作为函数的实际参数传进去。并且字典的键要与函数的形式参数名一致

具体地说，之前在这个地方出现问题，是因为我写的model.py的forward函数中，除self以外，只有input_ids, input_mask,segment_ids三个形式参数，没有加上concat_prog_ids和multi_prog_ids（当时我觉得不需要这两个）。

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221008182425.png)

但其实，在``loss = self.model(**inputs)``中，字典input中的5个键对应的值，会一对一地被传到forward函数中，作为forward的实际参数。所以在forward只有三个形式参数的情况下，concat_prog_ids当然就是unexpected argument了（multi_prog_ids也是，只是报错没说）



下图是我之前报错的情况，复习一下

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20221008182249.png)





然而时间有限，对招新题的探索只能到这里了。

我没能成功地运行trainer.py，只能遗憾自己能力有限，止步于此了。












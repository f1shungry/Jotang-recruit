# #3 Linux/WSL



### 基本任务：

选择的是**在虚拟机软件中安装配置Ubuntu发行版**

上网搜索学习了一下，虚拟机软件可以创建虚拟机（多个也可以），虚拟机是一个独立的个体，虚拟机之间不会相互干扰，也不会对电脑本身产生干扰。虚拟机是一个很好的学习其他操作系统的途径。一般选择的虚拟机软件有VirtualBox，VMWare等，我选择了VMWare。

以下是基本任务的完成过程：

1.下载VMWare Workstation Pro

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE(59).png)

2. 安装VMWare(略过)

3. 安装Ubuntu

先在官网中下载Ubuntu，版本22.04.1

![QQ截图20220908201749](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908201749.png)

打开VMWare

![QQ截图20220908201924](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908201924.png)

点击**创建新的虚拟机**

![QQ截图20220908202052](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908202052.png)

然后按图示步骤安装Ubuntu

![QQ截图20220908202122](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908202122.png)

![QQ截图20220908202226](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908202226.png)

![QQ截图20220908202249](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908202249.png)

![QQ截图20220908202318](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908202318.png)



![QQ截图20220908202435](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908202435.png)







![QQ截图20220908202511](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908202511.png)

![QQ截图20220908202606](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908202606.png)



![QQ截图20220908202730](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908202730.png)

![QQ截图20220908202827](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908202827.png)

（这个选地址有点奇怪，合理怀疑中国地区中只能选上海）



创建用户名

![QQ截图20220908202846](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908202846.png)

![QQ截图20220908202901](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908202901.png)

![QQ截图20220908202915](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908202915.png)

安装完成。接下来我试着在上面安装了git



![QQ截图20220908203019](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908203019.png)

![QQ截图20220908203030](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908203030.png)

输入红框中的命令，查看git版本并创建用户名和邮箱

![QQ截图20220908203228](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908203228.png)

至此基本任务完成。





## 进阶任务

#### 一. VS Code Remote插件远程连接Ubuntu

- 查看虚拟机IP地址

打开虚拟机，输入**ifconfig**命令，红框中即为虚拟机IP地址

![QQ截图20220908232859](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908232859.png)

- 开启虚拟机ssh服务

```
#输入如下命令：
sudo apt-get install openssh-server	 #安装
sudo /etc/init.d/ssh start	#启动
ps -e|grep ssh	#确认服务是否开启
```

​	(第一条命令没有截图)

![image-20220909191505035](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20220909191505035.png)

![image-20220909191554697](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20220909191554697.png)

- 打开VSCode，下载Remote插件



![QQ截图20220908232533](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908232533.png)

点击上图所示红圈处，下图的两个插件都自动下好了

![QQ截图20220908232558](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908232558.png)

- 点击左下角，在弹出的框中选择**connect to host**

![image-20220909184507929](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20220909184507929.png)

![image-20220909184607554](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20220909184607554.png)	

选择添加新的主机

![image-20220909184653149](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20220909184653149.png)-



输入 ssh 主机名@ip

我的主机名是VM，IP地址192.168.106.128（主机名原来是f1hungry-virtual-machine,后来被我改成了VM）

于是输入ssh VM@192.168.106.128



- 打开配置文件

![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908232235.png)



再点击连接就有Ubuntu这个主机名。点击**Ubuntu**，此时输入框会要求输入虚拟机密码，输入密码后等待连接。右下角变成下图这样就说明连上了。



![image-20220909192107594](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20220909192107594.png)



现在在Windows终端中输入命令ssh Ubuntu，回车后，输入虚拟机密码即可在Windows终端中打开虚拟机的终端。



#### 二. 配置基于SSH密钥的远程的服务器登陆



>  想要远程免密登录，需要ssh生成公钥和私钥，公钥存在虚拟机的.ssh文件夹下，私钥保存在Windows本地



用如下命令生成公钥和私钥，id_rsa.pub文件是公钥，id_rsa文件是私钥，都存储在.ssh文件夹下

（下图是我用使用ssh连接git 和GitHub时，生成的公钥和私钥）



![image-20220909193004430](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20220909193004430.png)



这里问题出现了：我打开.ssh文件夹，里面有id_rsa，但没有id_rsa.pub



![image-20220909193449139](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20220909193449139.png)

尝试在搜索框中搜索关键词“id”，搜索结果中有id_rsa.pub，但是打不开这个文件

![QQ截图20220908200716](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908200716.png)



很奇怪的问题。我刚开始觉得有点难处理，然后突然想起来可以试试用cat命令读取id_rsa.pub中的内容



![image-20220909224606519](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20220909224606519.png)



第二个问题出现了，我在虚拟机中始终找不到.ssh文件夹。上网搜索后，有人说用输入如下命令，我尝试了一下，但还是找不到。



![QQ截图20220908233217](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908233217.png)







翻文件翻了很久，最后打开了一个**显示隐藏文件夹**的开关，突然找到了它

![QQ截图20220908233241](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908233241.png)



在VSCode的配置文件中加上下图最后一行，红框中是私钥文件id_rsa的路径

![image-20220910105615686](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/image-20220910105615686.png)



然而第三个问题又出现了：由于我操作失误，在虚拟机中也输入命令生成了ssh密钥，并把虚拟机上**不与Windows的私钥相匹配**的id_rsa.pub文件写进了authorized_keys文件，导致我一直没有成功免密登录，总是要求我输入密码才能登陆。

![QQ截图20220909225351](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220909225351.png)

![QQ截图20220909225413](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220909225413.png)

![QQ截图20220909225435](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220909225435.png)

我研究了很久，最后打开authorized_keys查看了一下，突然发现这不是我在Windows下生成的公钥！

![QQ截图20220908234056](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908234056.png)



这是authorized_keys 的内容

![QQ截图20220910110233](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220910110233.png)



下图是虚拟机生成的公钥的内容。从红线划出来的地方可以看出来不是Windows下的公钥（Windows的公钥是以我的QQ邮箱结尾的）

![QQ截图20220908234035](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908234035.png)

这是Windows下的公钥

![QQ截图20220909232357](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220909232357.png)



对比以上三图划红线处，可以看出authorized_keys的内容不正确！



我把正确的公钥写入authorized_keys中，打开Windows终端，输入命令

```
ssh Ubuntu	#Ubuntu是我起的Host名
```



然后成功地免密登录了！

![QQ截图20220908221732](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908221732.png)



现在我们来试一下在VSCode中远程编程



点击新建文件

![QQ截图20220908221829](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908221829.png)



写了一个”HelloWorld“ C程序![QQ截图20220908221843](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908221843.png)



按下Ctrl+s保存文件，在弹出的框中确定文件名为Helloworld.c

![QQ截图20220908221901](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908221901.png)



接着打开虚拟机，在对应的目录下也找到了Helloworld.c 文件

![QQ截图20220908200448](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220908200448.png)

 到这里任务就完成啦！

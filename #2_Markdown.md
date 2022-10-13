GitHub地址：https://github.com/f1shungry/Jotang-recruit

# #2 Markdown

## 使用图床



焦糖的宣讲会中有学长提到markdown文件最好使用图床存储图片，否则别人是看不见你md文件中的图片的

上网搜索后，我首先尝试了使用**七牛云+PicGo**上传自己的图片。七牛云的好处是可以免费使用30天。



#### 七牛云：白嫖失败



- 安装PicGo

  ![QQ截图20220909093803](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220909093803.png)

  

- 进入七牛云官网，创建账号并设定Bucket后，进入密钥管理



![QQ截图20220909100620](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220909100620.png)



- 打开PicGo，进入**图床设置**，将七牛云中的密钥，Bucket名和提供的临时域名（仅30天内有效，30天后需另行处理）填入PicGo中



![QQ截图20220909101734](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220909101734.png)



- 点击**设为默认图床**

  

  ![QQ截图20220909101804](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220909101804.png)

  

- 打开Typora，点击左上角**文件**，再点击**偏好设置**



![](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220909101953.png)



- 按下图进行设置

  ![QQ截图20220909102151](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220909102151.png)

![QQ截图20220909102350](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220909102350.png)



试着上传了一张图片，屏幕右下角通知*上传成功*



![QQ截图20220909102655](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220909102655.png)

本以为一切都准备妥当了，然而我新建了一个md文件，明明成功上传了图片，却连自己都看不到图片ToT

![QQ截图20220909105940](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220909105940.png)

发给朋友测试也不能看到图片

我尝试了很久都无法解决，上网学习看到很多博主推荐阿里云，于是决定用阿里云代替七牛云试一试。



#### 转战收费的阿里云



- 进入阿里云的官网，充值开通oss服务后，创建Bucket



![QQ截图20220909113450](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220909113450.png)



- 设置Bucket

  

![QQ截图20220909113607](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220909113607.png)



- 进入**AccessKey管理**



![QQ截图20220909113655](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220909113655.png)



- 创建AccessKey



![QQ截图20220909113734](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220909113734.png)



- 打开PicGo，显示的图床选择阿里云OOS，并填好阿里云的设置



![QQ截图20220909114030](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220909114030.png)

![QQ截图20220909114557](https://f1sh-hungry.oss-cn-chengdu.aliyuncs.com/QQ%E6%88%AA%E5%9B%BE20220909114557.png)

- 上传图片，可以成功地预览了！发送给朋友也能看见图片。

**果然白嫖的东西多少是会有问题的！**


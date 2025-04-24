# myocr2-invoice

#### 介绍
发票OCR识别，实现方式使用RT-DERTv2目标检测提取关键位置发票信息，PaddleOCR根据提取的位置进行文字识别。
支持图片和PDF识别，主要识别了发票标题、发票代码、发票号码、开票日期、购买方名称、购买方识别号、销售方名称、销售方识别号、含税金额、不含税金额、税费等信息。

#### 说明
由于YOLO和PaddleOCR无法同时使用GPU加速, 使用RT-DERTv2替代YOLO, 在GPU环境下耗时变为二百多毫秒, 而且RT-DERT的开源协议是Apache-2.0 license使用起来顾虑会更少

#### 软件架构
RT-DERTv2+PaddleOCR+Flask+easyofd


#### 安装教程

方式一: docker安裝, 目前只封裝了cpu环境, 速度有些慢(大概6秒往上)。GPU环境请根据Dockerfile自行研究, 理论上将镜像启动后进入容器将paddlepaddle替换为gpu版本就行了。 <br>
  1. docker pull qwerdocker78/myocr2-invoice-mini:0.0.2 <br>
  2. docker run -d --name myocr2-invoice --privileged=true -p 5000:5000 qwerdocker78/myocr2-invoice-mini:0.0.2 -c https://pypi.tuna.tsinghua.edu.cn/simple 
  3. 说明: 由于需要下载的python依赖很大10多个G上传不上去, 镜像中未包含依赖信息, 依赖信息会在docker run时下载, 所以第一次运行时会需要花费很长时间。 
  4. 依赖下载并运行成功后可以进入容器更换paddlepaddle版本(镜像内置的paddlepaddle==2.5.2在不同服务器上paddlepaddle==2.6.2这两个版本速度不一定谁快谁慢, 能差个三四秒左右), 然后重启容器即可 <br>
  [1] docker exec -it myocr2-invoice2 /bin/bash <br>
  [2] pip install paddlepaddle==2.6.2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ <br>
  [3] 退出容器 exit 或者使用组合键:Ctrl + D <br>
  [4] docker restart myocr2-invoice2 <br>
<br>

方式二: Python3.9环境

  1. pip install -r requirements.txt <br>
  2. 若使用gpu， 请注释掉requirements.txt的paddlepaddle==2.6.2，根据官网中配置信息下载对应的，https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html <br>

[//]: # (3. pip install paddleocr)
  3. (1)gunicorn -w 4 -b 0.0.0.0:5000 main:app 端口可以自由设置 <br>
   (2)python main.py <br>
   以上都可以启动服务 <br>
  4. 启动成功发送地址测试http://127.0.0.1:5000/invoice_ocr

#### 测试截图

经测试GPU环境下平均二百多毫秒，CPU环境下一秒左右
![输入图片说明](images/imagesimage.png)(GPU环境平均耗时)
![输入图片说明](https://foruda.gitee.com/images/1733118115493827803/c57188ac_5748498.png "CPU环境平均耗时")

![输入图片说明](https://foruda.gitee.com/images/1733117764446225949/6cdf117d_5748498.png "屏幕截图")

#### 后续
目前训练数据种类比较少，后续逐步完善。

#### 自己训练模型
训练RT-DETRv2检测模型可以使用auto_label.py进行半自动标注，标注完成后使用PPOCRLabel打开directory变量的目录进行微调即可，调整完成后转换为coco格式数据就可以训练啦，后续会整理提供格式转换的脚本。


#### 注意注意注意



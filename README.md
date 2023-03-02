# Brain-Tumor-Classification-Based-on-Deep-Learning
## 项目简介
基于深度学习的脑肿瘤识别。搭建了一个卷积神经网络对输入的脑肿瘤图片进行多级下采样，随后通过一个线性层输出分类标签，实现对脑肿瘤的分级（即判断为HGG还是LGG）。
## 平台及环境
Windows10/11，python3.7与配套的pytorch环境。
## 数据集
BraTS2018数据集，下载网址见https://blog.csdn.net/csdn_hxs/article/details/114703185

在项目中原始数据集路径为/tumor_data。若github下载速度过慢可以选择在上述网址下载原始数据集。
## 数据预处理
为了避免超出4G显存，将三维脑肿瘤图片降维至二维。因为脑肿瘤只分布在较少的切片层中，所以使用分割标签选出脑肿瘤中心所在的切片层作为输入图片，将155\*240\*240的输入图片降维成1\*240\*240。

根据数据集在/tumor_data/HGG路径还是/tumor_data/LGG路径，分别将分类标签编码为1和0。
## 模型
每个下采样块由两个卷积块和残差通道组成。下采样四次。随后通过一个线性层输出分类标签。
## 代码与文件
DatasetGenerator.py和TumorDataset.py从原始数据集生成pytorch的数据集。Model.py描述神经网络模型，TrainResNet.py训练模型并测试。

每个python文件都是一个类和类的主方法，顺次运行上述五个文件的类主方法即可。运行中会产生/pkl目录文件，存放从原始数据集制作pytorch数据集的中间结果。下载本项目时可以不下载/pkl目录文件，运行程序即可得到相同文件。

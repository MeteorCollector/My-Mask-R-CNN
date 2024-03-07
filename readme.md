# My Mask R-CNN
根据[pytorch 官方教程](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)复现[Mask R-CNN](https://export.arxiv.org/pdf/1703.06870.pdf)。

使用方法：见[文章](https://meteorcollector.github.io/2024/03/Mask-RCNN-reproduce/)。

可以根据`requirements.txt`配置环境，但是不推荐，感觉这些脚本的环境要求其实是比较宽松的。我使用的是`python 3.10`+`cuda 12.1`。建议到`pytorch`官网复制适用于自己电脑的`pytorch`安装指令。

请自行下载PennFudan数据集，下载方法为

```
wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip -P data
cd data && unzip PennFudanPed.zip
```

训练：
`python train.py`
目前遵循官方文档仅训练2个epoch。如果想要修改请自行改变`num_epochs`的值。

推理：
`visualize.py`
请自行修改`image = read_image("path/to/image")`路径来制定图片。
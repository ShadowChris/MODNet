# 1. 实现抠图：
1. 把输入图片全放入data/input中
2. 预训练模型
   1. https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR
   2. 下载抠图模型：modnet_photographic_portrait_matting.ckpt，并放到pretrained/内

2. 运行以下命令：
````
python -m workspace.portrait.get_foreground
````
3. data/output中会输出所有图片的抠图前景
4. 可以通过命令行参数修改输入输出图片的路径，也可以到get_foreground.py内修改默认值

# 2. 训练模型：
1. 运行workspace/portrait/train.py
2. 训练完毕后，收敛曲线生成位置：doc/xxx.png

训练代码参考:
https://github.com/ZHKKKe/MODNet/issues/200
https://www.kaggle.com/code/daggerx/modnet/notebook

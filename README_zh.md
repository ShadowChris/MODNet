# 1. 运行抠图demo：
1. 把输入图片全放入data/input中
2. 运行以下命令：
````
python -m workspace.portrait.get_foreground
````
3. data/output中会输出抠图前景
4. 可以通过命令行参数修改输入输出图片的路径，也可以到get_foreground.py内修改默认值

# 2. 训练模型：
1. workspace/portrait/230328-train.ipynb
2. 运行所有cell，得到训练结果

训练代码参考:
https://github.com/ZHKKKe/MODNet/issues/200
https://www.kaggle.com/code/daggerx/modnet/notebook

预训练模型:
https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR
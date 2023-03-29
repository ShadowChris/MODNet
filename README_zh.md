运行抠图demo：
1. 把输入图片全放入data/input中
2. 运行workspace/portrait/getForeground.ipynb
   * 需要根据自己需求修改.ckpt模型路径
3. data/output中会输出抠图前景

训练模型：
1. workspace/portrait/230328-train.ipynb
2. 运行所有cell，得到训练结果

训练代码参考:
https://github.com/ZHKKKe/MODNet/issues/200
https://www.kaggle.com/code/daggerx/modnet/notebook

预训练模型:
https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR
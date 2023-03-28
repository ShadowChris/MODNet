"""
MODNet的监督训练迭代
这个函数在一个有标签的数据集中训练MODNet一个迭代。

参数：
    modnet (torch.nn.Module): MODNet实例
    optimizer (torch.optim.Optimizer): 监督训练的优化器
    image (torch.autograd.Variable): 输入的RGB图片，其像素值应已被归一化
    trimap (torch.autograd.Variable): 用于计算损失的trimap，其像素值可以是0、0.5或1
                                      （前景=1，背景=0，未知区域=0.5）
    gt_matte (torch.autograd.Variable): 真实的alpha抠像，其像素值在[0, 1]之间
    semantic_scale (float): 语义损失的比例，请根据您的数据集进行调整
    detail_scale (float): 细节损失的比例，请根据您的数据集进行调整
    matte_scale (float): 融合损失的比例，请根据您的数据集进行调整

返回值：
    semantic_loss (torch.Tensor): 语义估计的损失 [低分辨率 (LR) 分支]
    detail_loss (torch.Tensor): 细节预测的损失 [高分辨率 (HR) 分支]
    matte_loss (torch.Tensor): 语义-细节融合的损失 [融合分支]

示例：
    import torch
    from src.models.modnet import MODNet
    from src.trainer import supervised_training_iter

    bs = 16         # 批量大小
    lr = 0.01       # 学习率
    epochs = 40     # 总共的训练轮次

    modnet = torch.nn.DataParallel(MODNet()).cuda()
    optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * epochs), gamma=0.1)

    dataloader = CREATE_YOUR_DATALOADER(bs)     # 注意：请完成这个函数

    for epoch in range(0, epochs):
        for idx, (image, trimap, gt_matte) in enumerate(dataloader):
            semantic_loss, detail_loss, matte_loss = \
                supervised_training_iter(modnet, optimizer, image, trimap, gt_matte)
        lr_scheduler.step()
"""

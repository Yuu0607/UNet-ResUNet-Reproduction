# UNet 图像分割复现

## GitHub 项目

- 仓库：https://github.com/milesial/Pytorch-UNet
- 使用版本：v3.0
- 模型：U-Net
- 预训练数据：Kaggle Carvana Image Masking Challenge
- 任务：车辆前景/背景二分类语义分割

## 当前环境

- Python：D:\anaconda\python.exe（Python 3.13.5）
- PyTorch：2.15.0+cpu
- torchvision：0.29.2+cpu
- 推理设备：CPU

## 已下载文件

- `Pytorch-UNet-3.0/`：GitHub 源码
- `Pytorch-UNet-v3.0.tar.gz`：GitHub 源码压缩包
- `unet_carvana_scale0.5_epoch2.pth`：官方 Carvana 预训练权重（124239509 字节）
- `carvana_sample.jpg`：Carvana 样例输入图
- `carvana_sample_mask_255.png`：二值分割掩码
- `carvana_unet_result.jpg`：输入图、掩码和叠加结果三栏预览
- `1.py`：一键复现脚本

## 运行

在 PyCharm 中直接运行 `1.py`，或在 PowerShell 执行：

```powershell
D:\anaconda\python.exe D:\UNet_Reproduction\1.py
```

## 本次复现结果

- 输入尺寸：1918 x 1280
- 车辆掩码像素：630005
- 掩码占比：25.66%
- CPU 推理时间：约 6.08 秒
- 模型权重加载和输出保存均正常

## 使用官方预测脚本

```powershell
cd D:\UNet_Reproduction\Pytorch-UNet-3.0
D:\anaconda\python.exe predict.py --model D:\UNet_Reproduction\unet_carvana_scale0.5_epoch2.pth --input D:\UNet_Reproduction\carvana_sample.jpg --output D:\UNet_Reproduction\official_predict_mask.png --scale 0.5
```

## 水平翻转 TTA 对比实验

- 普通推理代码：`1.py`
- TTA 对比代码：`2_tta_compare.py`
- 真实训练样例：`carvana_train_image.jpg`
- 真实标注掩码：`carvana_train_gt_mask.png`

运行：

```powershell
D:\anaconda\python.exe D:\UNet_Reproduction\2_tta_compare.py
```

输出：

- `tta_baseline_mask.png`：普通 UNet 掩码
- `tta_result_mask.png`：水平翻转 TTA 掩码
- `tta_ground_truth_mask.png`：真实标注掩码
- `tta_comparison.jpg`：五栏对比图
- `tta_metrics.json`：Dice、IoU、Precision、Recall 和耗时

本次结果：

- 普通推理 Dice：0.990721
- 翻转 TTA Dice：0.990905
- Dice 提升：0.000184
- 普通推理 IoU：0.981613
- 翻转 TTA IoU：0.981974
- IoU 提升：0.000362
- TTA 修正像素：400
- TTA 破坏像素：227
- 普通推理耗时：约 6.54 秒
- TTA 推理耗时：约 12.66 秒

说明：TTA 通过多一次翻转预测提高了稳定性，但推理时间约翻倍。

## ResUNet 结构修改对比

- 模型代码：`resunet_model.py`
- 对比训练与验证代码：`3_resunet_compare.py`
- 普通 UNet 微调权重：`unet_finetuned.pth`
- ResUNet 微调权重：`resunet_finetuned.pth`
- 详细指标：`resunet_metrics.json`
- 单张代表图对比：`resunet_comparison.jpg`
- 8 张验证图完整对比：`resunet_validation_montage.jpg`

结构修改：

- 在每个 DoubleConv 双卷积块中加入残差支路；
- 通道不一致时使用 1×1 卷积加 BatchNorm 做投影；
- 每个残差支路使用可学习缩放系数，初始值为 0；
- 原 UNet 参数约 31.04M，ResUNet 参数约 32.44M。

公平对比设置：

- 同一批 Carvana 数据：32 张训练图、8 张验证图；
- 相同预训练权重；
- 相同损失函数、学习率和 5 轮微调；
- 使用 CPU 训练和验证。

验证集平均结果：

- UNet Dice：0.973830
- ResUNet Dice：0.974017
- Dice 提升：+0.000187
- UNet IoU：0.949071
- ResUNet IoU：0.949418
- IoU 提升：+0.000346
- 8 张验证图中，ResUNet 在 4 张上更好，4 张上略差。
- UNet 微调耗时约 140.78 秒，ResUNet 微调耗时约 154.62 秒。

运行：

```powershell
D:\anaconda\python.exe D:\UNet_Reproduction\3_resunet_compare.py
```

## 仓库文件说明

- 预训练权重和微调权重未上传到 Git，因为单个文件超过 GitHub 100 MB 限制。
- Carvana 数据集未上传，`resunet_data/` 需要从 Kaggle 获取。
- 结果图片、指标 JSON 和对比图已包含在仓库中，便于直接查看。
- 原始预训练权重下载地址：
  `https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth`
- 本项目基于 `milesial/Pytorch-UNet` 修改，遵循 GPL-3.0 许可证。

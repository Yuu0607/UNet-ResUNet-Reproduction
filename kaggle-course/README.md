# Kaggle 机器学习课程实验代码

四个实验各有一个独立运行入口。训练数据、测试图像、模型权重和比赛提交结果由本地或 Kaggle 会话保存。仓库只包含代码和运行说明。

| 任务 | 入口 | 训练和评估输出 |
| --- | --- | --- |
| 植物幼苗分类 | `01-plants/run.py` | 最佳权重、训练曲线数据、验证概率、错分样本表、提交 CSV |
| 六类表情分类 | `02-expressions/run.py` | 最佳权重、训练曲线数据、验证概率、错分样本表、提交 CSV |
| 眼底中心凹定位 | `03-fovea/run.py` | 折权重、分折日志、OOF 坐标、测试坐标、提交 CSV |
| 眼底血管分割 | `04-vessels/run.py` | 折权重、分折日志、OOF Dice、概率归档、RLE 提交 CSV |

## 环境与运行顺序

在 `kaggle-course` 目录运行下列命令。示例中的 `data/` 路径按自己的 Kaggle 数据挂载位置替换。Kaggle Notebook 可在代码单元开头使用 `%%bash`，再执行相同命令。每个配置使用不同的输出目录。

```bash
python -m pip install -r requirements.txt
python 01-plants/run.py train --help
python 02-expressions/run.py train --help
python 03-fovea/run.py --help
python 04-vessels/run.py train --help
```

PyTorch 和 torchvision 应使用匹配的版本。Kaggle 已预装的版本可直接使用。下载预训练权重需要联网或已有缓存。分类任务可用 `--scratch` 进行短轮次检查，中心凹旧骨干也支持该选项。切换随机种子或折分文件后要重新生成配套的 OOF 文件。历史分数来自课程运行记录，下列重实现应重新训练后单独记录成绩。

## 1 植物幼苗分类

`train/` 下每个类别一个子目录，`test/` 下放测试图。示例采用 80/20 分层划分、随机种子 42，以 Macro F1 保存最佳轮次。运行顺序是基线、裁剪对照、分辨率对照、其他骨干和正则化，最后验证融合。每条训练命令均保存原始推理与指定 TTA 的结果。

```bash
for recipe in b0-224 b0-224-light b0-224-no-crop b0-224-smoothing \
              b0-288-light b0-320-light b2-224-light b2-320-light \
              b3-320-light convnext-tiny efficientnetv2-s; do
  python 01-plants/run.py train --train-dir data/plants/train \
    --recipe "$recipe" --out "runs/plants/$recipe"
done

python 01-plants/run.py compare --inputs \
  runs/plants/b0-320-light/validation_four.csv \
  runs/plants/b2-320-light/validation_four.csv \
  --out runs/plants/b0-b2-validation.json

python 01-plants/run.py predict --test-dir data/plants/test \
  --template data/plants/sample_submission.csv --id-column ID \
  --checkpoints runs/plants/b0-320-light/best.pth runs/plants/b2-320-light/best.pth \
  --weights 1 1 --tta four --out runs/plants/submission.csv
```

四个独立裁剪消融配置为 `crop-0.75`、`crop-0.85`、`crop-0.90`、`crop-0.95`。224 轻裁剪增强和 0.05 标签平滑来自保存的实验记录。后续骨干和各尺寸共享训练框架。各版本请用同一划分进行验证，比较 `metrics_none.json`、`metrics_four.json` 与 `errors_*.csv`。测试模板如使用其他 ID 列名，修改 `--id-column`。

## 2 六类表情分类

训练目录包含 Angry、Fear、Happy、Neutral、Sad、Surprise 六个子目录。默认以准确率选模并采用 85/15 分层划分，重现 80/20 比较时给所有候选加同一个 `--val-fraction .20`。

```bash
for recipe in resnet18-scratch resnet18-pretrained efficientnet-b0 \
              vit-torchvision vit-imagenet vit-trpakov vit-hardlyhumans; do
  python 02-expressions/run.py train --train-dir data/expressions/train \
    --recipe "$recipe" --out "runs/expressions/$recipe"
done

python 02-expressions/run.py train --train-dir data/expressions/train \
  --recipe vit-weighted --init runs/expressions/vit-hardlyhumans/best.pth \
  --out runs/expressions/weighted
python 02-expressions/run.py train --train-dir data/expressions/train \
  --recipe vit-weighted-low-lr --init runs/expressions/weighted/best.pth \
  --out runs/expressions/weighted-low-lr

python 02-expressions/run.py compare --inputs \
  runs/expressions/vit-imagenet/validation_horizontal.csv \
  runs/expressions/efficientnet-b0/validation_horizontal.csv \
  --weights .8 .2 --out runs/expressions/vit-cnn-validation.json

python 02-expressions/run.py predict --test-dir data/expressions/test \
  --template data/expressions/submission.csv \
  --checkpoints runs/expressions/vit-imagenet/best.pth \
  --label-order Angry Fear Happy Sad Surprise Neutral \
  --out runs/expressions/submission.csv
```

`--label-order` 必须与比赛提供的整数类别定义一致。示例取自一份历史推理日志，提交前按官方说明核对。领域权重候选、加权续训和低学习率尝试均保留供对照。正式选用外部表情权重前需要确认竞赛规则及权重训练数据来源。 `original_notebook.ipynb` 保留找回的原 Notebook 代码单元，包含原来的临时调试及不同版本，适合核对历史。顺序训练和导出使用 `run.py`。

## 3 眼底中心凹定位

标签可用原比赛的 `data,Fovea_X,Fovea_Y`，或规范列 `ID,True_X,True_Y`。ID 与文件名去后缀后的键对应。模板需要 `ImageID,value`，每张图有 `ID_Fovea_X` 和 `ID_Fovea_Y`。若保留了历史五折划分，传入 `--fold-csv`，其列为 `ID,fold`。否则以固定种子重新划分。

```bash
fit_fovea() {
  version="$1"; shift
  python 03-fovea/run.py --version "$version" --train-dir data/fovea/train \
    --test-dir data/fovea/test --labels data/fovea/fovea_localization_train_GT.csv \
    --template data/fovea/sample_submission.csv --out "runs/fovea/$version" "$@"
}
for version in mean v1 v3 v4 v8 v12; do
  fit_fovea "$version"
done

python 03-fovea/blend.py --inputs runs/fovea/v4/oof.csv runs/fovea/v8/oof.csv \
  --out runs/fovea/v4-v8/oof.csv
python 03-fovea/blend.py --inputs runs/fovea/v4/test_predictions.csv \
  runs/fovea/v8/test_predictions.csv --out runs/fovea/v4-v8/test_predictions.csv \
  --template data/fovea/sample_submission.csv

for version in v9 v11; do
  fit_fovea "$version" --coarse-oof runs/fovea/v4-v8/oof.csv \
    --coarse-test runs/fovea/v4-v8/test_predictions.csv
done
```

V1/V3 为单次划分 ResNet18 热图对照，V3 使用等比例填充。V4/V8 是五折热图重实现，V9 是局部修正，V11 是带限幅的 ResNet34 修正。局部模型命令用于方法对照。若要得到严格的两阶段 OOF 分数，应在每个外层折内部生成训练集粗预测，外层验证图的标签不能用于训练任何给该折提供输入的粗模型。示例中的公共粗 OOF 输入不构成嵌套交叉验证。

V12 保留六通道结构与原始混合目标，默认三种子、五折、80 轮、60/70/80 轮 EMA 平均和水平 TTA。对比最佳单轮 EMA 时另开目录并用 `--strategy best`。原脚本保留在 `original_v12.py`，它依赖原 Kaggle 目录和原五折 CSV。新入口可直接从标注生成划分。标为 (0,0) 的训练图会从坐标回归中排除。目前没有经过验证的测试可见性判别器，因此测试导出为可见点回归；若比赛数据含不可见图，需另提供可靠的可见性判定。

## 4 眼底血管分割

训练图、测试图和标签分别放在不同目录。同一图的文件名主干必须可对齐，例如 `01.png` 和 `1.png` 都归为 ID 1；带额外后缀的原始标签需先规范命名。 `--foreground dark` 表示黑色为血管，`bright` 表示白色为血管。对照既有旧目标，也有按比赛缩放规则处理后的目标。若有已经确认过的比赛目标掩膜，优先直接提供。下列转换按恢复到的双线性缩放后非零即前景实现，输出白色前景。

```bash
python 04-vessels/run.py prepare-masks --train-dir data/vessels/train \
  --mask-dir data/vessels/masks --foreground dark --out data/vessels/official

for recipe in v1 v2 v3 v4 extra-trees; do
  python 04-vessels/run.py train --recipe "$recipe" --foreground dark \
    --train-dir data/vessels/train --test-dir data/vessels/test \
    --mask-dir data/vessels/masks --out "runs/vessels/$recipe"
done

for recipe in v5 v7-tversky v8-core v9-stride32 v10-thin v11-focal-trial v12-context; do
  python 04-vessels/run.py train --recipe "$recipe" --foreground bright \
    --train-dir data/vessels/train --test-dir data/vessels/test \
    --mask-dir data/vessels/masks --official-mask-dir data/vessels/official \
    --out "runs/vessels/$recipe"
done

python 04-vessels/run.py compare --inputs runs/vessels/v5/oof.npz \
  runs/vessels/v7-tversky/oof.npz runs/vessels/v9-stride32/oof.npz \
  --weights .5 .25 .25 --foreground bright --mask-dir data/vessels/official \
  --thresholds .45 .50 .55 --out runs/vessels/ensemble-validation
```

默认输出四方向 TTA，可加 `--no-tta` 得到普通推理。快速重做局部对照时，V12 可用 `--only-folds 0 2 3`，再以 `compare --common-ids` 对齐到同一批验证图。 `compare` 会同时输出无后处理、闭运算和单像素短断点填补的 Dice、精确率、召回率及错误像素数。V1+V2、V1+V4、神经网络+ExtraTrees 等融合可传入同一标签口径下的多个归档。血管分割各模型为依照可查记录重新实现，V8 辅助核心采用训练目标腐蚀近似。V6 的独立训练配置和 V11 的完整旧代码未找回。V11 Focal BCE 配置是计划试验实现，现有记录不能确认它已完成。

导出提交时，按照比赛说明给出正确列名、展开顺序与起始索引。以下命令使用 `Id`、`Predicted`、列优先和从 1 开始作为命令参数示例，具体值请以模板为准。

```bash
python 04-vessels/run.py submit --inputs runs/vessels/v5/test.npz \
  --template data/vessels/sample_submission.csv --id-column Id \
  --target-column Predicted --order F --start 1 --threshold .50 \
  --out runs/vessels/submission.csv
```

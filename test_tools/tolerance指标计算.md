# tolerance 指标加速那
## 为什么要使用tolerance指标计算
因为我们进行的是篡改边缘检测任务，所以在对比的时候我们首先需要将篡改区域转化为篡改边缘
## 如何完成指标计算
### 1 输入：gt; pred_area/pred_edge

1. area--> std edge
2. std edge -->edge
3. 

### 2 如何计算tolerance precision
TP/(TP+FP)
如果设置允许容忍度为1，则相当于gt上为白色像素的点按照 kernel=3 膨胀了
但是FP 还是原来的像素数，其他容忍度同理

### 3 如何计算tolerance rcall
recall=TP/(TP+FN)
如果设置允许容忍度为1，则相当于gt上为白色像素的点按照 kernel=3 膨胀了
FN 还是原来的像素数，其他容忍度同理


### 4 F1
略

### 
[[146296   3806]
 [  1671    947]]
 表示
 第i行第j列条目的混淆矩阵表示真实标签为第i类且预测标签为第j类的样本数。
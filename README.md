# semantic_segmentation_contest_deeplabv3
遥感图像稀疏表征与智能分析竞赛之语义分割。晃晃悠悠比赛就结束了，因为是第一次参加自己方向
上的比赛，主要是通过这个比赛学习和巩固语义分割的知识，同时也为随后的比赛增加经验。所有的
代码都已经整理上传。
最后的结果：（仅仅只用了deeplabv3 + dropout）第27名。
## 结果

|       |Method                                | OS  | kappa       |
|:-----:|:------------------------------------:|:---:|:----------:|
| repo  | MG(1,2,4)+ASPP(6,12,18)+Image Pooling|16   | **50.662%** |

图片结果：
<p align="center">
      <img src="resource/2.png" width=1000 height=500>
</p>
<p align="center">
  <img src="resource/1.png" width=1000 height=500>
</p>

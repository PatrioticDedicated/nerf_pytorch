NeRF Refactoring
=====
The code of [NeRF](https://arxiv.org/pdf/2003.08934.pdf)(Neural Radiance Fields) is restructured according to pytorch_template.

# Features
* Clear folder structure which is suitable for many deep learning projects.
* Checkpoint saving and resuming,training process logging, and more.
* Multi-gpu training
* Reconstructed color grid

# Tutorial
NeRF uses an MLP to represent a static scene (implicit reconstruction). The input is spatial coordinates and viewing Angle, and the output is color and volume density. After volume rendering, a composite image from a new perspective can be obtained.  

Input: coordinates $x(x,y,z)$ in space, camera angle direction $d(\theta,\phi )$  

Mapping: $f_{\Theta }:(x,d) \to (c,\sigma )$, $\Theta$ is a parameter of the networ  

Intermediate output: color $c(r,g,b)$, volume density $\sigma$  

Final output: The volume is rendered to get an RGB image

## NeRF 实现过程
### 第一步 光线的生成
$\text { 输入：一张RGB图像 输出：图像每个像素的 } o(x, y, z), d(x, y, z) \text { 以及由 } o, d \text { 得到的 } x(x, y, z)$
MLP的输入是一系列空间坐标的点 $x(x, y, z)$ ，这些点同时具有相机视角（或者说方向）这个属 性，这一系列的点可以模拟出一条从相机发射出的光线。光线的生成过程如下:
#### 1.1 坐标系的转换：在进行光线的生成之前，首先需要了解在 NeRF 中一些相关坐标概念:
在NeRF中数据的处理涉及到三种坐标系:
世界坐标系：表示物理上的三维世界坐标
相机坐标系：表示虚拟的三维相机坐标
图像坐标系：表示输入图片的二维坐标

其中不同坐标系下的坐标有以下的转换关系：相机中的坐标 $\left[X_c, Y_c, Z_c\right]^T$ 和三维世界的坐标 $[X, Y, Z]^T$
$
\begin{bmatrix}
 &  &  &  \\
 &  &  &  \\
 &  &  &  \\
 &  &  &  \\
\end{bmatrix}
$















![llff](https://github.com/PatrioticDedicated/Result/blob/main/gif/llff.gif)
![lego](https://user-images.githubusercontent.com/61340340/236772533-a7d382ab-2155-47f1-8c57-87efa8949ec2.gif)

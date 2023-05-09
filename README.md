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
$$
\left[\begin{array}{l}
X_c \\
Y_c \\
Z_c \\
1
\end{array}\right]=\left[\begin{array}{llll}
r_{11} & r_{12} & p_{13} & t_x \\
r_{21} & r_{22} & r_{23} & t_y \\
r_{31} & r_{32} & r_{33} & t_z \\
0 & 0 & 0 & 1
\end{array}\right]\left[\begin{array}{l}
X \\
Y \\
Z \\
1
\end{array}\right]
$$
等式右边的矩阵是一个仿射变换矩阵，用于从世界坐标转换到相机坐标，而在 NeRF 中会提供其 逆矩阵用于从相机坐标转换到统一的世界坐标。而二维图片的坐标 $[x, y]^T$ 和相机坐标系的坐标转 换关系为:
$$
\left[\begin{array}{c}
x \\
y \\
1
\end{array}\right]=\left[\begin{array}{ccc}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{array}\right]\left[\begin{array}{l}
X_c \\
Y_c \\
Z_c
\end{array}\right]
$$
等式右边的矩阵指相机的内参，包含焦距以及图像中心点的坐标，对于相同的数据集内参矩阵一般 是固定的。














![llff](https://github.com/PatrioticDedicated/Result/blob/main/gif/llff.gif)
![lego](https://user-images.githubusercontent.com/61340340/236772533-a7d382ab-2155-47f1-8c57-87efa8949ec2.gif)

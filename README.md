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
输入：一张RGB图像  
输出：图像每个像素的 $o(x, y, z)$, $d(x, y, z)$ 以及由 $o, d$ 得到的 $x(x, y, z)$
MLP的输入是一系列空间坐标的点 $x(x, y, z)$ ，这些点同时具有相机视角属性，这一系列的点可以模拟出一条从相机发射出的光线。光线的生成过程如下:
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

等式右边的矩阵是一个仿射变换矩阵，用于从世界坐标转换到相机坐标，而在 NeRF 中会提供其 逆矩阵用于从相机坐标转换到统一的世界坐标。而二维图片的坐标 $[x, y]^T$ 和相机坐标系的坐标转 换关系为：

$$
\left[\begin{array}{l}
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
有了以上的参数，我们可以对一张图像进行下面的处理:  
(1) 对一张图片进行二维的像素坐标构建  
(2) 进行像素坐标到相机坐标的变换（根据相机内参)  
(3) 进行相机坐标到世界坐标的变换（根据相机外参)
#### 1.2 光线的生成
通过相机的内外参数，可以将一张图片的像素坐标转换为统一的世界坐标系下的坐标，我们可以确 定一个坐标系的原点，而一张图片的每个像素都可以根据原点以及图片像素的位置计算出该像素相 对于原点的单位方向向量 $d$ (用三维坐标表示，原点也是使用三维坐标表示)，根据原点 $O$ 和方 向 $d$ ，改变不同的深度 $t$ ，就可以通过构建一系列离散的点模拟出一条经过该像素的光线 $r(t)=o+t d$ 。这些点的坐标和方向就是 NeRF 的 MLP 输入，输出经过体渲染得到的值与这条 光线经过的像素的值得到 loss。

### 第二步 位置编码
输入: 一条光线上的 $x(x, y, z), d(x, y, z)$ 输出: $\gamma(x), \gamma(d)$
位置编码公式: $\gamma(p)=\left(\sin \left(2^0 \pi p\right), \cos \left(2^0 \pi p\right), \ldots, \sin \left(2^{L-1} \pi p\right), \cos \left(2^{L-1} \pi p\right)\right)$ 这里 应该关注维度的变化： $R->R^{2 L}$ ，需要注意在上一步提到模拟一条光线时用的是一系列离散的 点，那么对应这些点的坐标都是不同的（64个），但单位方向 $d(x, y, z)$ 对于这条光线上的点来 说都是相同的，对每一个点进行位置编码，原来是 3 维， L 取 10，那么最终的维度就是 $3 \times 2 \times 10=60$ 维，同理单位方向向量维度也是 3 ， L 取 4，最终是 24 维，这就是论文中 MLP 网 络上提到的 $\gamma(x) 60$ 和 $\gamma(d) 24$ 。
![v2-83926be94a1f3f0876b0e4752f35eae2_r](https://user-images.githubusercontent.com/61340340/237008128-7e28dab6-1f60-419a-b8a3-f141fc498ca2.jpg)


### 第三步 MLP预测
输入: $\gamma(x), \gamma(d)$ 输出: $c(r, g, b), \sigma$
![MLP](https://user-images.githubusercontent.com/61340340/237011945-ce4f502a-55f6-45e0-ade3-ac74dea45240.PNG)

用一系列的点模拟一条光线， 一条光线穿过一个像素，也就是说对一条光线上的每个点，都需要经过一次MLP，在文中提到一条 光线粗采样64 个点那么这 64 个点都会经过MLP，也就是会输出 64 个 $\sigma$ ，然后再加入 $\gamma(d)$ ， 注意对这 64 个点来说它们都是处在同一条光线上，所以每个点的 $\gamma(d)$ 都是一样的，然后得到 64 个点对应预测的 rgb 值。

### 第四步 体渲染
输入: 一条光线上的 $c(r, g, b), \sigma$ 输出: 渲染后的 RGB 值
在传统的体渲染方法中，通过吸收发射模型进行光强的计算:
$I(0)=\int_0^{\infty} g(s) T^{\prime}(0, s) d s=\int_0^{\infty} T^{\prime}(0, t) \tau(t) c(t) dt$
其中 $T^{\prime}(s)=\exp \left(-\int_s^D \tau(x) d x\right)$ ，这一项被称为透明度，吸收发射模型等式第一项表示来自背景的光，乘以空间的透明度，这一部分表示光照经过介质后被吸收剩下的光强，第二项是源项 $\mathrm{g}$ （s）(表示介质通过外部照明的发射或反射增加的光）乘以位置 $\mathrm{s}$ 到眼晴位置 $D$ 的透明度（即 $\left.T^{\prime}(s)\right)$ 在每个位置 $\mathbf{s}$ 贡献的积分 (注意这个思想，我们使用一系列的点模拟一条光线，那么每个 点都有它的属性) 。在 NeRF 中吸收发射模型等式第一项视作背景光，忽略不计，通过坐标换算 之后得到：
$$
I(0)=\int_0^{\infty} g(s) T^{\prime}(0, s) d s=\int_0^{\infty} T^{\prime}(0, t) \tau(t) c(t) d t
$$
其中 $T^{\prime}(0, t)=\exp \left(-\int_0^t \tau(x) d x\right)$
那么 $\sigma(r(t))$ 可以表示在这条射线上， $t$ 位置的体积密度（也就是体密度，预测出来的 $\sigma$ ）， $c(r(t), d)$ 就可以表示在这条射线上， $t$ 位置 $d$ 方向的光强。再考虑到不是每个位置上都有介质， 取了介质的边界平面 $t_n, t_f$ ，最终得到论文中的公式:
$C(r)=\int_{t_n}^{t_f} T(t) \sigma(r(t)) c(r(t), d) d t ; \int T(t)=\exp \left(-\int_{t^n}^t \sigma(r(s)) d s\right), r(t)=o+t d$



# Training

 

![llff](https://github.com/PatrioticDedicated/Result/blob/main/gif/llff.gif)
![lego](https://user-images.githubusercontent.com/61340340/236772533-a7d382ab-2155-47f1-8c57-87efa8949ec2.gif)

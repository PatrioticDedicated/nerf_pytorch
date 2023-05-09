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

* ```Input```:  coordinates $x(x,y,z)$ in space, camera angle direction $d(\theta,\phi )$  

* ```Mapping```:  $f_{\Theta }:(x,d) \to (c,\sigma )$, $\Theta$ is a parameter of the networ  

* ```Intermediate output```:  color $c(r,g,b)$, volume density $\sigma$  

* ```Final output```:  The volume is rendered to get an RGB image

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

等式右边的矩阵是一个仿射变换矩阵，用于从世界坐标转换到相机坐标，而在 NeRF 中会提供其逆矩阵用于从相机坐标转换到统一的世界坐标。而二维图片的坐标 $[x, y]^T$ 和相机坐标系的坐标转换关系为：

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
通过相机的内外参数，可以将一张图片的像素坐标转换为统一的世界坐标系下的坐标，我们可以确定一个坐标系的原点，而一张图片的每个像素都可以根据原点以及图片像素的位置计算出该像素相对于原点的单位方向向量 $d$ (用三维坐标表示，原点也是使用三维坐标表示)，根据原点 $O$ 和方 向 $d$ ，改变不同的深度 $t$ ，就可以通过构建一系列离散的点模拟出一条经过该像素的光线 $r(t)=o+t d$ 。这些点的坐标和方向就是 NeRF 的 MLP 输入，输出经过体渲染得到的值与这条 光线经过的像素的值得到 loss。

### 第二步 位置编码
输入: 一条光线上的 $x(x, y, z), d(x, y, z)$ 输出: $\gamma(x), \gamma(d)$
位置编码公式: $\gamma(p)=\left(\sin \left(2^0 \pi p\right), \cos \left(2^0 \pi p\right), \ldots, \sin \left(2^{L-1} \pi p\right), \cos \left(2^{L-1} \pi p\right)\right)$ 这里 应该关注维度的变化： $R->R^{2 L}$ ，需要注意在上一步提到模拟一条光线时用的是一系列离散的点，那么对应这些点的坐标都是不同的（64个），但单位方向 $d(x, y, z)$ 对于这条光线上的点来说都是相同的，对每一个点进行位置编码，原来是 3 维， L 取 10，那么最终的维度就是 $3 \times 2 \times 10=60$ 维，同理单位方向向量维度也是 3 ， L 取 4，最终是 24 维，这就是论文中 MLP 网络上提到的 $\gamma(x) 60$ 和 $\gamma(d) 24$ 。
![v2-83926be94a1f3f0876b0e4752f35eae2_r](https://user-images.githubusercontent.com/61340340/237008128-7e28dab6-1f60-419a-b8a3-f141fc498ca2.jpg)


### 第三步 MLP预测
输入: $\gamma(x), \gamma(d)$  
输出: $c(r, g, b), \sigma$
![MLP](https://user-images.githubusercontent.com/61340340/237011945-ce4f502a-55f6-45e0-ade3-ac74dea45240.PNG)

用一系列的点模拟一条光线，一条光线穿过一个像素，也就是说对一条光线上的每个点，都需要经过一次MLP，在文中提到一条光线粗采样64 个点那么这 64 个点都会经过MLP，也就是会输出 64 个 $\sigma$ ，然后再加入 $\gamma(d)$ ， 注意对这 64 个点来说它们都是处在同一条光线上，所以每个点的 $\gamma(d)$ 都是一样的，然后得到 64 个点对应预测的 rgb 值。

### 第四步 体渲染
\`输入`: 一条光线上的 $c(r, g, b), \sigma$  
输出: 渲染后的 RGB 值

光线的颜色值公式:
$C(r)=\int_{t_n}^{t_f} T(t) \sigma(r(t)) c(r(t), d) dt ; \int T(t)=\exp \left(-\int_{t^n}^t \sigma(r(s)) ds\right), r(t)=o + td$
由于没有办法对连续对每个点进行采样得到积分值，因此引入了它的离散形式，把区间进行划分再进行采样：

$$C(r)=\sum_{i=1}^N T_i\left(1-\exp \left(-\sigma_i \delta_i\right)\right) c_i,where  T_i=\exp \left(-\sum_{j=1}^{i-1} \sigma_i \delta_i\right)$$

其中 $t_i \sim U\left[t_n+\frac{i-1}{N}\left(t_f-t_n\right), t_n+\frac{i}{N}\left(t_f-t_n\right)\right], \delta_i=t_{i+1}-t_i$, $T_i, c_i$ 都是和连续积分公式中采用一致的形式，即透明度和光强，而 $1-\exp \left(-\sigma_i \delta_i\right)$中, $\sigma$ 的含义为体密度，也被称为不透明度或消光系数,实际上不透明度的定义为 $\alpha=1-T(s)=1-\exp \left(-\int_0^s \tau(t) d t\right)$ (即1 - 透明度), 当划分的区间足够小时，可以得到 $\alpha=1-\exp (-\tau s)$ ，其中 $s=\delta, \tau=\sigma$ 。

除此以外，对于离散化的体洹染公式还有另一种理解，注意到在这个吸收发射模型中我们一直用的 是光强 $C$ 这个概念，实际上我们人眼看到的是颜色 $C$ ，在此处个人也有一些疑惑关于两者的联系， 但在这个模型中我们可以认为光强 $C$ 进入人眼所看到的是颜色（作一个这样的理解，从人眼或者说相机出发一条光线经过一段具有某种透明度的物体后击中了某个不透明的物体，这个物体的不透 明度 $\alpha$ ，也就是光线终止在这点的概率，那么眼睛 “看到“ 的就是该物体的颜色 $C$，也因此可以认为透明度是光线穿过这点的概率，不透明度是在这点击中粒子终止的概率) 如果将 $C$ 表示为颜 色，那么论文中离散化的体渲染公式我们可以得到以下的理解:

$$
C(r)=\sum_{i=1}^N T_i \alpha c_i ; T_i=\exp \left(-\sum_{j=1}^{i-1} \sigma_i \delta_i\right), \alpha=\left(1-\exp \left(-\sigma_i \delta_i\right)\right)
$$

alpha blending 在计算机图形学中用于不同透明庵的图像合成，对于具有颜色 $c_f$ 不透明度为 $\alpha$ 的前景，与颜色为 $c_b$ 的背景合成后的颜色为:

$$
c=\alpha c_f+(1-\alpha) c_b
$$

这反映了前景和后景对成像点颜色的贡献，取两个极端值，假设 $\alpha$ 的值为 1 ，即不透明度为 1 ，完全不透明，那么最终成像点的颜色就完全取决于前景的颜色，后景的颜色对成像无贡献， $\alpha$ 为 0 ， 不透明度为 0 ，即完全透明，那么前景的颜色对最终成像的颜色没有贡献。在体洹染公式中，同样可以这样理解，在公式 $\alpha=\left(1-\exp \left(-\sigma_i \delta_i\right)\right)$ 中，体密度 $\delta=0$ ，则 $\alpha=0$ ，即当体密度为 0时，不透明度为0，完全透明，也就是这一段不存在物体，对最终成像的颜色也就没有贡献。体渲染的离散求积公式可以表述为以下形式:

$\begin{aligned} \hat{\mathbf{C}}= & \mathbf{c}_1 \alpha_1 \\ & +\mathbf{c}_2 \alpha_2\left(1-\alpha_1\right) \\ & +\mathbf{c}_3 \alpha_3\left(1-\alpha_1\right)\left(1-\alpha_2\right) \\ & +. \\ & +\mathbf{c}_n \alpha_n\left(1-\alpha_1\right)\left(1-\alpha_2\right) \ldots\left(1-\alpha_{n-1}\right)\end{aligned}$

$T_i=\exp \left(-\sum_{j=1}^{i-1} \sigma_i \delta_i\right)$ 表示前 $\mathrm{i}-1$ 个位置侽积的透明度， $\alpha_i$ 表示第 $\mathrm{i}$ 个位置的不透明度， $c_i$ 是第 $\mathrm{i}$ 个采样点预测出来的颜色，最终成像点的颜色就是根据每个点的颜色贡献 (不透明度) 的叠加，MLP 实现的功能就是预 测每个点的 $c$ 和 $\sigma$ 。这其实也解释了为什么颜色的预测值输出与视角方向有关 (view-dependent)，在不同的视角观察物体，对于同一个物体其在空间中的位置是固定的，也就是体 密度只与位置有关系（采样点的坐标已经统一到世界坐标系下)，而不同的视角代表着不同的光 线，当光线方向改变时，成像的颜色值取决于经过这条光线上的物体，而不同光线经过的物体显然是不一致的。


# Training

 

![llff](https://github.com/PatrioticDedicated/Result/blob/main/gif/llff.gif)
![lego](https://user-images.githubusercontent.com/61340340/236772533-a7d382ab-2155-47f1-8c57-87efa8949ec2.gif)

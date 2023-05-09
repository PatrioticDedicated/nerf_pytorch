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

NeRF [1]用一个 MLP 表示一个静态场景（隐式重建），输入是空间坐标和观察视角，输出是颜色和体密度，经过体渲染后可以得到新视角的合成图像。</p><p data-pid="zdVjlg7v">输入：空间中的坐标 <span class="ztext-math" data-eeimg="1" data-tex="x(x,y,z)">x(x,y,z)</span>, 相机视角方向 <span class="ztext-math" data-eeimg="1" data-tex="d(\theta,\phi)">d(\theta,\phi)</span> </p><p data-pid="MIUEjad0">映射：<span class="ztext-math" data-eeimg="1" data-tex="F_Θ : (x, d) → (c, σ)">F_Θ : (x, d) → (c, σ)</span>，<span class="ztext-math" data-eeimg="1" data-tex="Θ">Θ</span> 是网络的参数</p><p data-pid="Eu3PzCCB">中间输出：颜色 <span class="ztext-math" data-eeimg="1" data-tex="c(r,g,b)">c(r,g,b)</span>，体密度 <span class="ztext-math" data-eeimg="1" data-tex="\sigma">\sigma</span> </p><p data-pid="2JfMezMc">最终输出：体渲染后得到RGB图像</p><figure data-size="normal"><noscript><img src="https://pic1.zhimg.com/v2-69095a4acc5aa356d2b1e8eaf907b160_b.jpg" data-caption="" data-size="normal" data-rawwidth="1139" data-rawheight="347" class="origin_image zh-lightbox-thumb" width="1139" data-original="https://pic1.zhimg.com/v2-69095a4acc5aa356d2b1e8eaf907b160_r.jpg"/></noscript><img src="data:image/svg+xml;utf8,&lt;svg xmlns=&#39;http://www.w3.org/2000/svg&#39; width=&#39;1139&#39; height=&#39;347&#39;&gt;&lt;/svg&gt;" data-caption="" data-size="normal" data-rawwidth="1139" data-rawheight="347" class="origin_image zh-lightbox-thumb lazy" width="1139" data-original="https://pic1.zhimg.com/v2-69095a4acc5aa356d2b1e8eaf907b160_r.jpg" data-actualsrc="https://pic1.zhimg.com/v2-69095a4acc5aa356d2b1e8eaf907b160_b.jpg" data-original-token="v2-886ad8377695f5a1d7730ac01f44381d"/></figure><p class="ztext-empty-paragraph"><br/></p><h3>NeRF 的实现过程：(这里请大家注意每一步的输入输出，会对整体过程有更好的理解）</h3><h3>第一步：光线的生成</h3><p data-pid="NY2tvbwm">         输入：一张RGB图像      输出：图像<b>每个像素</b>的 <span class="ztext-math" data-eeimg="1" data-tex="o(x,y,z),d(x,y,z)">o(x,y,z),d(x,y,z)</span>以及由 <span class="ztext-math" data-eeimg="1" data-tex="o,d">o,d</span> 得到的 <span class="ztext-math" data-eeimg="1" data-tex="x(x,y,z)">x(x,y,z)</span> </p><p data-pid="dVI9SsBB">​

![llff](https://github.com/PatrioticDedicated/Result/blob/main/gif/llff.gif)
![lego](https://user-images.githubusercontent.com/61340340/236772533-a7d382ab-2155-47f1-8c57-87efa8949ec2.gif)

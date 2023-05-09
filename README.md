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
Input: coordinates$\![]https://latex.codecogs.com/svg.image?x(x,y,z)\$ in space, camera Angle direction
Mapping: is a parameter of the networ
Intermediate output: color, volume density
Final output: The volume is rendered to get an RGB image$\x,y$
$\sqrt{3x-1}+(1+x)^2$  $\x$


![llff](https://github.com/PatrioticDedicated/Result/blob/main/gif/llff.gif)
![lego](https://user-images.githubusercontent.com/61340340/236772533-a7d382ab-2155-47f1-8c57-87efa8949ec2.gif)

from torch.utils.data import Dataset
import glob
import os
from PIL import Image
from torchvision import transforms as T

from utils.ray_utils import *
from utils.poses_utils import *

import matplotlib.pyplot as plt
import json

class LLFFDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(504, 378), spheric_poses=False, val_num=1):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.spheric_poses = spheric_poses
        self.val_num = max(1, val_num)  # at least 1
        self.define_transforms()
        self.read_meta()

        self.white_back = False

    def read_meta(self):
        poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds.npy'))  # (N_images, 17)
        self.image_paths = sorted(
            glob.glob(os.path.join(self.root_dir, 'images/*')))  # load full resolution image then resize
        if self.split in ['train', 'val']:
            assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        self.bounds = poses_bounds[:, -2:]  # (N_images, 2)

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1]  # original intrinsics, same for all images
        assert H * self.img_wh[0] == W * self.img_wh[1], \
            f'You must set @img_wh to have the same aspect ratio as ({W}, {H}) !'

        self.focal *= self.img_wh[0] / W

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses)
        distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        val_idx = np.argmin(distances_from_center)  # choose val image as the closest to
        # center image

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.bounds.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal)  # (H, W, 3)

        if self.split == 'train':  # create buffer of all rays and rgb data
            # use first N_images-1 to train, the LAST is val
            self.all_rays = []
            self.all_rgbs = []
            for i, image_path in enumerate(self.image_paths):
                if i == val_idx:  # exclude the val image
                    continue
                c2w = torch.FloatTensor(self.poses[i])

                img = Image.open(image_path).convert('RGB')
                assert img.size[1] * self.img_wh[0] == img.size[0] * self.img_wh[1], \
                    f'''{image_path} has different aspect ratio than img_wh, 
                        please check your data!'''
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (3, h, w)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
                self.all_rgbs += [img]

                rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
                if not self.spheric_poses:
                    near, far = 0, 1
                    rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                                  self.focal, 1.0, rays_o, rays_d)
                    # near plane is always at 1.0
                    # near and far in NDC are always 0 and 1
                    # See https://github.com/bmild/nerf/issues/34
                else:
                    near = self.bounds.min()
                    far = min(8 * near, self.bounds.max())  # focus on central object only

                self.all_rays += [torch.cat([rays_o, rays_d,
                                             near * torch.ones_like(rays_o[:, :1]),
                                             far * torch.ones_like(rays_o[:, :1])],
                                            1)]  # (h*w, 8)


    def define_transforms(self):
        self.transform = T.ToTensor()

class BlenderDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(400, 400)):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()

        self.read_meta()
        self.white_back = True

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=800

        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)


        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        for frame in self.meta['frames'][::self.interval]:
            pose = np.array(frame['transform_matrix'])[:3, :4]
            self.poses += [pose]
            c2w = torch.FloatTensor(pose)

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, h, w)
            img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
            self.all_rgbs += [img]

            rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)

            self.all_rays += [torch.cat([rays_o, rays_d,
                                         self.near*torch.ones_like(rays_o[:, :1]),
                                         self.far*torch.ones_like(rays_o[:, :1])],
                                         1)] # (h*w, 8)

        self.all_rays = torch.cat(self.all_rays, 0).reshape(len(self.meta['frames']),h*w,8)
        self.all_rgbs = torch.cat(self.all_rgbs, 0).reshape(len(self.meta['frames']),h*w,3)

    def define_transforms(self):
        self.transform = T.ToTensor()

    @property
    def interval(self):
        if self.split == "train":
            return 1
        elif self.split == "val" or self.split == "test":
            return 5


def render(rays, N_samples, perturb = 1):
    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # (N_samples)

    # use linear sampling in depth space
    z_vals = near * (1-z_steps) + far * z_steps

    z_vals = z_vals.expand(N_rays, N_samples)

    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)


    return  xyz_coarse_sampled


root_dir = './data/nerf_example_data/nerf_llff_data/fern'
data = LLFFDataset(root_dir)

# root_dir = './data/nerf_example_data/nerf_synthetic/lego'
# data = BlenderDataset(root_dir)


rays =  data.all_rays[0]

ray = render(rays, N_samples = 16)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in ray[::3100]:
    X = i[:,0].numpy()
    Y = i[:,1].numpy()
    Z = i[:,2].numpy()

    ax.scatter(X, Y, Z)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


plt.show()





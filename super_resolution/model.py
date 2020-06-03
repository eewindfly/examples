import torch
import torch.nn as nn
import torch.nn.init as init


# def space_to_depth(x, block_size):
#     n, c, h, w = x.size()
#     unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
#     return unfolded_x.view(n, c * block_size**2, h // block_size,
#                            w // block_size)
class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs,
                   self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2,
                      4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs**2), H // self.bs,
                   W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


class Net(nn.Module):
    def __init__(self,
                 upscale_factor=1,
                 num_img_channel=1,
                 global_residual=False,
                 space_to_depth_block_size=1):
        super(Net, self).__init__()

        self.space_to_depth = SpaceToDepth(space_to_depth_block_size)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(
            num_img_channel * (space_to_depth_block_size**2), 64, (5, 5),
            (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(
            32,
            num_img_channel * (upscale_factor**2) *
            (space_to_depth_block_size**2), (3, 3), (1, 1), (1, 1))
        self.depth_to_space = nn.PixelShuffle(space_to_depth_block_size)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

        self.global_residual = global_residual

    def forward(self, input):
        x = input
        x = self.space_to_depth(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.depth_to_space(x)
        if self.global_residual:
            x = torch.add(input, x)
        output = self.pixel_shuffle(x)

        return output

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

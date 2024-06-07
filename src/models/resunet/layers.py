from torch import nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride = 1):
        super(ResidualBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride = stride, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        )
        self.idt_conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, stride = stride),
        )
        

    def forward(self, x):
        
        return self.res_block(x) + self.idt_conv(x)

class ResUnetEncoder(nn.Module):
    def __init__(self, in_dim, depths):
        super(ResUnetEncoder, self).__init__()
        self.first_res_block = nn.Sequential(
            nn.Conv2d(in_dim, depths[0], kernel_size=3, padding=1, padding_mode = 'reflect'),
            nn.BatchNorm2d(depths[0]),
            nn.ReLU(),
            nn.Conv2d(depths[0], depths[0], kernel_size=3, padding=1, padding_mode = 'reflect')
        )
        self.first_res_cov = nn.Conv2d(in_dim, depths[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList(
            [ResidualBlock(depths[i], depths[i+1], stride = 2) for i in range(len(depths)-1)]
        )

    def forward(self, x):
        #first block
        x_idt = self.first_res_cov(x)
        x = self.first_res_block(x)
        x_0 = x + x_idt

        #encoder blocks
        x = [x_0]
        for i, block in enumerate(self.blocks):
            x.append(block(x[i]))

        return tuple(x)
    
class ResUnetDecoder(nn.Module):
    def __init__(self, depths):
        super(ResUnetDecoder, self).__init__()

        self.dec_blocks = nn.ModuleList(
            [ResidualBlock(depths[i-1] +  depths[i], depths[i-1]) for i in range(1, len(depths))]
        )

        self.up_blocks = nn.ModuleList(
            [nn.Upsample(scale_factor=2) for i in range(1, len(depths))]
        )


    def forward(self, x):
        x_out = x[-1]
        for i in range(len(x)-1, 0, -1):
            x_out = self.up_blocks[i-1](x_out)
            x_out = torch.cat((x_out, x[i-1]), dim=1)
            x_out = self.dec_blocks[i-1](x_out)

        return x_out  

# class IdentityFusion(nn.Module):
#     def __init__(self, depths):
#         super().__init__()
#         self.out_depths = depths

#     def forward(self, x):
#         return x  

class ResunetPoolings(nn.Module):
    def __init__(self, depths) -> None:
        super().__init__()
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(2, 2)
            for _ in range(len(depths))
        ])
        
    def forward(self, x):
        out = [x]
        x_ = x
        for i, pooling in enumerate(self.poolings):
            x_ = pooling(x_)
            out.append(x_)
        return out

class BNIdentity(nn.Module):
    def __init__(self, depths):
        super().__init__()
        self.out_depths = depths

    def forward(self, x):
        return x  

class ResUnetClassifier(nn.Module):
    def __init__(self, depths, n_classes, last_activation = nn.Softmax):
        super(ResUnetClassifier, self).__init__()
        self.res_block = ResidualBlock(depths[0], depths[0])
        self.last_conv = nn.Conv2d(depths[0], n_classes, kernel_size=1)
        self.last_act = last_activation(dim=1)


    def forward(self, x):
        x = self.res_block(x)
        x = self.last_conv(x)
        x = self.last_act(x)
        return x
    

class TemporalConcat(nn.Module):
    def __init__(self, depths):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Conv2d(2*depths[i], depths[i], 1)
            for i in range(len(depths))
        ])

    def forward(self, x):
        x_0, x_1 = x
        out = []
        for i, proj in enumerate(self.projs):
            x_ = torch.cat([x_0[i], x_1[i]], axis=1)
            x_ = proj(x_)
            out.append(x_)
        return out
    
class TemporalConcatPrevMap(nn.Module):
    def __init__(self, depths):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Conv2d(2*depths[i]+1, depths[i], 1)
            for i in range(len(depths))
        ])
        
    def forward(self, x):
        x_0, x_1, x_2 = x
        out = []
        for i, proj in enumerate(self.projs):
            x_ = torch.cat([x_0[i], x_1[i], x_2[i]], axis=1)
            x_ = proj(x_)
            out.append(x_)
        return out    
    
class ModalConcat(nn.Module):
    def __init__(self, depths):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Conv2d(2*depths[i], depths[i], 1)
            for i in range(len(depths))
        ])

    def forward(self, x):
        x_0, x_1 = x
        out = []
        for i, proj in enumerate(self.projs):
            x_ = torch.cat([x_0[i], x_1[i]], axis=1)
            x_ = proj(x_)
            out.append(x_)
        return out
    
class ModalLateConcat(nn.Module):
    def __init__(self, depths):
        super().__init__()
        self.proj= nn.Conv2d(2*depths[0], depths[0], 1)

    def forward(self, x):
        x = torch.cat(x, axis=1)
        x = self.proj(x)
        return x
    
    
class CrossFusion(nn.Module):
    def __init__(self, depths):
        super().__init__()
        self.proj_pre_a= nn.Conv2d(depths[0], depths[0], 1)
        self.proj_pre_b= nn.Conv2d(depths[0], depths[0], 1)
        
        self.proj_o= nn.Conv2d(depths[0], depths[0], 1)
        
        self.proj_s= nn.Conv2d(2*depths[0], depths[0], 1)
        
        self.proj_last= nn.Conv2d(3*depths[0], depths[0], 1)

    def forward(self, x):
        x_0, x_1 = x
        x_pre_0_a = self.proj_pre_a(x_0)
        x_pre_0_b = self.proj_pre_b(x_0)
        x_pre_1_a = self.proj_pre_a(x_1)
        x_pre_1_b = self.proj_pre_b(x_1)
        
        sum_a = x_pre_0_a + x_pre_1_a
        sum_b = x_pre_0_b + x_pre_1_b
        
        concat_sum = torch.cat([self.proj_o(sum_a), self.proj_o(sum_b)], axis=1)
        
        concat_0 = torch.cat([x_pre_0_a, x_pre_0_b], axis=1)
        concat_1 = torch.cat([x_pre_1_a, x_pre_1_b], axis=1)
        
        out = torch.cat([self.proj_s(concat_sum), self.proj_s(concat_0), self.proj_s(concat_1)], axis=1)
        out = self.proj_last(out)
        
        return out
    
# class ResUnetDecoderJF(nn.Module):
#     def __init__(self, depths):
#         super(ResUnetDecoderJF, self).__init__()

#         self.dec_blocks = nn.ModuleList(
#             [ResidualBlock(depths[i-1]) for i in range(1, len(depths)-1)]
#         )
#         self.dec_blocks.append(
#             ResidualBlock(depths[-2])
#         )

#         self.up_blocks = nn.ModuleList(
#             [nn.Upsample(scale_factor=2) for i in range(1, len(depths))]
#         )


#     def forward(self, x):
#         x_out = x[-1]
#         for i in range(len(x)-1, 0, -1):
#             x_out = self.up_blocks[i-1](x_out)
#             x_out = torch.cat((x_out, x[i-1]), dim=1)
#             x_out = self.dec_blocks[i-1](x_out)

#         return x_out  

# class ResUnetDecoderJFNoSkip(nn.Module):
#     def __init__(self, depths):
#         super(ResUnetDecoderJFNoSkip, self).__init__()

#         self.dec_blocks = nn.ModuleList(
#             [ResidualBlock(depths[i-1]) for i in range(1, len(depths)-1)]
#         )
#         self.dec_blocks.append(
#             ResidualBlock(depths[-2])
#         )

#         self.up_blocks = nn.ModuleList(
#             [nn.Upsample(scale_factor=2) for i in range(1, len(depths))]
#         )


#     def forward(self, x):
#         x_out = x
#         for i in range(len(self.up_blocks)-1, -1, -1):
#             x_out = self.up_blocks[i](x_out)
#             x_out = self.dec_blocks[i](x_out)

#         return x_out  

# class ResUnetClassifier(nn.Module):
#     def __init__(self, depths, n_classes, last_activation = nn.Softmax):
#         super(ResUnetClassifier, self).__init__()
#         self.res_block = ResidualBlock(depths[0], depths[0])
#         self.last_conv = nn.Conv2d(depths[0], n_classes, kernel_size=1)
#         self.last_act = last_activation(dim=1)


#     def forward(self, x):
#         x = self.res_block(x)
#         x = self.last_conv(x)
#         x = self.last_act(x)
#         return x
    
# class ResUnetRegressionClassifier(nn.Module):
#     def __init__(self, depth):
#         super().__init__()
#         self.res_block = ResidualBlock(depth)
#         self.last_conv = nn.LazyConv2d(2, kernel_size=1)
#         self.last_act = nn.Sigmoid()

#     def forward(self, x):
#         x = self.res_block(x)
#         x = self.last_conv(x)
#         x = self.last_act(x)
#         return x
from tqdm import trange
import torch
import numpy
class IntegratedGradients():
    def __init__(self, model):
        model.eval()
        model.to('cuda:0')
        self.model = model
        
    def attributes(self, x, target=None):
        mean_grad = 0
        
        baselines = []
        for feat in x:
            baselines.append(torch.zeros_like(feat))
        self.model.eval()
        n = 50
        for dx in trange(0, 50+1, leave=False):
            x_ = []
            for feat_i, baseline_i in zip(x, baselines):
                x_i = baseline_i + torch.tensor(dx) * feat_i
                x_i.requires_grad = True
                x_i.to('cuda:0')
                x_.append(x_i)
            self.model.zero_grad()
            with torch.autograd.set_grad_enabled(True):
                self.model.eval()
                y = self.model(x_)
                grad, _ = torch.autograd.grad(y, x_, grad_outputs=[torch.ones(x_[0].shape), [torch.ones(x_[1].shape)]])
        
       
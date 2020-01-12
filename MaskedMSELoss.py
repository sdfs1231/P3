import torch
import torch.nn as nn

class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, input, target, mask):
        # diff2 = (torch.flatten(input) - torch.flatten(target)) ** 2.0
        mask = torch.reshape(mask,(input.shape[0],1))

        self.loss = self.criterion(input*mask,target*mask)

        return self.loss

# diff2 = (torch.flatten(input) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
# result = torch.sum(diff2) / torch.sum(mask)
# return result
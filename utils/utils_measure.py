from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# from dataloader import get_dataloaders

import math
import numbers
import itertools

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import cv2

import copy
import logging
from matplotlib import gridspec
import torch 
    
from PIL import Image
    

def normalize(x, dim=-1):
    norm1 = x / np.linalg.norm(x, axis=dim, keepdims=True)
    return norm1

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def _from_dense(num_timesteps, num_gt_ids, num_tracker_ids, gt_present, tracker_present, similarity):
    gt_subset = [np.flatnonzero(gt_present[t, :]) for t in range(num_timesteps)]
    tracker_subset = [np.flatnonzero(tracker_present[t, :]) for t in range(num_timesteps)]
    similarity_subset = [
            similarity[t][gt_subset[t], :][:, tracker_subset[t]]
            for t in range(num_timesteps)
    ]
    data = {
            'num_timesteps': num_timesteps,
            'num_gt_ids': num_gt_ids,
            'num_tracker_ids': num_tracker_ids,
            'num_gt_dets': np.sum(gt_present),
            'num_tracker_dets': np.sum(tracker_present),
            'gt_ids': gt_subset,
            'tracker_ids': tracker_subset,
            'similarity_scores': similarity_subset,
    }
    return data

# from __future__ import absolute_import
# import torch
# import torch.nn as nn
# import numpy as np
# # import matplotlib.pyplot as plt
# import os
# import sys
# # from dataloader import get_dataloaders

# import math
# import numbers
# import itertools

# import torch
# import torch.optim as optim
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# from torch.utils.data import DataLoader
# import torch.nn.functional as F

# import cv2

# import copy
# import logging
# # from matplotlib import gridspec
# import torch 
    
# from PIL import Image
    

# def normalize(x, dim=-1):
#     norm1 = x / np.linalg.norm(x, axis=dim, keepdims=True)
#     return norm1


# def to_video(image_folder, video_folder, video_name, downsample = 2):
    
#     images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
#     images = np.sort(np.array(images, dtype="object"))

#     frame = cv2.imread(os.path.join(image_folder, images[0]))
#     height, width, layers = frame.shape

#     height = int(float(height)/downsample)
#     width = int(float(width)/downsample)


#     video = cv2.VideoWriter(video_folder + "/" + video_name + ".avi", 0, 1, (width,height))

#     i = 0
#     for image in images:
#         video.write(cv2.resize( cv2.imread(os.path.join(image_folder, image)), (width,height) )   )
#         i += 1

#     video.release()
    
    
    
# def weights_init_normal(m):
#     classname = m.__class__.__name__
    
#     if classname=="Conv2d":
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
# #         torch.nn.init.xavier_uniform(m.weight.data)  
# #     elif classname.find("ConvTranspose2d") != -1:
# #         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find("BatchNorm2d") != -1:
#         try:
#             torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#             torch.nn.init.constant_(m.bias.data, 0.0)
#         except:
#             pass
        
        
        
        
# class GaussianSmoothing(nn.Module):
#     """
#     Apply gaussian smoothing on a
#     1d, 2d or 3d tensor. Filtering is performed seperately for each channel
#     in the input using a depthwise convolution.
#     Arguments:
#         channels (int, sequence): Number of channels of the input tensors. Output will
#             have this number of channels as well.
#         kernel_size (int, sequence): Size of the gaussian kernel.
#         sigma (float, sequence): Standard deviation of the gaussian kernel.
#         dim (int, optional): The number of dimensions of the data.
#             Default value is 2 (spatial).
#     """
#     def __init__(self, channels, kernel_size, sigma, dim=2):
#         super(GaussianSmoothing, self).__init__()
#         if isinstance(kernel_size, numbers.Number):
#             kernel_size = [kernel_size] * dim
#         if isinstance(sigma, numbers.Number):
#             sigma = [sigma] * dim

#         # The gaussian kernel is the product of the
#         # gaussian function of each dimension.
#         kernel = 1
#         meshgrids = torch.meshgrid(
#             [
#                 torch.arange(size, dtype=torch.float32)
#                 for size in kernel_size
#             ]
#         )
#         for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
#             mean = (size - 1) / 2
#             kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
#                       torch.exp(-((mgrid - mean) / std) ** 2 / 2)

#         # Make sure sum of values in gaussian kernel equals 1.
#         kernel = kernel / torch.sum(kernel)

#         # Reshape to depthwise convolutional weight
#         kernel = kernel.view(1, 1, *kernel.size())
#         kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

#         self.register_buffer('weight', kernel)
#         self.groups = channels

#         if dim == 1:
#             self.conv = F.conv1d
#         elif dim == 2:
#             self.conv = F.conv2d
#         elif dim == 3:
#             self.conv = F.conv3d
#         else:
#             raise RuntimeError(
#                 'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
#             )

#     def forward(self, input):
#         """
#         Apply gaussian filter to input.
#         Arguments:
#             input (torch.Tensor): Input to apply gaussian filter on.
#         Returns:
#             filtered (torch.Tensor): Filtered output.
#         """
#         return self.conv(input, weight=self.weight, groups=self.groups)
    
    
    
# class LabelSmoothing(nn.Module):
#     """
#     NLL loss with label smoothing.
#     """
#     def __init__(self, smoothing=0.0):
#         """
#         Constructor for the LabelSmoothing module.
#         :param smoothing: label smoothing factor
#         """
#         super(LabelSmoothing, self).__init__()
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing

#     def forward(self, x, target):
#         logprobs = torch.nn.functional.log_softmax(x, dim=-1)

#         nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
#         nll_loss = nll_loss.squeeze(1)
#         smooth_loss = -logprobs.mean(dim=-1)
#         loss = self.confidence * nll_loss + self.smoothing * smooth_loss
#         return loss.mean()
    

# class BCEWithLogitsLoss(nn.Module):
#     def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None, num_classes=64):
#         super(BCEWithLogitsLoss, self).__init__()
#         self.num_classes = num_classes
#         self.criterion = nn.BCEWithLogitsLoss(weight=weight, 
#                                               size_average=size_average, 
#                                               reduce=reduce, 
#                                               reduction=reduction,
#                                               pos_weight=pos_weight)
#     def forward(self, input, target):
#         target_onehot = F.one_hot(target, num_classes=self.num_classes)
#         return self.criterion(input, target_onehot)
    

# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


# def adjust_learning_rate(epoch, opt, optimizer):
#     """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
# #     steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
#     if epoch in opt.lr_decay_epochs:
#         steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
        
#         new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = new_lr


# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res

    
    
    
# class Logger(object):
#     '''Save training process to log file with simple plot function.'''
#     def __init__(self, fpath, title=None, resume=False): 
#         self.file = None
#         self.resume = resume
#         self.title = '' if title == None else title
#         if fpath is not None:
#             if resume: 
#                 self.file = open(fpath, 'r') 
#                 name = self.file.readline()
#                 self.names = name.rstrip().split('\t')
#                 self.numbers = {}
#                 for _, name in enumerate(self.names):
#                     self.numbers[name] = []

#                 for numbers in self.file:
#                     numbers = numbers.rstrip().split('\t')
#                     for i in range(0, len(numbers)):
#                         self.numbers[self.names[i]].append(numbers[i])
#                 self.file.close()
#                 self.file = open(fpath, 'a')  
#             else:
#                 self.file = open(fpath, 'w')

#     def set_names(self, names):
#         if self.resume: 
#             pass
#         # initialize numbers as empty list
#         self.numbers = {}
#         self.names = names
#         for _, name in enumerate(self.names):
#             self.file.write(name)
#             self.file.write('\t')
#             self.numbers[name] = []
#         self.file.write('\n')
#         self.file.flush()


#     def append(self, numbers):
#         assert len(self.names) == len(numbers), 'Numbers do not match names'
#         for index, num in enumerate(numbers):
#             self.file.write("{0:.6f}".format(num))
#             self.file.write('\t')
#             self.numbers[self.names[index]].append(num)
#         self.file.write('\n')
#         self.file.flush()

#     def plot(self, names=None):   
#         names = self.names if names == None else names
#         numbers = self.numbers
#         for _, name in enumerate(names):
#             x = np.arange(len(numbers[name]))
#             plt.plot(x, np.asarray(numbers[name]))
#         plt.legend([self.title + '(' + name + ')' for name in names])
#         plt.grid(True)
        

#     def close(self):
#         if self.file is not None:
#             self.file.close()
            
            
            
            
            
            
# # def generate_final_report(model, opt, wandb):
# #     from eval.meta_eval import meta_test
    
# #     opt.n_shots = 1
# #     train_loader, val_loader, meta_testloader, meta_valloader, _ = get_dataloaders(opt)
    
# #     #validate
# #     meta_val_acc, meta_val_std = meta_test(model, meta_valloader)
    
# #     meta_val_acc_feat, meta_val_std_feat = meta_test(model, meta_valloader, use_logit=False)

# #     #evaluate
# #     meta_test_acc, meta_test_std = meta_test(model, meta_testloader)
    
# #     meta_test_acc_feat, meta_test_std_feat = meta_test(model, meta_testloader, use_logit=False)
        
# #     print('Meta Val Acc : {:.4f}, Meta Val std: {:.4f}'.format(meta_val_acc, meta_val_std))
# #     print('Meta Val Acc (feat): {:.4f}, Meta Val std (feat): {:.4f}'.format(meta_val_acc_feat, meta_val_std_feat))
# #     print('Meta Test Acc: {:.4f}, Meta Test std: {:.4f}'.format(meta_test_acc, meta_test_std))
# #     print('Meta Test Acc (feat): {:.4f}, Meta Test std (feat): {:.4f}'.format(meta_test_acc_feat, meta_test_std_feat))
    
    
# #     wandb.log({'Final Meta Test Acc @1': meta_test_acc,
# #                'Final Meta Test std @1': meta_test_std,
# #                'Final Meta Test Acc  (feat) @1': meta_test_acc_feat,
# #                'Final Meta Test std  (feat) @1': meta_test_std_feat,
# #                'Final Meta Val Acc @1': meta_val_acc,
# #                'Final Meta Val std @1': meta_val_std,
# #                'Final Meta Val Acc   (feat) @1': meta_val_acc_feat,
# #                'Final Meta Val std   (feat) @1': meta_val_std_feat
# #               })

    
# #     opt.n_shots = 5
# #     train_loader, val_loader, meta_testloader, meta_valloader, _ = get_dataloaders(opt)
    
# #     #validate
# #     meta_val_acc, meta_val_std = meta_test(model, meta_valloader)
    
# #     meta_val_acc_feat, meta_val_std_feat = meta_test(model, meta_valloader, use_logit=False)

# #     #evaluate
# #     meta_test_acc, meta_test_std = meta_test(model, meta_testloader)
    
# #     meta_test_acc_feat, meta_test_std_feat = meta_test(model, meta_testloader, use_logit=False)
        
# #     print('Meta Val Acc : {:.4f}, Meta Val std: {:.4f}'.format(meta_val_acc, meta_val_std))
# #     print('Meta Val Acc (feat): {:.4f}, Meta Val std (feat): {:.4f}'.format(meta_val_acc_feat, meta_val_std_feat))
# #     print('Meta Test Acc: {:.4f}, Meta Test std: {:.4f}'.format(meta_test_acc, meta_test_std))
# #     print('Meta Test Acc (feat): {:.4f}, Meta Test std (feat): {:.4f}'.format(meta_test_acc_feat, meta_test_std_feat))

# #     wandb.log({'Final Meta Test Acc @5': meta_test_acc,
# #                'Final Meta Test std @5': meta_test_std,
# #                'Final Meta Test Acc  (feat) @5': meta_test_acc_feat,
# #                'Final Meta Test std  (feat) @5': meta_test_std_feat,
# #                'Final Meta Val Acc @5': meta_val_acc,
# #                'Final Meta Val std @5': meta_val_std,
# #                'Final Meta Val Acc   (feat) @5': meta_val_acc_feat,
# #                'Final Meta Val std   (feat) @5': meta_val_std_feat
# #               })





# def plot_attention(attn_input, input_images, ids, plot_name):
    
    

#     logger = logging.getLogger()
#     old_level = logger.level
#     logger.setLevel(100)


#     mean = np.array([123.675, 116.280, 103.530])
#     std = np.array([58.395, 57.120, 57.375])



#     T = 20
#     person = 40*9 + 1
#     n_heads = attn_input[0].shape[1]


#     nrow = 1 + n_heads*3
#     ncol = 50

#     fig = plt.figure(figsize=(50, nrow+4))
#     gs = gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.0, top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1),  left=0.5/(ncol+1), right=1-0.5/(ncol+1))  #width_ratios=[1, 1, 1], 



#     r = 0
#     for att_i, att_name in enumerate(["att_bbox", "att_wa", "att_all"]):

#         # att_heads = np.load(att_name+ ".npy")
#         att_heads = attn_input[att_i]

#         for head in range(n_heads):
#             att = att_heads[0, head]

#             att_p = att[person]
#             att_p_norm = (att_p - np.min(att_p))/np.ptp(att_p)*0.6


#             c = 0
#             for t in range(T):

#                 loc_ = np.where(ids[t]!=-1)[0]
#                 for p in range(len(loc_)):

#                     try:

#                         if(r==0):
#                             image_np = input_images[t, loc_[p]] * torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1) 
#                             image_np = image_np + torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1)
#                             image_np_ = copy.deepcopy(image_np)
#                             image_np_[ 0, :, :] = image_np[ 2, :, :]
#                             image_np_[ 1, :, :] = image_np[ 1, :, :]
#                             image_np_[ 2, :, :] = image_np[ 0, :, :]
#                             image_np_ = image_np_.numpy()
#                             image_np_ = np.transpose(image_np_, (1, 2, 0))

#                             ax = plt.subplot(gs[r,c])
#                             ax.set_xlabel("t="+str(t)+", id="+str(int(ids[t][loc_[p]].numpy())))
#                             ax.set_xticks([])
#                             ax.set_yticks([])

#                             if(person==t*40+p):
#                                 cood = ax.get_position()
#                                 rect = plt.Rectangle(
#                                     # (lower-left corner), width, height
#                                     (cood.x0, cood.y0), 1.0*(cood.x1-cood.x0), 1.0*(cood.y1-cood.y0), fill=False, color='r', lw=5, 
#                                     zorder=1000, transform=fig.transFigure, figure=fig
#                                 )
#                                 ax.patches.extend([rect])

#                             ax.imshow(image_np_)



#                         attention = np.ones_like(image_np_) * att_p_norm[t*40+p]
#                         if(att_name=="att_bbox"):
#                             attention[:, :, 1:] = 0
#                         if(att_name=="att_wa"):
#                             attention[:, :, :2] = 0
#                         if(att_name=="att_all"):
#                             attention[:, :, 0] = 0
#                             attention[:, :, 2] = 0

#                         ax = plt.subplot(gs[r+1,c])
#                         ax.axis('off')
#                         ax.imshow(attention)
#                         ax.imshow(attention)



#                         c += 1
#                     except Exception as e:
#                         print(e)
#             r += 1


#     # plt.show()
#     plt.savefig("/home/jathu/3D/HMR_tracking/plots/"+plot_name+".png", bbox_inches='tight')
#     print("XXXXXXXXXX   saved")
#     image = Image.open("/home/jathu/3D/HMR_tracking/plots/"+plot_name+".png")
    
#     return image

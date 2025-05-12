import torch
import numpy as np
def create_optimizer(model, args):
    """
    创建优化器，根据用户传递的参数来选择合适的优化器和学习率。
    
    参数:
        model (torch.nn.Module): 要优化的模型。
        args: 包含超参数的对象，应该包括优化器类型、学习率、权重衰减等信息。

    返回:
        optimizer (torch.optim.Optimizer): 创建的优化器。
    """
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {args.optimizer}")
    
    return optimizer

def adjust_learning_rate(optimizer, epoch, args):
    """
    根据当前的epoch动态调整学习率。
    
    参数:
        optimizer (torch.optim.Optimizer): 优化器。
        epoch (int): 当前的epoch。
        args: 包含超参数的对象，应该包括初始学习率和学习率调整策略。

    返回:
        None
    """
    if args.lr_scheduler == 'step':
        # 每隔step_size个epoch，学习率衰减gamma倍
        if epoch % args.step_size == 0 and epoch > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.gamma
            print(f"Learning rate adjusted to {optimizer.param_groups[0]['lr']} at epoch {epoch}")
    elif args.lr_scheduler == 'cosine':
        # 余弦退火调度器
        lr = 0.5 * args.lr * (1 + np.cos(np.pi * epoch / args.epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f"Cosine annealing adjusted learning rate to {lr:.6f} at epoch {epoch}")
    else:
        raise ValueError(f"Unsupported learning rate scheduler: {args.lr_scheduler}")
import torch
import torch.nn.functional as F

from config import config

def adjust_learning_rate(scheduler, epoch, step, num_steps_per_epoch,
                     warmup_lr_epochs=0, schedule_lr_per_epoch=False, size=1):
    if epoch < warmup_lr_epochs:
        epoch += step / num_steps_per_epoch
        factor = (epoch * (size - 1) / warmup_lr_epochs + 1) / size
        for param_group, base_lr in zip(scheduler.optimizer.param_groups,
                                        scheduler.base_lrs):
            param_group['lr'] = base_lr * factor
    elif schedule_lr_per_epoch and (step > 0 or epoch == 0):
        return
    elif epoch == warmup_lr_epochs and step == 0:
        for param_group, base_lr in zip(scheduler.optimizer.param_groups,
                                        scheduler.base_lrs):
            param_group['lr'] = base_lr
        return
    else:
        scheduler.step()


def logprint(message, logfile):
    print(message)
    print(message, file=logfile)


def test_model(model, valid_loader, logfile):
    model.eval()

    all_true = []
    all_pred = []

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for features, target in valid_loader:
            features, target = features.to(config.device), target.to(config.device)

            output = model(features)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_true.extend(target.tolist())
            all_pred.extend(pred.tolist())

    total = len(valid_loader.dataset)
    test_loss /= total

    percent_correct = 100. * correct / total

    report = (
            f"Test set: "
            f"Average loss: {test_loss:.4f} "
            f"Accuracy: {correct}/{total} ({percent_correct:.2f}%)"
            )
    
    logprint(report, logfile)



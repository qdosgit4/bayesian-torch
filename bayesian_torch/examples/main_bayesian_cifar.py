import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import bayesian_torch.models.bayesian.resnet_variational as resnet
import numpy as np


##  Extract model names from bayesian-torch library.

model_names = sorted(
    name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
    and name.startswith("resnet") and callable(resnet.__dict__[name]))


##  Quantify the training and test set numbers.

print(model_names)
len_trainset = 50000
len_testset = 10000

##  CIFAR-10:  50000 training, 10000 testing.
##             6000 images per class.

##  CIFAR-100: 50000 training, 10000 testing.
##             600 images per class.


##  Argument parser designer for the CIFAR10 dataset.

parser = argparse.ArgumentParser(description='CIFAR10')


##  Choose resnet model.

parser.add_argument('--arch',
                    '-a',
                    metavar='ARCH',
                    default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet20)')


##  Subprocesses to use for data loading:
##  https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

parser.add_argument('-j',
                    '--workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 8)')


##  Quantity of times full dataset is run through model.

parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')


##  Quantity of samples per training batch.
##  https://docs.pytorch.org/docs/stable/data.html

parser.add_argument('-b',
                    '--batch-size',
                    default=128,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 128)')


##  Hyperparameter, frequently depicted as alpha in algebra.
##  Rate at which gradient descent occurs.

parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.001,
                    type=float,
                    metavar='LR',
                    help='initial learning rate')


##  Not covered by course; accelerates learning rate somehow.
##  https://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')


##  Similar to L2 regularisation, adds a value to the loss function to
##  prevent overfitting.

parser.add_argument('--weight-decay',
                    '--wd',
                    default=5e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 5e-4)')

parser.add_argument('--print-freq',
                    '-p',
                    default=50,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 20)')


##  A checkpoint is a state of the set of model parameters, which is
##  reached via training (parameters may be iterated to any value).

parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')


##  The validation stage is about testing if the hyperparameters of
##  the model have let to an overfit.

parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')


##  Probably: load model weights.

parser.add_argument('--pretrained',
                    dest='pretrained',
                    action='store_true',
                    help='use pre-trained model')


##  Probably: store weights as Float16.

parser.add_argument('--half',
                    dest='half',
                    action='store_true',
                    help='use half-precision(16-bit) ')


##  Directory for storing weights, common activity when training
##  models.

parser.add_argument('--save-dir',
                    dest='save_dir',
                    help='The directory used to save the trained models',
                    default='./checkpoint/bayesian',
                    type=str)


##  Store state of weights are n training iterations.

parser.add_argument(
    '--save-every',
    dest='save_every',
    help='Saves checkpoints at every specified number of epochs',
    type=int,
    default=10)


##  Toggle between training and testing mode.

parser.add_argument('--mode', type=str, required=True, help='train | test')


##  Samples of weights are drawn via Monte Carlo from the
##  approximation function of the posterior; this is seemingly how the
##  approximation function is iterated.

parser.add_argument(
    '--num_monte_carlo',
    type=int,
    default=20,
    metavar='N',
    help='number of Monte Carlo samples to be drawn during inference')

parser.add_argument('--num_mc',
                    type=int,
                    default=1,
                    metavar='N',
                    help='number of Monte Carlo runs during training')


##  Library from TensorFlow for visualising training (e.g. loss
##  function output changes).

parser.add_argument(
    '--tensorboard',
    type=bool,
    default=True,
    metavar='N',
    help='use tensorboard for logging and visualization of training progress')

parser.add_argument(
    '--log_dir',
    type=str,
    default='./logs/cifar/bayesian',
    metavar='N',
    help='use tensorboard for logging and visualization of training progress')

best_prec1 = 0


##  MOPED is a means of selecting the priors.

##  Surrogate posterior is the approximation function of the posterior
##  (surrogate means substitution).

##  Note that this function is only called in the _dnn2bnn equivalent
##  version of this file, and involves using a trained DNN as a
##  starting point.

def MOPED_layer(layer, det_layer, delta):
    """
    Set the priors and initialize surrogate posteriors of Bayesian NN with Empirical Bayes
    MOPED (Model Priors with Empirical Bayes using Deterministic DNN)
    Reference:
    [1] Ranganath Krishnan, Mahesh Subedar, Omesh Tickoo.
        Specifying Weight Priors in Bayesian Deep Neural Networks with Empirical Bayes. AAAI 2020.
    """

    if (str(layer) == 'Conv2dReparameterization()'):
        #set the priors
        print(str(layer))
        layer.prior_weight_mu = det_layer.weight.data
        if layer.prior_bias_mu is not None:
            layer.prior_bias_mu = det_layer.bias.data

        #initialize surrogate posteriors
        layer.mu_kernel.data = det_layer.weight.data
        layer.rho_kernel.data = get_rho(det_layer.weight.data, delta)
        if layer.mu_bias is not None:
            layer.mu_bias.data = det_layer.bias.data
            layer.rho_bias.data = get_rho(det_layer.bias.data, delta)

    elif (isinstance(layer, nn.Conv2d)):
        print(str(layer))
        layer.weight.data = det_layer.weight.data
        if layer.bias is not None:
            layer.bias.data = det_layer.bias.data

    elif (str(layer) == 'LinearReparameterization()'):
        print(str(layer))
        layer.prior_weight_mu = det_layer.weight.data
        if layer.prior_bias_mu is not None:
            layer.prior_bias_mu = det_layer.bias.data

        #initialize the surrogate posteriors
        layer.mu_weight.data = det_layer.weight.data
        layer.rho_weight.data = get_rho(det_layer.weight.data, delta)
        if layer.mu_bias is not None:
            layer.mu_bias.data = det_layer.bias.data
            layer.rho_bias.data = get_rho(det_layer.bias.data, delta)

    elif str(layer).startswith('Batch'):
        #initialize parameters
        print(str(layer))
        layer.weight.data = det_layer.weight.data
        if layer.bias is not None:
            layer.bias.data = det_layer.bias.data
        layer.running_mean.data = det_layer.running_mean.data
        layer.running_var.data = det_layer.running_var.data
        layer.num_batches_tracked.data = det_layer.num_batches_tracked.data


def main():
    global args, best_prec1
    args = parser.parse_args()
 
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ##  DataParallel duplicates the model and runs forward and
    ##  backwards propagation in parallel, for multiple training data
    ##  instances.
    ##  https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    ##  Instruct Nvidia's benchmarking algorithms to run tests to
    ##  deduce most efficient means of handling specific model.
    ##  Provies a modest speedup, allegedly.
            
    cudnn.benchmark = True

    ##  TensorBoard is a visualisation tool for ML.

    tb_writer = None
    if args.tensorboard:
        logger_dir = os.path.join(args.log_dir, 'tb_logger')
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)
        tb_writer = SummaryWriter(logger_dir)

    ##  Setup overlay for modifying the training data. This is
    ##  equivalent to the classical statistics equation:
    ##  z = (x - mu) / sigma

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    ##  Load CIFAR10 dataset, making some random adjustments.

    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
        root='./data',
        train=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]),
        download=True),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    ##  Load CIFAR10 dataset, for validation.

    val_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
        root='./data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ##  Setup loss function.

    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cpu()

    ##  Set FP16 data types.

    if args.half:
        model.half()
        criterion.half()

    ##  Set learning rate.

    if args.arch in ['resnet110']:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.1

    ##  Validate dataset, model, and loss function.

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    

    if args.mode == 'train':

        for epoch in range(args.start_epoch, args.epochs):

            ##  Set learning rate based on quantity of iterations.

            lr = args.lr
            if (epoch >= 80 and epoch < 120):
                lr = 0.1 * args.lr
            elif (epoch >= 120 and epoch < 160):
                lr = 0.01 * args.lr
            elif (epoch >= 160 and epoch < 180):
                lr = 0.001 * args.lr
            elif (epoch >= 180):
                lr = 0.0005 * args.lr

            ##  Adam algorithm.

            optimizer = torch.optim.Adam(model.parameters(), lr)

            ##  Train model using full dataset.

            # train for one epoch
            print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
            train(args, train_loader, model, criterion, optimizer, epoch,
                  tb_writer)

            ##  Validate model (after training).

            prec1 = validate(args, val_loader, model, criterion, epoch,
                             tb_writer)

            ##  Compare to preloaded checkpoint value or 0.

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            ##  If the latest epoch led to best validate() output,
            ##  store checkpoint.

            if is_best:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                    },
                    is_best,
                    filename=os.path.join(
                        args.save_dir,
                        'bayesian_{}_cifar.pth'.format(args.arch)))


    ##  If testing, load from checkpoint, run evaluate().

    elif args.mode == 'test':
        checkpoint_file = args.save_dir + '/bayesian_{}_cifar.pth'.format(
            args.arch)
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_file)
        else:
            checkpoint = torch.load(checkpoint_file,
                                    map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        evaluate(args, model, val_loader)


def train(args,
          train_loader,
          model,
          criterion,
          optimizer,
          epoch,
          tb_writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    ##  Note that this function is called multiples times, such that
    ##  the weights are gradually updated.

    # switch to train mode
    model.train()

    ##  Begin timer.

    end = time.time()

    ##  Iterate through training dataset.
    
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        ##  Associate training data (x, y) tuple with hardware
        ##  available.

        if torch.cuda.is_available():
            target = target.cuda()
            input_var = input.cuda()
            target_var = target
        else:
            target = target.cpu()
            input_var = input.cpu()
            target_var = target

        ##  Setup FP16 number data type.

        if args.half:
            input_var = input_var.half()

        # compute output
        output_ = []
        kl_ = []

        ##  Begin MC iterations.
        
        for mc_run in range(args.num_mc):

            ##  Run data through model via model's forward(), get
            ##  output.
            
            output, kl = model(input_var)

            ##  Build list of all output.
            
            output_.append(output)

            ##  Build list of all KL loss.
            
            kl_.append(kl)

        ##  Find mean of model output and KL loss.

        ##  Note that torch.mean only outputs one value, but in tensor
        ##  format.

        output = torch.mean(torch.stack(output_), dim=0)
        kl = torch.mean(torch.stack(kl_), dim=0)

        ##  Compare model output and target using cross entropy loss.

        cross_entropy_loss = criterion(output, target_var)
        scaled_kl = kl / args.batch_size

        ##  Summate loss function outputs (allegedly ELBO).
        ##  Note that `loss` is a tensor object.

        loss = cross_entropy_loss + scaled_kl

        # compute gradient and do SGD step
        optimizer.zero_grad()

        ##  The graph (as in graph theory, whereby a function is
        ##  treated as a graph) is differentiated via the chain rule.
        ##  Results are stored in the leaves.
        ##  Note that `loss` is a tensor object.

        loss.backward()

        ##  Run Adam algorithm, which is relevant to stochastic loss
        ##  functions (i.e. MC variational inference).

        ##  The Adam algorithm ultimately uses the following to
        ##  choose the next weight set:
        ##    - change in output of loss function in respect to
        ##      previous set
        ##    - hyperparameters beta_1, beta_2

        ##  https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html
        
        optimizer.step()

        ##  Convert to simpler data type.

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss

        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1))

        if tb_writer is not None:
            tb_writer.add_scalar('train/cross_entropy_loss',
                                 cross_entropy_loss.item(), epoch)
            tb_writer.add_scalar('train/kl_div', scaled_kl.item(), epoch)
            tb_writer.add_scalar('train/elbo_loss', loss.item(), epoch)
            tb_writer.add_scalar('train/accuracy', prec1.item(), epoch)
            tb_writer.flush()


def validate(args, val_loader, model, criterion, epoch, tb_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    ##  Disable gradient calculation (backward() will become
    ##  unavailable).

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                target = target.cuda()
                input_var = input.cuda()
                target_var = target.cuda()
            else:
                target = target.cpu()
                input_var = input.cpu()
                target_var = target.cpu()

            if args.half:
                input_var = input_var.half()

            ##  Quantity of MC iterations for selecting new set of
            ##  weights.

            # compute output
            output_ = []
            kl_ = []
            for mc_run in range(args.num_mc):
                output, kl = model(input_var)
                output_.append(output)
                kl_.append(kl)

            ##  Equivalent process to training.
                
            output = torch.mean(torch.stack(output_), dim=0)
            kl = torch.mean(torch.stack(kl_), dim=0)
            cross_entropy_loss = criterion(output, target_var)
            scaled_kl = kl / args.batch_size 
            #ELBO loss
            loss = cross_entropy_loss + scaled_kl

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i,
                          len(val_loader),
                          batch_time=batch_time,
                          loss=losses,
                          top1=top1))

            if tb_writer is not None:
                tb_writer.add_scalar('val/cross_entropy_loss',
                                     cross_entropy_loss.item(), epoch)
                tb_writer.add_scalar('val/kl_div', scaled_kl.item(), epoch)
                tb_writer.add_scalar('val/elbo_loss', loss.item(), epoch)
                tb_writer.add_scalar('val/accuracy', prec1.item(), epoch)
                tb_writer.flush()

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    ##  Return...?

    return top1.avg


def evaluate(args, model, val_loader):
    pred_probs_mc = []
    test_loss = 0
    correct = 0
    output_list = []
    labels_list = []
    model.eval()

    ##  Disable gradient calculation.
    
    with torch.no_grad():
        begin = time.time()
        for data, target in val_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data.cpu(), target.cpu()
            output_mc = []
            for mc_run in range(args.num_monte_carlo):
                output, _ = model.forward(data)
                output_mc.append(output)
            output_ = torch.stack(output_mc)
            output_list.append(output_)
            labels_list.append(target)
        end = time.time()
        print("inference throughput: ", len_testset / (end - begin),
              " images/s")

        ##  Build new tensor.

        output = torch.stack(output_list)

        ##  Adjust arrangement.
        
        output = output.permute(1, 0, 2, 3)

        ##  Create contiguous piece of memory for tensor, adjust
        ##  shape.
        
        output = output.contiguous().view(args.num_monte_carlo, len_testset,
                                          -1)

        ##  Run softmax function.
        
        output = torch.nn.functional.softmax(output, dim=2)

        ##  Concatenate labels.
        
        labels = torch.cat(labels_list)

        ##  Calculate mean.
        
        pred_mean = output.mean(dim=0)

        ##  Find maximum element within pred_mean.
        
        Y_pred = torch.argmax(pred_mean, axis=1)

        print('Test accuracy:',
              (Y_pred.data.cpu().numpy() == labels.data.cpu().numpy()).mean() *
              100)
        np.save('./probs_cifar_mc.npy', output.data.cpu().numpy())
        np.save('./cifar_test_labels_mc.npy', labels.data.cpu().numpy())


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


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


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

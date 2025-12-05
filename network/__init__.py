"""
Network Initializations
"""

import logging
import importlib
import torch
import datasets



def get_net(args):
    """
    Get Network Architecture based on arguments provided
    """
    # net = get_model(args=args, num_classes=datasets.num_classes)
    enc, dec = get_model(args=args, num_classes=datasets.num_classes)
    # num_params = sum([param.nelement() for param in net.parameters()])
    num_params = sum([param.nelement() for param in enc.parameters()] + [param.nelement() for param in dec.parameters()])
    logging.info('Model params = {:2.3f}M'.format(num_params / 1000000))

    # net = net.cuda()
    enc = enc.cuda()
    dec = dec.cuda()
    return enc, dec


def warp_network_in_dataparallel(net, gpuid):
    """
    Wrap the network in Dataparallel
    """
    # torch.cuda.set_device(gpuid)
    # net.cuda(gpuid)    
    # net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpuid], find_unused_parameters=True)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpuid])
    return net


def get_model(args, num_classes):
    """
    Fetch Network Function Pointer
    """
    network = args.arch
    module = network[:network.rfind('.')]
    model = network[network.rfind('.') + 1:]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    # net = net_func(args=args, num_classes=num_classes)
    enc, dec = net_func(args=args, num_classes=num_classes)
    return enc, dec
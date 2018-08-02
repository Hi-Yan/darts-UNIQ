#### https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/master/pytorch_segmentation_detection/utils/flops_benchmark.py
import math
import torch
from torch.nn import Linear

# ---TBD :: Need to pass this arguments from imagenet.py
param_bitwidth = 4
act_bitwidth = 4


# ---- Public functions


def add_flops_counting_methods(net_main_module):
    """Adds flops counting functions to an existing model. After that
    the flops count should be activated and the model should be run on an input
    image.

    Example:

    fcn = add_flops_counting_methods(fcn)
    fcn = fcn.cuda().train()
    fcn.start_flops_count()


    _ = fcn(batch)

    fcn.compute_average_flops_cost() / 1e9 / 2 # Result in GFLOPs per image in batch

    Important: dividing by 2 only works for resnet models -- see below for the details
    of flops computation.

    Attention: we are counting multiply-add as two flops in this work, because in
    most resnet models convolutions are bias-free (BN layers act as bias there)
    and it makes sense to count muliply and add as separate flops therefore.
    This is why in the above example we divide by 2 in order to be consistent with
    most modern benchmarks. For example in "Spatially Adaptive Computatin Time for Residual
    Networks" by Figurnov et al multiply-add was counted as two flops.

    This module computes the average flops which is necessary for dynamic networks which
    have different number of executed layers. For static networks it is enough to run the network
    once and get statistics (above example).

    Implementation:
    The module works by adding batch_count to the main module which tracks the sum
    of all batch sizes that were run through the network.

    Also each convolutional layer of the network tracks the overall number of flops
    performed.

    The parameters are updated with the help of registered hook-functions which
    are being called each time the respective layer is executed.

    Parameters
    ----------
    net_main_module : torch.nn.Module
        Main module containing network

    Returns
    -------
    net_main_module : torch.nn.Module
        Updated main module with new methods/attributes that are used
        to compute flops.
    """

    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(net_main_module)
    net_main_module.compute_average_bops_cost = compute_average_bops_cost.__get__(net_main_module)

    net_main_module.reset_flops_count()

    # Adding varialbles necessary for masked flops computation
    net_main_module.apply(add_flops_mask_variable_or_reset)

    return net_main_module


def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns current mean flops consumption per image.

    """

    batches_count = self.__batch_counter__

    flops_sum = 0

    for module in self.modules():

        if isinstance(module, torch.nn.Conv2d):
            flops_sum += module.__flops__

    return flops_sum / batches_count


def compute_average_bops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns current mean flops consumption per image.

    """

    batches_count = self.__batch_counter__

    bops_sum = 0

    for module in self.modules():

        if isinstance(module, torch.nn.Conv2d):
            bops_sum += module.__bops__

    return bops_sum / batches_count


def start_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.

    """

    add_batch_counter_hook_function(self)

    self.apply(add_flops_counter_hook_function)


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.

    """

    remove_batch_counter_hook_function(self)

    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """

    add_batch_counter_variables_or_reset(self)

    self.apply(add_flops_counter_variable_or_reset)


def add_flops_mask(module, mask):
    def add_flops_mask_func(module):
        if isinstance(module, torch.nn.Conv2d):
            module.__mask__ = mask

    module.apply(add_flops_mask_func)


def remove_flops_mask(module):
    module.apply(add_flops_mask_variable_or_reset)


# ---- Internal functions
def conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    bops = 0
    input = input[0]
    batch_size = input.shape[0]
    output_height, output_width = output.shape[2:]

    kernel_height, kernel_width = conv_module.kernel_size
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    # We count multiply-add as 2 flops
    conv_per_position_flops = 2 * kernel_height * kernel_width * in_channels * out_channels / groups ** 2

    active_elements_count = batch_size * output_height * output_width

    if conv_module.__mask__ is not None:
        # (b, 1, h, w)
        flops_mask = conv_module.__mask__.expand(batch_size, 1, output_height, output_width)
        active_elements_count = flops_mask.sum()

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops
    conv_module.__flops__ += overall_flops

    # Bops code
    param_bitwidth = conv_module.__param_bitwidth__
    act_bitwidth = conv_module.__act_bitwidth__

    bit_ops = 0
    num_of_conv_mults = batch_size * output_height * output_width * out_channels * in_channels * kernel_height * kernel_width
    num_of_conv_adds = batch_size * output_height * output_width * out_channels * (
            in_channels * kernel_height * kernel_width - 1)
    if param_bitwidth != 1:
        max_mac_value = (2 ** act_bitwidth - 1) * (in_channels * kernel_height * kernel_width) / (groups ** 2) * (2 ** (
                param_bitwidth - 1) - 1)  # param_bitwidth-1 becuase 1 bit is sign bit and not really participate in the multiplication
    else:
        max_mac_value = (2 ** act_bitwidth - 1) * (in_channels * kernel_height * kernel_width) / (
                groups ** 2)  # param_bitwidth-1 becuase 1 bit is sign bit and not really participate in the multiplication
    log2_max_mac_value = math.ceil(math.log2(max_mac_value))
    bit_ops += num_of_conv_mults * (param_bitwidth - 1) * act_bitwidth
    bit_ops += num_of_conv_adds * (log2_max_mac_value)
    conv_module.__bops__ += bit_ops


def batch_counter_hook(module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]

    module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module):
    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()

        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
    if isinstance(module, torch.nn.Conv2d):
        module.__flops__ = 0
        module.__bops__ = 0


def add_flops_counter_hook_function(module):
    if isinstance(module, torch.nn.Conv2d):

        if hasattr(module, '__flops_handle__'):
            return

        handle = module.register_forward_hook(conv_flops_counter_hook)
        module.__flops_handle__ = handle


def remove_flops_counter_hook_function(module):
    if isinstance(module, torch.nn.Conv2d):

        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()

            del module.__flops_handle__


# --- Masked flops counting


# Also being run in the initialization
def add_flops_mask_variable_or_reset(module):
    if isinstance(module, torch.nn.Conv2d):
        module.__mask__ = None


def add_bitwidths_attr(model, param_bitwidth, act_bitwidth):
    i = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            module.__param_bitwidth__ = param_bitwidth[i]
            module.__act_bitwidth__ = act_bitwidth
            i += 1


# def count_flops(model, batch_size, device, dtype, input_size, in_channels, *params):
def count_flops(model, batch_size, input_size, in_channels):
    # net = model(*params)
    net = model
    net.prepare_uniq()

    net = add_flops_counting_methods(net)

   # net.to(device=device, dtype=dtype)
    net.cuda()
    net = net.train()

   # batch = torch.randn(batch_size, in_channels, input_size, input_size).to(device=device, dtype=dtype)
    batch = torch.randn(batch_size, in_channels, input_size, input_size).cuda()
    net.start_flops_count()
    if isinstance(model.op, Linear):
        batch = batch.view(batch.size(0), -1)
    if net.useResidual:
        _ = net(batch, batch)
    else:
        _ = net(batch)
    flops, bops = net.compute_average_flops_cost() / 2, net.compute_average_bops_cost()
    net.stop_flops_count()
    return (flops, bops)  # Result in FLOPs

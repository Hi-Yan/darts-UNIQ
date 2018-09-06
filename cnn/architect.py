from torch.optim import Adam, SGD
from torch.nn.utils.clip_grad import clip_grad_norm_


# from torch import cat, zeros_like
# from torch.autograd import Variable, grad
# from cnn.model_search import Network


# def _concat(xs):
#     return cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, modelReplicator, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.lr = args.arch_learning_rate
        self.weight_decay = args.arch_weight_decay

        self.modelReplicator = modelReplicator

        # self.model = model
        # self.optimizer = Adam(self.model.arch_parameters(), lr=args.arch_learning_rate,
        #                       betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
        # self.optimizer = SGD(self.model.arch_parameters(), lr=args.arch_learning_rate, momentum=args.momentum,
        #                      weight_decay=args.arch_weight_decay)

    def step(self, model, input_valid, target_valid):
        # collect architecture parameters
        arch_parameters = model.arch_parameters()
        # init optimizer
        optimizer = SGD(arch_parameters, lr=self.lr, momentum=self.network_momentum)
        # reset optimizer gradients
        optimizer.zero_grad()
        # calc loss
        # loss = self.modelReplicator.loss(model, input_valid, target_valid)
        loss = model._loss(input_valid, target_valid)
        # clip grad norm
        clip_grad_norm_(arch_parameters, 10.0)
        # perform optimizer step
        optimizer.step()

        return loss

    # def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    #     arch_parameters = self.model.arch_parameters()
    #     # init optimizer for current arch_parameters
    #
    #     self.optimizer = SGD(arch_parameters, lr=self.lr, momentum=self.network_momentum)
    #     # , weight_decay=self.weight_decay)
    #     self.optimizer.zero_grad()
    #     if unrolled:
    #         loss = self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta,
    #                                             network_optimizer)
    #     else:
    #         loss = self._backward_step(input_valid, target_valid)
    #     # print('grads:{}'.format(self.model.block1.alphas.grad))
    #     grad_norm = clip_grad_norm_(arch_parameters, 10.0)
    #
    #     # print('grad_norm:{}'.format(grad_norm))
    #     # print('alphas:{}'.format(self.model.block1.alphas))
    #     # for p in self.optimizer.param_groups:
    #     #     p['params'][0].grad.data.clamp_(min=0)
    #     self.optimizer.step()
    #
    #     return grad_norm, loss
    #
    # def _backward_step(self, input_valid, target_valid):
    #     loss = self.model._loss(input_valid, target_valid)
    #     # for v in self.model.arch_parameters():
    #     #     if v.grad is not None:
    #     #         v.grad.data.zero_()
    #     # loss.backward(retain_graph=True)
    #     return loss

    # def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    #     loss = self.model._loss(input, target)
    #     theta = _concat(self.model.parameters()).data
    #     try:
    #         moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
    #             self.network_momentum)
    #     except:
    #         moment = zeros_like(theta)
    #     dtheta = _concat(grad(loss, self.model.parameters())).data + self.network_weight_decay * theta
    #     model_unrolled = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
    #     return model_unrolled
    #
    # def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    #     model_unrolled = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    #     loss = model_unrolled._loss(input_valid, target_valid)
    #     grads = grad(loss, model_unrolled.arch_parameters(), retain_graph=True)
    #
    #     theta = model_unrolled.parameters()
    #     dtheta = grad(loss, model_unrolled.parameters())
    #     vector = [dt.add(self.network_weight_decay, t).data for dt, t in zip(dtheta, theta)]
    #     implicit_grads = self._hessian_vector_product(model_unrolled, vector, input_train, target_train)
    #
    #     for g, ig in zip(grads, implicit_grads):
    #         g.data.sub_(eta, ig.data)
    #
    #     for v, g in zip(self.model.arch_parameters(), grads):
    #         if v.grad is None:
    #             v.grad = Variable(g.data)
    #         else:
    #             v.grad.data.copy_(g.data)
    #
    #     return loss
    #
    # def _construct_model_from_theta(self, theta):
    #     model_clone = Network(self.model._C, self.model._num_classes, self.model._layers, self.model._criterion).cuda()
    #
    #     for x, y in zip(model_clone.arch_parameters(), self.model.arch_parameters()):
    #         x.data.copy_(y.data)
    #     model_dict = self.model.state_dict()
    #
    #     params, offset = {}, 0
    #     for k, v in self.model.named_parameters():
    #         v_length = np.prod(v.size())
    #         params[k] = theta[offset: offset + v_length].view(v.size())
    #         offset += v_length
    #
    #     assert offset == len(theta)
    #     model_dict.update(params)
    #     model_clone.load_state_dict(model_dict)
    #     return model_clone.cuda()
    #
    # def _hessian_vector_product(self, model, vector, input, target, r=1e-2):
    #     R = r / _concat(vector).norm()
    #     for p, v in zip(model.parameters(), vector):
    #         p.data.add_(R, v)
    #     loss = model._loss(input, target)
    #     grads_p = grad(loss, model.arch_parameters())
    #
    #     for p, v in zip(model.parameters(), vector):
    #         p.data.sub_(2 * R, v)
    #     loss = model._loss(input, target)
    #     grads_n = grad(loss, model.arch_parameters())
    #
    #     for p, v in zip(model.parameters(), vector):
    #         p.data.add_(R, v)
    #
    #     return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

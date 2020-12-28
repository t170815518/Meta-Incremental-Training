import torch
import copy
import logging
from collections import OrderedDict
from LAN.utils.link_prediction import run_link_prediction


logger = logging.getLogger()


def meta_incremental_train(model: torch.nn.Module, optimizer, data_iterator, i: int, config, logger,
                           batch_size: int = 1024, window_limit: int = -1, eval_interval: int = 50) -> torch.nn.Module:
    """
    The model will be sent to GPU by default.
    :param batch_size: size of the batch/window
    :param model: torch.nn.module, the model to train. It must have loss(self, window_dict) method,
    where window_dict contains all input tensors, created by data_iterator and there is loss tensor returned
    :param optimizer: torch.optimizer.Optimizer, the optimizer object of the model
    :param data_iterator:
    :param i: positive int, the number of training epochs (the outer loop)
    :param window_limit:
    :return: trained model
    """
    inner_optimizer = torch.optim.Adam(model.parameters(), lr=config.inner_learning_rate)

    mr_opt = float("inf")
    epoch_num = 0
    stopping_flags = 0

    while True:
        model.train()

        grads = []
        logger.info("Inner loop: started")
        for window in data_iterator.iter_from_list(i, window_limit):
            g = run_inner(config, model, window)
            grads.append(g)
        logger.info("Inner loop: finished")

        perform_meta_update(config, data_iterator.windows[i], model, grads, optimizer)

        epoch_num += 1
        print("epoch {} completes".format(epoch_num))

        if epoch_num % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                mr = run_link_prediction(config, model, data_iterator, i, logger)
                stopping_flags += early_stopping(mr, mr_opt, config.threshold)
                if stopping_flags > 1:
                    logger.info("Early stopping at inner loop {}".format(epoch_num))
                    break
                if mr < mr_opt:
                    mr_opt = mr
                    stopping_flags = 0

    return model


def _temporary_update(model, feed_dict, optimizer):
    """ Updates the model with data (e.g. in the single window). """
    loss = model.loss(feed_dict)


def early_stopping(mr, mr_opt, threshold):
    if 100 * (mr / mr_opt - 1) > threshold:
        return 1
    else:
        return 0


def run_inner(config, model, task):
    """
    :param config:
    :param model:
    :param task: the feed_dict
    :return:
    """
    loss = model.loss(task)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    updated_params = OrderedDict()
    for (name, param), grad in zip(model.named_parameters(), grads):
        param_prime = copy.deepcopy(param)
        param_prime.data.sub_(config.inner_learning_rate * grad.data)
        updated_params[name] = param_prime
    meta_grads = perform_task_meta_update(config, task, model, updated_params)
    return meta_grads


def perform_task_meta_update(config, task, model, updated_params):
    task_loss = model.loss(task, updated_params)
    task_grads = torch.autograd.grad(task_loss, updated_params.values())

    model_loss = model.loss(task)
    model_grads = torch.autograd.grad(model_loss, model.parameters(), create_graph=True)

    meta_grads = {}
    for task_grad, model_grad, (name, param) in zip(task_grads, model_grads, model.named_parameters()):
        shape = model_grad.shape
        task_grad.volatile = False
        if len(shape) > 1:
            new_shape = 1
            for dim in shape:
                new_shape = new_shape * dim

            task_grad = task_grad.view(new_shape)
            model_grad = model_grad.view(new_shape)

        g = torch.dot(model_grad, task_grad)
        model_second_grad = torch.autograd.grad(g, param, retain_graph=True)
        if task_grad.shape != shape:
            task_grad = task_grad.view(*shape)

        final_grad = task_grad - torch.mul(model_second_grad[0], config.inner_learning_rate)
        meta_grads[name] = final_grad
    return meta_grads


def perform_meta_update(config, task, model, grads, opt):
    logger.info("Perform meta update")
    loss = model.loss(task)
    gradients = {k: sum(g[k] for g in grads) for k in grads[0].keys()}

    hooks = []
    for (k, v) in model.named_parameters():
        def get_closure():
            key = k

            def replace_grad(grad):
                return gradients[key]

            return replace_grad

        hooks.append(v.register_hook(get_closure()))

    opt.zero_grad()
    loss.backward()
    opt.step()

    loss = model.loss(task)
    # Remove the hooks before next training phase
    for h in hooks:
        h.remove()

    return loss

import torch


def meta_incremental_train(model: torch.nn.Module, optimizer, data_iterator, epoch_num: int = 100,
                           window_limit: int = -1) -> torch.nn.Module:
    """
    The model will be sent to GPU by default.
    :param model: torch.nn.module, the model to train. It must have loss(self, window_dict) method,
    where window_dict contains all input tensors, created by data_iterator and there is loss tensor returned
    :param optimizer: torch.optimizer.Optimizer, the optimizer object of the model
    :param data_iterator:
    :param epoch_num: positive int, the number of training epochs
    :param window_limit:
    :return: trained model
    TODO: cpu/cuda
    """
    if epoch_num <= 0:
        raise ValueError("epoch_num should be positive integer.")

    for i in range(epoch_num):
        model.zero_grad()

        for feed_dict in data_iterator.iter(i, 0):
            previous_param = model.state_dict().copy()
            _temporary_update(model, feed_dict, optimizer)

            def _closure():
                """ Uses the current parameters of the model on the current and previous windows within the range, and
                returns the total loss on these windows.
                """
                model.zero_grad()
                total_loss = 0
                for window_dict in data_iterator.iter(i, window_limit):
                    loss = model.loss(window_dict)
                    loss.backward()
                    total_loss += loss.item()

                return total_loss

            model.load_state_dict(previous_param)
            optimizer.step(closure=_closure)

    return model


def _temporary_update(model, feed_dict, optimizer):
    """ Updates the model with data (e.g. in the single window). """
    loss = model.loss(feed_dict)
    loss.backward()
    optimizer.step()

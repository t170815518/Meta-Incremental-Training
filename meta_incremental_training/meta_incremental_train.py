import torch
from LAN.utils.link_prediction import run_link_prediction


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
    TODO: cpu/cuda
    """

    new_data_generator = data_iterator.batch_iter_epoch(data_iterator.triplets_train, batch_size=batch_size)
    inner_optimizer = torch.optim.Adam(model.parameters(), lr=config.inner_learning_rate)
    data_iterator.windows = []
    mr_opt = float("inf")
    epoch_num = 0
    while True:
        model.train()
        try:
            data_iterator.windows.append(next(new_data_generator))  # add new data during each iteration
        except StopIteration:
            return model

        model.zero_grad()

        feed_dict = data_iterator.windows[epoch_num]

        previous_param = model.state_dict().copy()
        _temporary_update(model, feed_dict, inner_optimizer)

        def _closure():
            """ Uses the current parameters of the model on the current and previous windows within the range, and
            returns the total loss on these windows.
            """
            model.zero_grad()
            total_loss = 0
            for window_dict in data_iterator.iter_from_list(epoch_num, window_limit):
                loss = model.loss(window_dict)
                loss.backward()
                total_loss += loss.item()

            model.load_state_dict(previous_param)
            return total_loss

        optimizer.step(closure=_closure)

        epoch_num += 1
        if epoch_num % 100 == 0:
            print("{} batches completes".format(epoch_num))

        # if epoch_num % eval_interval == 0:
        #     model.eval()
        #     with torch.no_grad():
        #         mr = run_link_prediction(config, model, data_iterator, i, logger)
            # is_early_stopping = early_stopping(mr, mr_opt, config.threshold)
            # if is_early_stopping:
            #     logger.info("Early stopping at inner loop {}".format(epoch_num))
            #     break
            # if mr < mr_opt:
            #     mr_opt = mr

    return model


def _temporary_update(model, feed_dict, optimizer):
    """ Updates the model with data (e.g. in the single window). """
    loss = model.loss(feed_dict)
    loss.backward()
    optimizer.step()


def early_stopping(mr, mr_opt, threshold):
    return 100 * (mr / mr_opt - 1) > threshold
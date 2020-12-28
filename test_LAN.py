""" This module is for testing meta-incremental training with LAN. """

import argparse
import logging
import time
import datetime
import os
import torch
from LAN.utils.data_helper import DataSet
from LAN.utils.link_prediction import run_link_prediction
from LAN.framework import LAN
from meta_incremental_training.incremental_iter import IncrementalIterator
from meta_incremental_training.meta_incremental_train import meta_incremental_train
from NSCaching.BernCorrupter import BernCorrupter


# inherit IncrementalIterator to use meta-incremental training
class MetaIncrementalDataset(DataSet, IncrementalIterator):
    def __init__(self, config, logger, win_size):
        DataSet.__init__(self, config, logger)
        IncrementalIterator.__init__(self, win_size)


logger = logging.getLogger()
file_name = str(datetime.datetime.now())


def main():
    config = parse_arguments()
    if config.training_method == "meta_incremental":
        run_meta_incre_training(config)
    elif config.training_method == "batch":
        run_batch_training(config)


def parse_arguments():
    """ Parses arguments from CLI. """
    parser = argparse.ArgumentParser(description="Configuration for LAN model")
    parser.add_argument('--data_dir', '-D', type=str, default="data/FB15k-237")
    parser.add_argument('--save_dir', '-S', type=str, default="data/FB15k-237")
    parser.add_argument('--log_file_path', type=str, default="{}.log".format(file_name))
    # model
    parser.add_argument('--use_relation', type=int, default=1)
    parser.add_argument('--embedding_dim', '-e', type=int, default=100)
    parser.add_argument('--max_neighbor', type=int, default=64)
    parser.add_argument('--n_neg', '-n', type=int, default=1)
    parser.add_argument('--aggregate_type', type=str, default='attention')
    parser.add_argument('--score_function', type=str, default='TransE')
    parser.add_argument('--loss_function', type=str, default='margin')
    parser.add_argument('--margin', type=float, default='1.0')
    parser.add_argument('--corrupt_mode', type=str, default='both')
    # training
    parser.add_argument('--training_method', type=str, choices=["batch", "meta_incremental"], default="meta_incremental")
    parser.add_argument('--learning_rate', type=float, default=0.003)
    parser.add_argument('--num_epoch', type=int, default=73)
    parser.add_argument('--weight_decay', '-w', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--evaluate_size', type=int, default=100)
    parser.add_argument('--steps_per_display', type=int, default=100)
    parser.add_argument('--epoch_per_checkpoint', type=int, default=50)
    parser.add_argument("--load_model", type=bool, default=False)
    # sampling mode
    parser.add_argument('--sampling_mode', type=str, choices=["random", "structured"], default="structured")
    # NSCaching
    parser.add_argument('--N_1', type=int, default=30)
    parser.add_argument('--N_2', type=int, default=90)
    # meta-incremental training option
    parser.add_argument('--window_size', type=int, default=-1)
    parser.add_argument('--threshold', type=int, default=-1)
    parser.add_argument('--inner_learning_rate', type=float, default=0.05)
    # gpu option
    parser.add_argument('--gpu_fraction', type=float, default=0.2)
    parser.add_argument('--gpu_device', type=str, default='0')
    parser.add_argument('--allow_soft_placement', type=bool, default=False)

    return parser.parse_args()


def run_batch_training(config):
    # set up GPU
    config.device = torch.device("cuda:0")

    set_up_logger(config)

    logger.info('args: {}'.format(config))

    # prepare data
    logger.info("Loading data...")
    dataset = DataSet(config, logger)
    logger.info("Loading finish...")

    dataset.cache = [dataset.get_cache()]

    corrputer = BernCorrupter(dataset.triplets_train, dataset.num_entity, dataset.num_relation*2)
    dataset.corrupter = corrputer

    model = LAN(config, dataset.num_training_entity, dataset.num_relation)
    save_path = os.path.join(config.save_dir, "train_model.pt")
    model.to(config.device)
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    dataset.model = model

    # training
    num_batch = dataset.num_sample // config.batch_size
    logger.info('Train with {} batches'.format(num_batch))
    best_performance = float("inf")

    for epoch in range(config.num_epoch):
        st_epoch = time.time()
        loss_epoch = 0.
        cnt_batch = 0

        for batch_data in dataset.batch_iter_epoch(dataset.triplets_train, config.batch_size, config.n_neg):
            model.train()

            st_batch = time.time()

            feed_dict = batch_data

            loss_batch = model.loss(feed_dict)

            cnt_batch += 1
            loss_epoch += loss_batch.item()

            loss_batch.backward()
            optim.step()
            model.zero_grad()

            en_batch = time.time()

            # print an overview every some batches
            if (cnt_batch + 1) % config.steps_per_display == 0 or (cnt_batch + 1) == num_batch:
                batch_info = 'epoch {}, batch {}, loss: {:.3f}, time: {:.3f}s'.format(epoch, cnt_batch, loss_batch,
                                                                                      en_batch - st_batch)
                print(batch_info)
                logger.info(batch_info)
        en_epoch = time.time()
        epoch_info = 'epoch {}, mean loss: {:.3f}, time: {:.3f}s'.format(epoch, loss_epoch / cnt_batch,
                                                                         en_epoch - st_epoch)
        print(epoch_info)
        logger.info(epoch_info)

        # evaluate the model every some steps
        if (epoch + 1) % config.epoch_per_checkpoint == 0 or (epoch + 1) == config.num_epoch:
            model.eval()
            st_test = time.time()
            with torch.no_grad():
                performance = run_link_prediction(config, model, dataset, epoch, logger, is_test=False)
            if performance < best_performance:
                best_performance = performance
                torch.save(model.state_dict(), "model_trained.pt")
                time_str = datetime.datetime.now().isoformat()
                saved_message = '{}: model at epoch {} save in file {}'.format(time_str, epoch, save_path)
                print(saved_message)
                logger.info(saved_message)

            en_test = time.time()
            test_finish_message = 'testing finished with time: {:.3f}s'.format(en_test - st_test)
            print(test_finish_message)
            logger.info(test_finish_message)

    finished_message = 'Training finished'
    print(finished_message)
    logger.info(finished_message)


def run_meta_incre_training(config):
    # set up GPU
    config.device = torch.device("cuda:0")

    set_up_logger(config)

    logger.info('args: {}'.format(config))

    # prepare data
    logger.info("Loading data...")
    dataset = MetaIncrementalDataset(config, logger, config.window_size)
    logger.info("Loading finish...")

    # training initialization
    model = LAN(config, dataset.num_training_entity, dataset.num_relation)
    if config.load_model and os.path.exists("model_trained.pt"):
        model.load_state_dict(torch.load("model_trained.pt"))
    model.to(config.device)
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    new_data_generator = dataset.batch_iter_epoch(dataset.triplets_train, batch_size=config.batch_size)

    for i in range(config.num_epoch):
        try:
            dataset.windows.append(next(new_data_generator))  # add new data during each iteration
        except StopIteration:
            pass

        logger.info("Iteration {} starts".format(i))
        trained_model = meta_incremental_train(model, optim, dataset, i, config, logger,
                                               config.batch_size, config.window_size, config.epoch_per_checkpoint)
        torch.save(model.state_dict(), "{}.pt".format(file_name))
        # if i % config.epoch_per_checkpoint == 0:
        #     torch.save(trained_model.state_dict(), "model_trained.pt")
        #     model.eval()
        #     with torch.no_grad():
        #         mr = run_link_prediction(config, model, dataset, i, logger)


def set_up_logger(config):
    checkpoint_dir = config.save_dir
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(config.log_file_path, 'w+')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


if __name__ == '__main__':
    main()

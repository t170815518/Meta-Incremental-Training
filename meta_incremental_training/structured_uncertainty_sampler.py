""" This module is to implement structured uncertainty sampling in ActiveLink."""
import json
import logging
import os
from math import sqrt
from collections import defaultdict
from random import shuffle, choice

import torch
import numpy as np
from sklearn.cluster import KMeans


logger = logging.getLogger()


def calculate_uncertainty(pred):
    positive = pred
    positive_approx = torch.div(
        torch.sum(positive, 0),
        10
    )

    negative = torch.add(
        torch.neg(positive),
        1
    )
    negative_approx = torch.div(
        torch.sum(negative, 0),
        10
    )

    log_positive_approx = torch.log(positive_approx)
    log_negative_approx = torch.log(negative_approx)

    entropy = torch.neg(
        torch.add(
            torch.mul(positive_approx, log_positive_approx),
            torch.mul(negative_approx, log_negative_approx)
        )
    )

    uncertainty = torch.mean(entropy, 1)
    return uncertainty


class StructuredUncertaintySampler:
    """
    Attributes:
        - clusters: dict, {cluster_id: triplets in np.array}
    """
    def __init__(self, data_dir, args, dataset_helper, num_entities):
        """
        :param data_dir: data directory that contains embedding (e.g. entity2vec file) and additional_graph info
        (e.g. graph_info.json)
        :param data_size:
        :param args: config file, with n_clusters, sample_size, uncertainty_eval_size
        :param dataset_helper: Object, prepare feed_dict given the raw samples
        """
        self.n_clusters = args.n_clusters
        self.sample_size = args.batch_size
        self.uncertainty_eval_size = args.uncertainty_eval_size
        self.dataset_helper = dataset_helper
        self.is_use_cache = args.is_use_cache
        self.num_negative = args.n_neg
        self.batch_size = args.batch_size

        self.entity_emb_path = os.path.join(data_dir, "entity2vec")
        if not os.path.exists(self.entity_emb_path):
            raise Exception("Entities embedding file is missing")

        graph_info_path = os.path.join(data_dir, "graph_info.json")
        with open(graph_info_path, 'r') as json_file:
            self.graph_info = json.load(json_file)

        self.clusters = defaultdict(lambda: [])

    def initialize(self):
        """ Does the clustering and returns the initial sample (without computing uncertainty score). """
        logger.info("Start clustering")

        if os.path.exists("labels_list.pkl"):  # load the pre-clustered information
            import pickle
            with open("labels_list.pkl", 'rb') as f:
                labels = pickle.load(f)
        else:
            entity_embeddings = np.loadtxt(self.entity_emb_path)
            kmeans = KMeans(n_clusters=self.n_clusters).fit(entity_embeddings)
            labels = kmeans.labels_.tolist()

            import pickle
            with open("labels_list.pkl", 'wb') as f:
                pickle.dump(labels, f)

        for entity_id, cluster_id in enumerate(labels):
            self.clusters[cluster_id].extend(self.graph_info[str(entity_id)]["triplets_as_head"])

        empty_clusters = []

        for key, item in self.clusters.items():
            if item:  # when the cluster is not empty
                shuffle(item)
                self.clusters[key] = np.array(item) # convert list to array for easier index selection
            else:
                empty_clusters.append(key)  # to remove the empty clusters

        for cluster_id in empty_clusters:
            self.clusters.pop(cluster_id)

        logger.info("Completes clustering")

        # print the clustering information
        items_num = [len(cluster_items) for cluster_items in self.clusters.values()]
        mean = sum(items_num) / len(items_num)
        max_num = max(items_num)
        min_num = min(items_num)
        logger.info("Average items in the cluster = {}\nMax items = {}\nMin items={}".format(mean, max_num, min_num))

        # return initial samples
        return self.random_select_initial_samples_from_clusters()

    def random_select_initial_samples_from_clusters(self):
        """ Randomly selects samples from each cluster without computing uncertainty. """
        samples = []
        empty_clusters = []
        sample_size = 0

        triples_per_cluster = int(round(self.sample_size / len(self.clusters)))
        if triples_per_cluster == 0:
            triples_per_cluster = 1

        for cluster_id, cluster_data in self.clusters.items():
            end_index = min(triples_per_cluster, len(cluster_data))
            np.random.shuffle(cluster_data)

            selections = cluster_data[:end_index]
            sample_size += len(selections)
            samples.append(selections)

            if len(cluster_data) - end_index > 1:
                self.clusters[cluster_id] = cluster_data[end_index:]
            else:
                empty_clusters.append(cluster_id)

            if sample_size == self.sample_size:
                break

        for cluster_id in empty_clusters:
            self.clusters.pop(cluster_id)

        return np.concatenate(samples, axis=0)

    def update(self, model):
        """ Feeds more samples for the training, based on structured and uncertainty sampling. """
        samples = []
        sample_size = 0
        empty_clusters = []
        all_clusters_size = sum(len(v) for v in self.clusters.values())

        for cluster_id, cluster_data in self.clusters.items():
            current_cluster_ratio = float(len(cluster_data)) / all_clusters_size
            n = int(round(current_cluster_ratio * self.sample_size))

            if n == 0:
                n = 1

            batch_size = n * 2  # choose top 50% data with high uncertainty
            if batch_size < 50:  # at least evaluate 50 triplets
                batch_size = 50

            cluster = self.clusters[cluster_id]
            cluster_size = cluster.shape[0]  # cluster content should be in np.array
            indices = np.random.randint(low=0, high=cluster_size, size=min(batch_size, cluster_size))
            batch = cluster[indices]
            predictions = torch.FloatTensor(batch_size, self.uncertainty_eval_size)
            for i, (h, r, _) in enumerate(batch):
                feed_dict = self.dataset_helper.prepare_batch_for_uncertainty(h, r, self.uncertainty_eval_size)
                predictions[i] = model.get_positive_score(feed_dict)
            uncertainty = calculate_uncertainty(predictions)
            uncertainty_sorted, uncertainty_indices_sorted = torch.sort(uncertainty, 0, descending=True)
            indices = indices[uncertainty_indices_sorted[:n].detach().cpu().numpy()]
            selections = cluster[indices]
            samples.append(selections)
            sample_size += len(selections)
            # noinspection PyTypeChecker
            self.clusters[cluster_id] = np.delete(cluster, indices)  # pop up elements
            if cluster_data.shape[0] < 1:
                empty_clusters.append(cluster_id)
            if  sample_size >= self.sample_size:
                break

        for cluster_id in empty_clusters:
            self.clusters.pop(cluster_id)

        return np.concatenate(samples, axis=0)

    def iterate(self, model):
        """
        TODO:
         - add using cache
         - when sample size not equal to batch size """
        initial_samples = self.initialize()  # return the initial sample
        yield from self.dataset_helper.batch_iter_epoch(initial_samples, self.batch_size, self.num_negative, corrupt=True,
                                             shuffle=False, is_use_cache=self.is_use_cache)

        while True:
            samples = self.update(model)
            if samples:
                yield from self.dataset_helper.batch_iter_epoch(samples, self.batch_size, self.num_negative,
                                                                corrupt=True, shuffle=False, is_use_cache=
                                                                self.is_use_cache)
            else:
                raise StopIteration  # when there are no new samples

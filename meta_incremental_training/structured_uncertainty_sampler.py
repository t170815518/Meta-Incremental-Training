""" This module is to implement structured uncertainty sampling in ActiveLink."""
import json
import logging
import os
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
    def __init__(self, data_dir, args, dataset_helper):
        """
        :param data_dir: data directory that contains embedding (e.g. entity2vec file) and additional_graph info
        (e.g. graph_info.json)
        :param data_size:
        :param args: config file, with n_clusters, sample_size, uncertainty_eval_size
        :param dataset_helper: Object, prepare feed_dict given the raw samples
        """
        self.n_clusters = args.n_clusters
        self.sample_size = args.sample_size
        self.uncertainty_eval_size = args.uncertainty_eval_size
        self.dataset_helper = dataset_helper

        self.entity_emb_path = os.path.join(data_dir, "entity2vec")
        if not os.path.exists(self.entity_emb_path):
            raise Exception("Entities embedding file is missing")

        graph_info_path = os.path.join(data_dir, "graph_info.json")
        with open(graph_info_path, 'r') as json_file:
            self.graph_info = json.load(json_file)

        self.clusters = defaultdict(lambda: [])
        self.non_empty_cluster_ids = set()

    def initialize(self):
        """ Does the clustering and returns the initial sample (without computing uncertainty score). """
        logger.info("Start clustering")

        entity_embeddings = np.loadtxt(self.entity_emb_path)
        kmeans = KMeans(n_clusters=self.n_clusters).fit(entity_embeddings)

        labels = kmeans.labels_.tolist()
        for entity_id, cluster_id in enumerate(labels):
            self.clusters[cluster_id].extend(self.graph_info[cluster_id]["triplets_as_head"])

        # shuffle for future random selection via pop
        for key, item in self.clusters.items():
            if item:  # when the cluster is not empty
                shuffle(item)
                self.non_empty_cluster_ids.add(key)
                self.clusters[key] = np.array(item)  # convert list to array for easier index selection
        self.non_empty_cluster_ids = list(self.non_empty_cluster_ids)  # convert set to list

        logger.info("Completes clustering")

        items_num = [len(cluster_items) for cluster_items in self.clusters.values()]
        mean = sum(items_num) / len(items_num)
        max_num = max(items_num)
        min_num = min(items_num)
        logger.info("Average items in the cluster = {}\nMax items = {}\nMin items={}".format(mean, max_num, min_num))

        # return initial samples
        return self.random_select_initial_samples_from_clusters()

    def random_select_initial_samples_from_clusters(self):
        """ Randomly selects samples from each cluster with the size proportional to the cluster size,
        without computing uncertainty. """
        samples = []
        lengths = np.array([len(self.clusters[i]) for i in range(self.n_clusters)])
        probabilities = lengths / np.sqrt(np.sum(lengths ** 2))  # normalize to probability vector (unit vector)
        sample_choices = np.random.choice(a=len(self.clusters.keys()), size=self.sample_size, p=probabilities)
        cluster_ids, sample_nums = np.unique(sample_choices, return_counts=True)
        for cluster_id, sample_num in zip(cluster_ids, sample_nums):
            cluster = self.clusters[cluster_id]
            cluster_size = cluster.shape[0]  # cluster content should be in np.array
            indices = np.random.randint(low=0, high=cluster_size, size=min(sample_num, cluster_size))
            batch = cluster[indices]
            samples.append(cluster[indices])
            self.clusters[cluster_id] = np.delete(cluster, indices)  # pop up elements
        return np.concatenate(samples, axis=0)

    def update(self, model):
        samples = []
        lengths = np.array([len(self.clusters[i]) for i in range(self.n_clusters)])
        probabilities = lengths / np.sqrt(np.sum(lengths ** 2))
        sample_choices = np.random.choice(a=len(self.clusters.keys()), size=self.sample_size, p=probabilities)
        cluster_ids, sample_nums = np.unique(sample_choices, return_counts=True)
        for cluster_id, sample_num in zip(cluster_ids, sample_nums):
            batch_size = self.sample_size * 2  # choose top 50% data with high uncertainty
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
            indices = indices[uncertainty_indices_sorted[:sample_num].detach().cpu().numpy()]
            samples.append(cluster[indices])
            # noinspection PyTypeChecker
            self.clusters[cluster_id] = np.delete(cluster, indices)  # pop up elements
        return np.concatenate(samples, axis=0)

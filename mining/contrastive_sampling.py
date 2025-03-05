import numpy as np
import torch
from .strategy import Strategy

class CASampling(Strategy):
	def __init__(self, train_dataset, idxs_lb, net, handler, train_transform, test_transform, args, source_dataset):
		super(CASampling, self).__init__(train_dataset, idxs_lb, net, handler, train_transform, test_transform, args, source_dataset)
		self.output_dir = args.output_dir
		self.seed = args.seed

	def label_statistic(self, labels, metrics, num_label, topk=500, descending=True):
		idx_topk = metrics.sort(descending=descending)[1][:topk]

		statistic = torch.zeros(num_label)
		for idx in idx_topk:
			l = int(labels[idx])
			statistic[l] += 1.0
		label_weight = statistic / statistic.max()
		instance_weight = label_weight[labels]
		return instance_weight

	def query(self, n):
		assert n <= len(self.idxs_lb)
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		unlabeled_dataset = [self.train_dataset[i] for i in idxs_unlabeled.tolist()]
		probs = self.predict_prob(unlabeled_dataset)
		if self.prev_pred is not None:
			U_probs = probs + self.args.cd_ratio * (probs - self.prev_pred)
		else:
			U_probs = probs
		probs_sorted, _ = U_probs.sort(descending=True)
		U = probs_sorted[:, 0] - probs_sorted[:,1]
		_, pseudo_lbls = torch.max(probs, 1)

		if self.args.lambda_topk > 0:
			topk = self.args.topk
			instance_weight = self.label_statistic(pseudo_lbls, U, probs.size(1), topk=topk, descending=True)
			U = U / U.max()
			U = U + self.args.lambda_topk * instance_weight

		idx_selected = U.sort()[1][:n]
		idxs_unlabeled = idxs_unlabeled[idx_selected]
		self.prev_pred = np.delete(probs, idx_selected, axis=0)
		return idxs_unlabeled
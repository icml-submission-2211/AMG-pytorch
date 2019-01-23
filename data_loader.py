#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils import data

from utils import *

class DataFolder(data.Dataset):
	"""Load Data for Iterator. """
	def __init__(self, support, support_t, u_indices, v_indices, labels,
					   u_features_side, v_features_side, class_cnt):
		"""Initializes image paths and preprocessing module."""

		self.support = support.reshape(support.shape[0], class_cnt, -1)
		self.support_t = support_t.reshape(support_t.shape[0], class_cnt, -1)
		self.u_indices = u_indices
		self.v_indices = v_indices
		self.labels = labels
		self.u_features_side = u_features_side
		self.v_features_side = v_features_side

	def __getitem__(self, index):
		"""Reads an Data and Neg Sample from a file and returns."""
		u_index = self.u_indices[index]
		v_index = self.v_indices[index]
		label = self.labels[index]

		support = self.support[u_index]
		support_t = self.support_t[v_index]
		u_features_side = self.u_features_side[u_index]
		v_features_side = self.v_features_side[v_index]

		return u_index, v_index, label, support, support_t, u_features_side, v_features_side

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.labels)


def get_loader(data_type):
	"""Builds and returns Dataloader."""
	SYM = True
	DATASET = data_type
	datasplit_path = 'data/' + DATASET + '/withfeatures.pickle'

	if DATASET == 'flixster' or DATASET == 'douban' or DATASET == 'yahoo_music':
	    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
	        val_labels, val_u_indices, val_v_indices, test_labels, \
	        test_u_indices, test_v_indices, class_values = load_data_monti(data_type)
	else:
	    print("Using random dataset split ...")
	    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
	        val_labels, val_u_indices, val_v_indices, test_labels, \
	        test_u_indices, test_v_indices, class_values = create_trainvaltest_split(data_type, datasplit_path=datasplit_path)
	'''
	elif DATASET == 'ml_100k':
	    print("Using official MovieLens dataset split u1.base/u1.test with 20% validation set size...")
	    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
	        val_labels, val_u_indices, val_v_indices, test_labels, \
	        test_u_indices, test_v_indices, class_values = load_official_trainvaltest_split(data_type)
	'''

	num_users, num_items = adj_train.shape

	print("Normalizing feature vectors...")
	u_features_side = normalize_features(u_features)
	v_features_side = normalize_features(v_features)

	u_features_side, v_features_side = preprocess_user_item_features(u_features_side, v_features_side)

	# 943x41, 1682x41
	u_features_side = np.array(u_features_side.todense(), dtype=np.float32)
	v_features_side = np.array(v_features_side.todense(), dtype=np.float32)

	num_side_features = u_features_side.shape[1]

	# node id's for node input features
	id_csr_u = sp.identity(num_users, format='csr')
	id_csr_v = sp.identity(num_items, format='csr')

	u_features, v_features = preprocess_user_item_features(id_csr_u, id_csr_v)

	u_features = u_features.toarray()
	v_features = v_features.toarray()

	num_features = u_features.shape[1]

	return num_users, num_items, num_side_features, num_features, \
		   u_features, v_features, u_features_side, v_features_side, class_values

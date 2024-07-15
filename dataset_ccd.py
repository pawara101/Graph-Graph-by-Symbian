import glob
import random
import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms

import scipy.io as io

from torch_geometric.data import Data 
from torch_geometric.data import InMemoryDataset 

from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
import time 

class Dataset(Dataset):
	def __init__(self, dataset_path, img_dataset_path, split_path, ref_interval, objmap_file, annofile, training):
		
		"""  
		Input: 
		dataset_path: Path to the video frames 
		img_dataset_path: Path to the I3D frame features
		split_path: Path to the split files 
		ref_interval: Number of frames temporally connected in frame-level graph
		objmap_file: The file mapping labels to label names for objects
		annofile: File containing TOA annotations 
		training: Flag to determine train/test phase 
		
		"""
		self.training = training
		self.img_dataset_path = img_dataset_path
		self.feature_paths, self.vid_labels = self._extract_feature_paths(dataset_path, split_path, training)
		self.transform = transforms.Compose([transforms.ToTensor(),])
		self.ref_interval = ref_interval   # Reference interval in the frame-level graph
		self.temporal_ref = 1   # Connections between objects of previous frames 
		self.dilation_factor = 1   # To increase temporal receptive field in frame-level graph
		self.topk = 10
		self.frame_stats_path = "data/ccd/frames_stats"  # files containing (height, width) for every frame 
		self.n_frames = 50
		self.toa_dict = self.get_toa_all(annofile)   #Get the time-to-accident for all videos (used of evaluation in test phase)

		#Obj label to word embeddings 
		self.idx_to_classes_obj = json.load(open(objmap_file))
		self.nlp = spacy.load('en_core_web_md', disable=['ner', 'parser'])
		self.obj_embeddings = torch.from_numpy(np.array([self.nlp(obj).vector for obj in self.idx_to_classes_obj]))

	def _extract_feature_paths(self, dataset_path, split_path="splits_ccd/", training=True):
		
		""" Function to extract paths to frames given the specified train/test split 
		Input: 
		dataset_path: Path to the video frames 
		split_path: Path to the split files 
		training: Flag to determine train/test phase 
		
		Returns: 
		feature_paths: List of all the video paths in a split
		vid_labels: List of labels for all the videos in the split
		
		"""

		fn = "train.txt" if training else "test.txt"
		split_path = os.path.join(split_path, fn)
		with open(split_path) as file:
			lines = file.read().splitlines()
		feature_paths = []
		vid_labels = []
		for line in lines:
			if training:
				feature_paths += [os.path.join(dataset_path, line.split(" ")[0].split(".")[0])]
				vid_labels += [int(line.split(" ")[1])]
			else: 
				feature_paths += [os.path.join(dataset_path, line.split(" ")[0].split(".")[0])]
				vid_labels += [int(line.split(" ")[1])]
		return feature_paths, vid_labels

	def _frame_number(self, feat_path):

		""" Function to extract the frame number from the input 
		Input: 
		feat_path: Input path for a frame 
		
		Returns: 
		Integer of the frame number (temporal position of a frame in a video)
		
		"""
		return int(feat_path.split('/')[-1].split('.mat')[0].split('_')[-1])

	def get_toa_all(self, annofile):
		
		""" Taken from baseline - https://github.com/Cogito2012/UString/blob/master/src/DataLoader.py
		Function to TOA (time-to-accident) for the videos given the annotation file 
		Input: 
		annofile: Path to the annotation file provided by the dataset 
		
		Returns: 
		toa_dict: Dictionary containing the TOA for every video 
		
		"""

		toa_dict = {}
		annoData = self.read_anno_file(annofile)
		for anno in annoData:
			labels = np.array(anno['label'], dtype=np.int8)
			toa = np.where(labels == 1)[0][0]
			toa = min(max(1, toa), self.n_frames-1) 
			toa_dict[anno['vid']] = toa
		return toa_dict

	def read_anno_file(self, anno_file):
		
		# Taken from baseline - https://github.com/Cogito2012/UString/blob/master/src/DataLoader.py		
		""" Function to read the annotation file for TOA (time-to-accident) 
		Input: 
		annofile: Path to the annotation file provided by the dataset 
		
		Returns: 
		toa_dict: Dictionary containing the TOA for every video 
		
		"""
		assert os.path.exists(anno_file), "Annotation file does not exist! %s"%(anno_file)
		result = []
		with open(anno_file, 'r') as f:
			for line in f.readlines():
				items = {}
				items['vid'] = line.strip().split(',[')[0]
				labels = line.strip().split(',[')[1].split('],')[0]
				items['label'] = [int(val) for val in labels.split(',')]
				assert sum(items['label']) > 0, 'invalid accident annotation!'
				others = line.strip().split(',[')[1].split('],')[1].split(',')
				items['startframe'], items['vid_ytb'], items['lighting'], items['weather'], items['ego_involve'] = others
				result.append(items)
		f.close()
		return result

	def __getitem__(self, index):

		feature_path = self.feature_paths[index % (len(self))]
		curr_vid_label = self.vid_labels[index % (len(self))]
		if curr_vid_label == 1:
			curr_toa = self.toa_dict[feature_path.split("/")[-1]]
		else: 
			curr_toa = self.n_frames + 1

		#Load the data.npy (for features) and det.npy (for bounding boxes) files
		all_feat = torch.from_numpy(np.load(f"{feature_path}/data.npy")[:, 1:, :])
		all_bbox = torch.from_numpy(np.load(f"{feature_path}/det.npy")).float()	   #(x1, y1, x2, y2, prob, cls) bottom left and top right coordinates
		 
		#Reading frame (i3d) features for the frames
		img_file = os.path.join(self.img_dataset_path, feature_path.split('/')[-2], feature_path.split('/')[-1] + '.npy')
		all_img_feat = self.transform(np.load(img_file)).squeeze(0)
		
		#Reading frame stats file
		frame_stats_file = os.path.join(self.frame_stats_path, feature_path.split('/')[-2], feature_path.split('/')[-1] + '.npy')
		frame_stats = torch.from_numpy(np.load(frame_stats_file)).float()
		
		#Calculating the bbox centers 
		cx, cy = (all_bbox[:, :, 0] + all_bbox[:, :, 2])/2, (all_bbox[:, :, 1] + all_bbox[:, :, 3])/2 
		all_obj_centers = torch.cat((cx.unsqueeze(2), cy.unsqueeze(2)), 2).float()
		
		#Obj label indexes 
		all_obj_label_idxs = all_bbox[:, :, -1].long()
		
		# ----------  Building the object-level spatio-temporal graph --------------- 
		video_data, edge_weights, edge_embeddings, obj_vis_feat = [], [], [], []
		start_idx = 0
		actual_frame_idx = 0
		obj_refs = {}
		num_objs_list = []
		
		for i in range(all_feat.shape[0]):
			
			obj_label_idxs, obj_feat, obj_centers = all_obj_label_idxs[i], all_feat[i], all_obj_centers[i] 
			num_objs = len(obj_label_idxs) 

			#Create the adj list (making big disconnected graph for one video) to run it in the library
			#relation pair idxs
			source_nodes = torch.arange(obj_label_idxs.shape[0]).unsqueeze(1).repeat(1, obj_label_idxs.shape[0]).flatten()
			target_nodes = torch.arange(obj_label_idxs.shape[0]).unsqueeze(0).repeat(1, obj_label_idxs.shape[0]).flatten()
			adj_list = torch.cat((source_nodes.unsqueeze(1), target_nodes.unsqueeze(1)), 1)
			repeat_idx = torch.where(adj_list[:,0]!=adj_list[:, 1])[0]  #removing the self loops from the adj list
			adj_list = adj_list[repeat_idx]
			adj_list = torch.LongTensor(adj_list)
			adj_list = adj_list.permute((1, 0))
			
			#Edge embeddings - dist_rel, obj centers of source, obj centers of target
			source_nodes, target_nodes = adj_list[0, :], adj_list[1, :]          
			obj_centers = obj_centers/frame_stats[i]   # normalize with frame height and width (from frame stats files) 
			source_centers = obj_centers[source_nodes]
			target_centers = obj_centers[target_nodes]
			
			dist_mat = torch.cdist(source_centers, target_centers).float()
			dist_mat = F.softmax(-dist_mat, dim=-1)
			dist_rel = dist_mat[adj_list[0, :], adj_list[1,:]].unsqueeze(1)

			#Concatenate to create the initial edge embeddings
			edge_embed = torch.cat((source_centers, target_centers, dist_rel), 1)

			adj_list += start_idx

			target = curr_vid_label             #Target
			
			#Adding obj label embeddings to the node features 
			label_embeddings = self.obj_embeddings[obj_label_idxs]
			x_embed = torch.cat((obj_feat, label_embeddings), 1)
			frame_data = Data(x=x_embed, edge_index=adj_list, y=target)

			video_data.append(frame_data)
			edge_embeddings.append(edge_embed)
			obj_vis_feat.append(obj_feat)

			obj_refs['f_'+str(actual_frame_idx)] = {}
			unique_obj_idxs = torch.unique(obj_label_idxs) 
			for obj_ in unique_obj_idxs: 
				idxx = torch.where(obj_label_idxs == obj_)[0] + start_idx
				obj_refs['f_'+str(actual_frame_idx)][obj_.item()] = idxx

			start_idx += num_objs

			#keep track of num of objects for pooling to create graph-embedding later
			num_objs_list += [torch.zeros(num_objs) + actual_frame_idx]
			actual_frame_idx += 1

		data, slices = InMemoryDataset.collate(video_data)
		edge_embeddings = torch.cat(edge_embeddings)
		obj_vis_feat = torch.cat(obj_vis_feat)
		num_objs_list = torch.cat(num_objs_list)

		#Generate the temporal connections adjacency matrix for the object-level graph 
		temporal_adj_list, temporal_edge_w = [], []
		for i, (key_, value_) in enumerate(obj_refs.items()):
			temporal_ref = min(i, self.temporal_ref)
			curr_frame = obj_refs['f_'+str(i)]
			for t_ in range(1, temporal_ref+1):
				prev_frame = obj_refs['f_'+str(i - t_)]
				for obj_l, t_n in curr_frame.items():
					if obj_l in prev_frame:
						s_n = prev_frame[obj_l]    #source node idxs
						num_s = s_n.shape[0] 
						#repeat all sources nodes length of target times - for example if source = [1, 2] and len(target)=2 it becomes [1,1,2,2]
						s_n = s_n.unsqueeze(0).repeat(len(t_n), 1).T.reshape(-1, 1)   
						t_n = t_n.repeat(num_s).unsqueeze(1)      #repeat target source times directly
						s_t = torch.cat((s_n, t_n), 1)
						temporal_adj_list += s_t.tolist()
		temporal_adj_list = torch.Tensor(temporal_adj_list).permute((1, 0)).long()

		#Generate the frame-level adjacency matrix 
		video_adj_list = []
		for i in range(len(video_data)): 
			bw_ref_interval = min(i, self.ref_interval)
			for j in range(1, bw_ref_interval+1, self.dilation_factor): 
				video_adj_list += [[i-j, i]]   #adding previous ref_interval neighbors
		video_adj_list = torch.Tensor(video_adj_list).permute((1, 0)).long()
		
		return data.x, data.edge_index, data.y, all_img_feat, video_adj_list, edge_embeddings, temporal_adj_list, obj_vis_feat, num_objs_list, curr_toa

	def __len__(self):
		return len(self.feature_paths)
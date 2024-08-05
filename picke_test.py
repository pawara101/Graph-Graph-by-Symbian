import torch
import numpy as np

frame_stats_file = "i3d_feat_sample.npy"
# import pickle
#
# with open("data/annotation-Mar9th-25fps.pkl", "rb") as f:
#     deserialized_dict = pickle.load(f)
#     print(deserialized_dict)


# i3d_feat = torch.from_numpy(np.load(frame_stats_file)).float()
#
# print(i3d_feat.shape)


## Obj Feat
test_all_data = np.load('b001_000490.npz')

test_data = test_all_data['data']
test_labels = test_all_data['labels']

print("Labels ",test_labels)

# print("data ",test_data[0][0])
#
# print("data shape",len(test_data[0][0]))
print(test_all_data)
print(test_all_data['ID'])
# print(test_all_data['det'])
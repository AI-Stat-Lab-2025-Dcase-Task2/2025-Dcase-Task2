import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.data.dataset import custom_collate_fn, CustomDataset

from model.ssmodel.models import audio_feature_extractor
from model.utils.pool import attentive_statistics_pooling

device = 'cuda' if torch.cuda.is_available() else 'cpu'

audio_feature_extractor = audio_feature_extractor(
    'eat',
    True
).to(device)

att_pool = attentive_statistics_pooling(
    input_dim=768,
    embed_dim=768,
    ds_rate=3,
    output_dim=512,
    dropout=0.2
).to(device)

dataset = CustomDataset('data/noA.csv', augment=False, freqm=80, timem=80)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,
    shuffle=False,
    collate_fn=custom_collate_fn,
    drop_last=False
)

features = []
for i, batch in tqdm(enumerate(dataloader), desc='processing files'):
    with torch.no_grad():
        xs = torch.stack(batch['source'], dim=0).to(device)
        feats = audio_feature_extractor(xs)
        feats = att_pool(feats)
    features.append(feats.cpu().numpy())

features = np.concatenate(features, axis=0)
np.save('data/noA_features.npy', features)
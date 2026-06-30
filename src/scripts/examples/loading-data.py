import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from mvswm.data import Spacecraft
from mvswm.model import SlidingWindowFeatureExtractor, VariationalAutoencoder

feature_extractor = SlidingWindowFeatureExtractor(window_size=60, step=60)

lim = 100000

messenger = Spacecraft("MESSENGER")
input_data = messenger.data[["|B| [nT]", "Br [nT]", "Bt [nT]", "Bn [nT]"]].to_numpy()[
    :lim
]
features = feature_extractor.calculate_features(input_data)


features_tensor = torch.tensor(features, dtype=torch.float32)

# parker = Spacecraft("Parker Solar Probe")
# parker_data = parker.data[["|B| [nT]", "Br [nT]", "Bt [nT]", "Bn [nT]"]].to_numpy()[:lim]
# parker_features = feature_extractor.calculate_features(parker_data)

dataset = TensorDataset(features_tensor)
data = DataLoader(dataset, batch_size=128, shuffle=True)

model = VariationalAutoencoder(48)
model.train_model(features, epochs=100)

fig, ax = plt.subplots()

model.plot_latent_space(features, ax=ax, color="black")
# model.plot_latent_space(parker_features, ax=ax, color="indianred")

plt.show()

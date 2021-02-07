from model.db_spectrogram import DBSpectrogram
from model.batch_norm import BatchNorm
from model.blstm import BLSTM
from model.embedding import Embedding

class HierarchicalModel(nn.Module):
    def __init__(self, num_features, num_audio_channels, hidden_size, num_layers, dropout, num_sources):
        super(HierarchicalModel, self).__init__()
        self.db_spectrogram = DBSpectrogram()
        self.batch_norm = BatchNorm(num_features)
        self.blstm = BLSTM(num_features * num_audio_channels, hidden_size, num_layers, dropout)
        self.embedding = Embedding(num_features, hidden_size * 2, num_sources, num_audio_channels)

    def forward(self, data):
        mix = data

        data = self.db_spectrogram(mix)
        data = self.batch_norm(data)
        data = self.blstm(data)
        mask = self.embedding(data)

        estimates = mix.unsqueeze(-1) * mask
        return estimates

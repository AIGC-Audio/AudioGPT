import torch
from torch import nn
from torch.nn import functional as F


class Conv1d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv1d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm1d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class LandmarkHubertSyncNet(nn.Module):
    def __init__(self, lm_dim=60):
        super(LandmarkHubertSyncNet, self).__init__()

        # hubert = torch.rand(B, 1, , t=10)
        self.hubert_encoder = nn.Sequential(
            Conv1d(1024, 128, kernel_size=3, stride=1, padding=1),

            Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv1d(512, 512, kernel_size=1, stride=1, padding=0),)


        # mouth = torch.rand(B, 20*3, t=5)
        self.mouth_encoder = nn.Sequential(
            Conv1d(lm_dim, 96, kernel_size=3, stride=1, padding=1),

            Conv1d(96, 128, kernel_size=3, stride=1, padding=1),
            Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv1d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv1d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            Conv1d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv1d(512, 512, kernel_size=1, stride=1, padding=0),)
        self.lm_dim = lm_dim
        self.logloss = nn.BCELoss()
    def forward(self, hubert, mouth_lm): 
        # hubert := (B, T=10, C=1024)
        # mouth_lm3d := (B, T=5, C=60)
        hubert = hubert.transpose(1,2)
        mouth_lm = mouth_lm.transpose(1,2)
        mouth_embedding = self.mouth_encoder(mouth_lm)
        audio_embedding = self.hubert_encoder(hubert)
        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        mouth_embedding = mouth_embedding.view(mouth_embedding.size(0), -1)
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        mouth_embedding = F.normalize(mouth_embedding, p=2, dim=1)
        return audio_embedding, mouth_embedding

    def cal_sync_loss(self, audio_embedding, mouth_embedding, label):
        if isinstance(label, torch.Tensor):
            gt_d = label.float().view(-1,1).to(audio_embedding.device)
        else:
            gt_d = (torch.ones([audio_embedding.shape[0],1]) * label).float().to(audio_embedding.device) # int
        d = nn.functional.cosine_similarity(audio_embedding, mouth_embedding)
        loss = self.logloss(d.unsqueeze(1), gt_d)
        return loss, d

    def cal_cosine_similarity(self, audio_embedding, mouth_embedding):
        d = nn.functional.cosine_similarity(audio_embedding, mouth_embedding)
        return d


if __name__ == '__main__':
    syncnet = LandmarkHubertSyncNet(lm_dim=204)
    hubert = torch.rand(2, 10, 1024)
    lm = torch.rand(2, 5, 204)
    mel_embedding, exp_embedding = syncnet(hubert, lm)
    label = torch.tensor([1., 0.])
    loss = syncnet.cal_sync_loss(mel_embedding, exp_embedding, label)
    print(" ")
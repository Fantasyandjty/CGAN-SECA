import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter


class VirtualBatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = Parameter(torch.normal(mean=1.0, std=0.02, size=(1, num_features, 1)))
        self.beta = Parameter(torch.zeros(1, num_features, 1))

    def get_stats(self, x):
        mean = x.mean(2, keepdim=True).mean(0, keepdim=True)
        mean_sq = (x ** 2).mean(2, keepdim=True).mean(0, keepdim=True)
        return mean, mean_sq

    def forward(self, x, ref_mean, ref_mean_sq):
        mean, mean_sq = self.get_stats(x)
        if ref_mean is None or ref_mean_sq is None:
            mean = mean.clone().detach()
            mean_sq = mean_sq.clone().detach()
            out = self.normalize(x, mean, mean_sq)
        else:
            batch_size = x.size(0)
            new_coeff = 1. / (batch_size + 1.)
            old_coeff = 1. - new_coeff
            mean = new_coeff * mean + old_coeff * ref_mean
            mean_sq = new_coeff * mean_sq + old_coeff * ref_mean_sq
            out = self.normalize(x, mean, mean_sq)
        return out, mean, mean_sq

    def normalize(self, x, mean, mean_sq):
        assert len(x.size()) == 3
        std = torch.sqrt(self.eps + mean_sq - mean ** 2)
        x = (x - mean) / std
        return x * self.gamma + self.beta

    def __repr__(self):
        return '{}(num_features={}, eps={})'.format(
            self.__class__.__name__, self.num_features, self.eps)


class IdentityVBN(Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features

    def forward(self, x, ref_mean, ref_mean_sq):
        return x, None, None


class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16, init_mode="identity", init_bias=4.0):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.sigmoid = nn.Sigmoid()
        self.init_mode = init_mode

        if init_mode == "identity":
            nn.init.zeros_(self.fc2.weight)
            nn.init.constant_(self.fc2.bias, init_bias)
        elif init_mode != "original":
            raise ValueError("init_mode 只能是 'identity' 或 'original'")

    def forward(self, x):
        b, c, _ = x.size()
        s = F.adaptive_avg_pool1d(x, 1).view(b, c)
        s = F.relu(self.fc1(s))
        s = self.fc2(s)
        e = self.sigmoid(s).view(b, c, 1)
        return x * e


class IdentitySE(nn.Module):
    def __init__(self, channels, reduction=16, **kw):
        super().__init__()

    def forward(self, x):
        return x


def _make_se(channels, use_seca, reduction=16, init_mode="identity"):
    return (SEBlock(channels, reduction, init_mode=init_mode)
            if use_seca else IdentitySE(channels))


def _make_vbn(channels, use_vbn):
    return VirtualBatchNorm1d(channels) if use_vbn else IdentityVBN(channels)


class Generator(nn.Module):
    def __init__(self, use_seca=True, width=1.0, bottleneck="none",
                 bottleneck_layers=2, se_init="identity"):
        super().__init__()
        self.use_seca = use_seca
        self.bottleneck_kind = bottleneck
        self.se_init = se_init

        def c(n):
            return max(4, int(round(n * width)))

        ch = [c(16), c(32), c(32), c(64), c(64), c(128),
              c(128), c(256), c(256), c(512), c(1024)]
        self.ch = ch
        self.z_ch = ch[10]


        enc_in = [1] + ch[:-1]
        self.encs = nn.ModuleList([
            nn.Conv1d(enc_in[i], ch[i], 32, 2, 15) for i in range(11)])
        self.enc_se = nn.ModuleList([_make_se(ch[i], use_seca, init_mode=se_init)
                                     for i in range(11)])
        self.enc_nl = nn.ModuleList([nn.PReLU() for _ in range(11)])


        if bottleneck in ("gru", "lstm"):
            RNN = nn.GRU if bottleneck == "gru" else nn.LSTM
            self.rnn = RNN(ch[10], ch[10] // 2, num_layers=bottleneck_layers,
                           batch_first=True, bidirectional=True)
            self.rnn_proj = nn.Conv1d(ch[10], ch[10], 1)
        else:
            self.rnn = None


        self.decs = nn.ModuleList()
        self.dec_se = nn.ModuleList()
        self.dec_nl = nn.ModuleList()
        for k in range(9, -1, -1):
            in_c = 2 * ch[k + 1]
            out_c = ch[k]
            self.decs.append(nn.ConvTranspose1d(in_c, out_c, 32, 2, 15))
            self.dec_se.append(_make_se(out_c, use_seca, init_mode=se_init))
            self.dec_nl.append(nn.PReLU())

        self.dec_final = nn.ConvTranspose1d(2 * ch[0], 1, 32, 2, 15)
        self.dec_tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x, z):
        e = []
        h = x
        for i in range(11):
            h = self.encs[i](h if i == 0 else self.enc_nl[i - 1](e[i - 1]))
            h = self.enc_se[i](h)
            e.append(h)

        c = self.enc_nl[10](e[10])

        if self.rnn is not None:
            r, _ = self.rnn(c.transpose(1, 2))
            c = c + self.rnn_proj(r.transpose(1, 2))

        d = torch.cat((c, z), dim=1)

        for j, k in enumerate(range(9, -1, -1)):
            d = self.decs[j](d)
            d = self.dec_se[j](d)
            d = self.dec_nl[j](torch.cat((d, e[k]), dim=1))

        return self.dec_tanh(self.dec_final(d))


class Discriminator(nn.Module):
    def __init__(self, use_seca=True, use_vbn=True, use_dropout=True, dropout_p=0.5,
                 se_init="identity", use_sigmoid=True):
        super().__init__()
        self.use_dropout = use_dropout
        self.use_sigmoid = use_sigmoid
        negative_slope = 0.03
        ch = [32, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 2048]
        in_ch = [2] + ch[:-1]

        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch[i], ch[i], 31, 2, 15) for i in range(11)])
        self.ses = nn.ModuleList([_make_se(ch[i], use_seca, init_mode=se_init)
                                  for i in range(11)])
        self.vbns = nn.ModuleList([_make_vbn(ch[i], use_vbn) for i in range(11)])
        self.lrelus = nn.ModuleList([nn.LeakyReLU(negative_slope) for _ in range(11)])


        self.drop_idx = {2, 5, 8}
        self.dropout = nn.Dropout(dropout_p)

        self.conv_final = nn.Conv1d(ch[10], 1, kernel_size=1, stride=1)
        self.lrelu_final = nn.LeakyReLU(negative_slope)
        self.fully_connected = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    @torch.no_grad()
    def compute_ref_stats(self, ref_x):
        stats = []
        h = ref_x
        for i in range(11):
            h = self.convs[i](h)
            h = self.ses[i](h)
            h, m, msq = self.vbns[i](h, None, None)
            stats.append((m, msq))
            if i < 10:
                h = self.lrelus[i](h)
        return stats


    def _ref_pass(self, ref_x):
        return self.compute_ref_stats(ref_x)

    def forward(self, x, ref_x=None, ref_stats=None):
        if ref_stats is None:
            assert ref_x is not None, "必须提供 ref_x 或 ref_stats"
            ref_stats = self.compute_ref_stats(ref_x)

        h = x
        for i in range(11):
            h = self.convs[i](h)
            h = self.ses[i](h)
            if self.use_dropout and i in self.drop_idx:
                h = self.dropout(h)
            h, _, _ = self.vbns[i](h, ref_stats[i][0], ref_stats[i][1])
            h = self.lrelus[i](h)

        h = self.lrelu_final(self.conv_final(h))
        h = h.squeeze(1)
        out = self.fully_connected(h)
        return self.sigmoid(out) if self.use_sigmoid else out


def build_models(para):
    si = getattr(para, "se_init", "identity")
    use_sig = getattr(para, "adv_loss", "lsgan") == "lsgan"
    G = Generator(use_seca=para.use_seca, se_init=si)
    D = Discriminator(use_seca=para.use_seca,
                      use_vbn=para.use_vbn,
                      use_dropout=para.use_dropout,
                      dropout_p=para.dropout_p,
                      se_init=si, use_sigmoid=use_sig)
    return G, D


if __name__ == "__main__":
    for seca in (True, False):
        g = Generator(use_seca=seca)
        d = Discriminator(use_seca=seca, use_vbn=True, use_dropout=True)
        x = torch.randn(2, 1, 16384)
        z = torch.randn(2, g.z_ch, 8)
        print("use_seca=%s | G %.2fM  D %.2fM | G_out %s  D_out %s" % (
            seca,
            sum(p.numel() for p in g.parameters()) / 1e6,
            sum(p.numel() for p in d.parameters()) / 1e6,
            tuple(g(x, z).shape),
            tuple(d(torch.randn(2, 2, 16384), torch.randn(2, 2, 16384)).shape)))

    d = Discriminator()
    print("batch=1 D_out:", tuple(d(torch.randn(1, 2, 16384), torch.randn(4, 2, 16384)).shape))

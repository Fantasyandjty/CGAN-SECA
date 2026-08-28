import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialLoss(nn.Module):

    def __init__(self, mode="ralsgan"):
        super().__init__()
        assert mode in ("lsgan", "ralsgan", "hinge")
        self.mode = mode

    def d_loss(self, real_out, fake_out):
        if self.mode == "lsgan":
            return torch.mean((real_out - 1.0) ** 2) + torch.mean(fake_out ** 2)

        if self.mode == "ralsgan":

            r_mean = real_out.mean()
            f_mean = fake_out.mean()
            return (torch.mean((real_out - f_mean - 1.0) ** 2) +
                    torch.mean((fake_out - r_mean + 1.0) ** 2)) / 2


        return (torch.mean(F.relu(1.0 - real_out)) +
                torch.mean(F.relu(1.0 + fake_out)))

    def g_loss(self, real_out, fake_out):
        if self.mode != "ralsgan" and real_out is None:
            real_out = fake_out.detach()
        if self.mode == "lsgan":
            return torch.mean((fake_out - 1.0) ** 2)

        if self.mode == "ralsgan":
            r_mean = real_out.mean()
            f_mean = fake_out.mean()
            return (torch.mean((real_out - f_mean + 1.0) ** 2) +
                    torch.mean((fake_out - r_mean - 1.0) ** 2)) / 2

        return -torch.mean(fake_out)


class MultiResSTFTLoss(nn.Module):

    def __init__(self, fft_sizes=(512, 1024, 2048),
                 hop_sizes=(128, 256, 512),
                 win_lengths=(512, 1024, 2048)):
        super().__init__()
        self.cfg = list(zip(fft_sizes, hop_sizes, win_lengths))
        for i, (_, _, wl) in enumerate(self.cfg):
            self.register_buffer("w%d" % i, torch.hann_window(wl))

    def forward(self, x, y):
        x, y = x.squeeze(1), y.squeeze(1)
        total = 0.0
        for i, (n_fft, hop, wl) in enumerate(self.cfg):
            win = getattr(self, "w%d" % i)
            X = torch.stft(x, n_fft, hop, wl, window=win, return_complex=True)
            Y = torch.stft(y, n_fft, hop, wl, window=win, return_complex=True)
            Xm, Ym = X.abs(), Y.abs()
            sc = torch.norm(Ym - Xm, p="fro") / (torch.norm(Ym, p="fro") + 1e-8)
            mag = F.l1_loss(torch.log(Xm + 1e-7), torch.log(Ym + 1e-7))
            total = total + sc + mag
        return total / len(self.cfg)


class ComplexSpecLoss(nn.Module):

    def __init__(self, n_fft=512, hop=128, power_compress=0.3, w_mag=1.0, w_ri=1.0):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.p = power_compress
        self.w_mag, self.w_ri = w_mag, w_ri
        self.register_buffer("win", torch.hann_window(n_fft))

    def _spec(self, x):
        X = torch.stft(x.squeeze(1), self.n_fft, self.hop, self.n_fft,
                       window=self.win, return_complex=True)
        mag = X.abs().clamp_min(1e-8)
        comp = mag ** self.p
        phase = X / mag
        return comp, comp * phase.real, comp * phase.imag

    def forward(self, x, y):
        mx, rx, ix = self._spec(x)
        my, ry, iy = self._spec(y)
        loss_mag = F.l1_loss(mx, my)
        loss_ri = F.l1_loss(rx, ry) + F.l1_loss(ix, iy)
        return self.w_mag * loss_mag + self.w_ri * loss_ri


def feature_matching_loss(feats_real, feats_fake):
    loss = 0.0
    n = 0
    for fr, ff in zip(feats_real, feats_fake):
        loss = loss + F.l1_loss(ff, fr.detach())
        n += 1
    return loss / max(n, 1)


class GeneratorLoss(nn.Module):

    def __init__(self, para):
        super().__init__()
        self.para = para
        self.adv = AdversarialLoss(getattr(para, "adv_loss", "lsgan"))
        self.stft = (MultiResSTFTLoss(para.stft_fft_sizes, para.stft_hop_sizes,
                                      para.stft_win_lengths)
                     if getattr(para, "w_stft", 0) > 0 else None)
        self.cplx = (ComplexSpecLoss(power_compress=getattr(para, "power_compress", 0.3))
                     if getattr(para, "w_cplx", 0) > 0 else None)

    def forward(self, fake, clean, real_out=None, fake_out=None):
        p = self.para
        terms = {}
        total = 0.0

        if fake_out is not None:
            a = p.w_adv * self.adv.g_loss(real_out, fake_out)
            terms["adv"] = a
            total = total + a

        l1 = p.w_l1 * torch.mean(torch.abs(fake - clean))
        terms["l1"] = l1
        total = total + l1

        if self.stft is not None:
            s = p.w_stft * self.stft(fake, clean)
            terms["stft"] = s
            total = total + s

        if self.cplx is not None:
            c = p.w_cplx * self.cplx(fake, clean)
            terms["cplx"] = c
            total = total + c

        return total, terms


if __name__ == "__main__":
    x = torch.randn(2, 1, 16384)
    y = torch.randn(2, 1, 16384)
    for m in ("lsgan", "ralsgan", "hinge"):
        a = AdversarialLoss(m)
        r, f = torch.rand(4, 1), torch.rand(4, 1)
        print("%-8s d=%.4f g=%.4f" % (m, a.d_loss(r, f).item(), a.g_loss(r, f).item()))
    print("stft  =", MultiResSTFTLoss()(x, y).item())
    print("cplx  =", ComplexSpecLoss()(x, y).item())

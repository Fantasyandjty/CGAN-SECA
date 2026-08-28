import os
import csv
import copy
import time
import math
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hparams import hparams
from dataset import SEGAN_Dataset
from model import Generator, Discriminator
from losses import AdversarialLoss, MultiResSTFTLoss, ComplexSpecLoss


class RandomSubsetSampler(torch.utils.data.Sampler):

    def __init__(self, n, ratio, seed=0):
        self.n = n
        self.ratio = ratio
        self.g = torch.Generator().manual_seed(seed)

    def __iter__(self):
        k = len(self)
        return iter(torch.randperm(self.n, generator=self.g)[:k].tolist())

    def __len__(self):
        return max(1, int(self.n * self.ratio))


def set_seed(seed, deterministic=True):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EMA:

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for s, m in zip(self.shadow.state_dict().values(),
                        model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                s.copy_(m)

    def state_dict(self):
        return self.shadow.state_dict()


def augment(clean, noisy, remix=False, gain_aug=False, gen=None):
    if remix:
        noise = noisy - clean
        idx = torch.randperm(noise.size(0), device=noise.device, generator=gen)
        noisy = clean + noise[idx]

    if gain_aug:
        g = torch.empty(clean.size(0), 1, 1, device=clean.device).uniform_(-6, 6)
        g = 10 ** (g / 20)
        clean = clean * g
        noisy = noisy * g
        peak = noisy.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        scale = torch.clamp(0.99 / peak, max=1.0)
        clean, noisy = clean * scale, noisy * scale

    return clean, noisy


def make_scheduler(opt, kind, n_epoch, steps_per_epoch, warmup=500):
    if kind == "cosine":
        total = n_epoch * steps_per_epoch

        def fn(step):
            if step < warmup:
                return step / max(warmup, 1)
            p = (step - warmup) / max(total - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        return torch.optim.lr_scheduler.LambdaLR(opt, fn)
    return None


@torch.no_grad()
def validate(G, loader, device, para):
    G.eval()
    tot, n = 0.0, 0
    for clean, noisy in loader:
        clean, noisy = clean.to(device), noisy.to(device)
        z = torch.zeros(clean.size(0), G.z_ch, para.size_z[1], device=device)
        fake = G(noisy, z)
        tot += torch.mean(torch.abs(fake - clean)).item() * clean.size(0)
        n += clean.size(0)
    G.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--no_seca", action="store_true")
    ap.add_argument("--se_init", type=str, default=None,
                    choices=["identity", "original"],
                    help="SE 初始化。identity=修复版(默认)，original=复现原始缺陷实现")
    ap.add_argument("--no_vbn", action="store_true")
    ap.add_argument("--no_dropout", action="store_true")

    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--adv_loss", type=str, default="lsgan",
                    choices=["lsgan", "ralsgan", "hinge"])
    ap.add_argument("--w_cplx", type=float, default=0.0)
    ap.add_argument("--remix", action="store_true")
    ap.add_argument("--gain_aug", action="store_true")
    ap.add_argument("--bottleneck", type=str, default="none",
                    choices=["none", "gru", "lstm"])
    ap.add_argument("--optimizer", type=str, default="rmsprop",
                    choices=["rmsprop", "adam"])
    ap.add_argument("--lr_schedule", type=str, default="none",
                    choices=["none", "cosine"])
    ap.add_argument("--z_zero", action="store_true",
                    help="训练时 z 恒为 0。SEGAN 的隐变量作用存疑，置零常更稳定")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n_epoch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None, help="同时设置 G 与 D 的学习率")
    ap.add_argument("--lr_D", type=float, default=None,
                    help="单独设置判别器学习率。当 d_loss 长期贴近 0（判别器赢太快、"
                         "生成器从对抗损失拿不到梯度）时，把它降到 G 的 1/2~1/4")
    ap.add_argument("--w_stft", type=float, default=None,
                    help="多分辨率 STFT 损失权重。默认 5.0 时该项常占总损失 70%% 以上，"
                         "会挤压对抗项与 L1 的话语权，建议 2.0")
    ap.add_argument("--w_adv", type=float, default=None, help="对抗损失权重")
    ap.add_argument("--reuse_real_out", action="store_true",
                    help="RaLSGAN 的生成器损失需要判别器对真实样本的输出。开启后复用 D 更新前的值，"
                         "每步省一次判别器前向（约 10-15%%）。属于近似：D 每步只走一小步，"
                         "影响很小，但严格来说不等价。追求严格可复现时勿开。")
    ap.add_argument("--ref_refresh", type=int, default=1,
                    help="每隔几步重算一次 VBN 参考统计量。默认 1（每步重算）。"
                         "设为 4 可省下约 15%% 时间。属于近似：参考统计量随 D 权重缓慢变化。")
    ap.add_argument("--d_every", type=int, default=1,
                    help="每隔几步更新一次判别器。设 2 约省 20%%。"
                         "代价：判别器更新变慢，对抗博弈节奏改变。"
                         "配合 D贴零%% 监控使用，若判别器本就偏弱则不要开。")
    ap.add_argument("--subset_ratio", type=float, default=1.0,
                    help="每个 epoch 随机使用的训练切片比例。设 0.5 时单 epoch 耗时减半。"
                         "代价：单 epoch 见到的数据减半，需要相应增加 epoch 数。")
    ap.add_argument("--stft_res", type=int, default=3, choices=[1, 2, 3],
                    help="多分辨率 STFT 损失使用几个分辨率。3->2 约省 5%%，"
                         "去掉最耗时的 2048 点分辨率。")
    ap.add_argument("--ref_batch", type=int, default=None,
                    help="覆盖 hparams 的 ref_batch_size。参考支路每次是一整个 D 前向，"
                         "64->32 约省 8%%。代价：VBN 参考统计量噪声变大。")
    ap.add_argument("--val_every", type=int, default=1,
                    help="每隔几个 epoch 做一次验证集评估。设 5 可省约 4%% 时间。")
    ap.add_argument("--width", type=float, default=1.0, help="通道宽度系数，调试或轻量版用")
    ap.add_argument("--fast", action="store_true",
                    help="关闭 cudnn 确定性、开启 benchmark 自动选算法。"
                         "速度通常提升 20-40%%，代价是不再逐位可复现。"
                         "正式实验若需严格复现请勿开启。")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()

    para = hparams()
    if args.no_seca: para.use_seca = False
    if args.se_init: para.se_init = args.se_init
    if args.no_vbn: para.use_vbn = False
    if args.no_dropout: para.use_dropout = False
    if args.seed is not None: para.seed = args.seed
    if args.n_epoch is not None: para.n_epoch = args.n_epoch
    if args.w_stft is not None: para.w_stft = args.w_stft
    if args.w_adv is not None: para.w_adv = args.w_adv
    if args.lr is not None: para.lr_G = para.lr_D = args.lr
    if args.lr_D is not None: para.lr_D = args.lr_D
    if args.ref_batch is not None: para.ref_batch_size = args.ref_batch
    if args.stft_res < 3:
        k = args.stft_res
        para.stft_fft_sizes = para.stft_fft_sizes[:k]
        para.stft_hop_sizes = para.stft_hop_sizes[:k]
        para.stft_win_lengths = para.stft_win_lengths[:k]
    para.adv_loss = args.adv_loss
    para.w_cplx = args.w_cplx

    tag = args.tag or "_".join(filter(None, [
        para.tag(),
        args.adv_loss if args.adv_loss != "lsgan" else "",
        "ema" if args.ema else "",
        "bn-" + args.bottleneck if args.bottleneck != "none" else "",
        "remix" if args.remix else "",
        "gain" if args.gain_aug else "",
        "cplx" if args.w_cplx > 0 else "",
    ]))
    save_dir = os.path.join(para.save_path, tag)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(para.log_path, exist_ok=True)

    if args.fast:
        para.deterministic = False
    set_seed(para.seed, para.deterministic)
    if args.fast:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[cfg] fast 模式：cudnn.benchmark 已开启，牺牲逐位复现换取速度")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("[cfg] %s\n[cfg] device=%s epochs=%d adv=%s bottleneck=%s" %
          (tag, device, para.n_epoch, args.adv_loss, args.bottleneck))

    G = Generator(use_seca=para.use_seca, bottleneck=args.bottleneck,
                  width=args.width, se_init=para.se_init).to(device)


    use_sigmoid = (args.adv_loss == "lsgan")
    D = Discriminator(use_seca=para.use_seca, use_vbn=para.use_vbn,
                      use_dropout=para.use_dropout, dropout_p=para.dropout_p,
                      se_init=para.se_init, use_sigmoid=use_sigmoid).to(device)
    print("[cfg] adv_loss=%s -> 判别器 %s Sigmoid" %
          (args.adv_loss, "使用" if use_sigmoid else "不使用"))
    print("[cfg] lr_G=%.2e lr_D=%.2e | w_adv=%.2f w_l1=%.1f w_stft=%.2f w_cplx=%.2f" %
          (para.lr_G, para.lr_D, para.w_adv, para.w_l1, para.w_stft, para.w_cplx))
    print("[cfg] G %.2fM  D %.2fM" % (
        sum(p.numel() for p in G.parameters()) / 1e6,
        sum(p.numel() for p in D.parameters()) / 1e6))

    ema = EMA(G, args.ema_decay) if args.ema else None

    def mk_opt(params, lr):
        if args.optimizer == "adam":
            return torch.optim.Adam(params, lr=lr, betas=(0.5, 0.9))
        return torch.optim.RMSprop(params, lr=lr)

    g_opt, d_opt = mk_opt(G.parameters(), para.lr_G), mk_opt(D.parameters(), para.lr_D)

    train_set = SEGAN_Dataset(para, "train")
    valid_set = SEGAN_Dataset(para, "valid")


    nw = para.num_workers
    dl_kw = dict(num_workers=nw, pin_memory=True)
    if nw > 0:
        dl_kw.update(persistent_workers=True, prefetch_factor=4)
    if args.subset_ratio < 1.0:
        sampler = RandomSubsetSampler(len(train_set), args.subset_ratio, para.seed)
        train_loader = DataLoader(train_set, batch_size=para.batch_size,
                                  sampler=sampler, drop_last=True, **dl_kw)
        print("[cfg] 每 epoch 使用 %d/%d 个切片 (%.0f%%)" %
              (len(sampler), len(train_set), args.subset_ratio * 100))
    else:
        train_loader = DataLoader(train_set, batch_size=para.batch_size, shuffle=True,
                                  drop_last=True, **dl_kw)
    valid_loader = DataLoader(valid_set, batch_size=para.batch_size, shuffle=False,
                              **dl_kw)
    print("[data] train %d  valid %d" % (len(train_set), len(valid_set)))

    ref_batch = train_set.ref_batch(para.ref_batch_size, seed=para.seed).to(device)

    adv = AdversarialLoss(args.adv_loss)
    need_real_for_g = (args.adv_loss == "ralsgan")
    stft_loss = (MultiResSTFTLoss(para.stft_fft_sizes, para.stft_hop_sizes,
                                  para.stft_win_lengths).to(device)
                 if para.w_stft > 0 else None)
    cplx_loss = ComplexSpecLoss().to(device) if args.w_cplx > 0 else None

    g_sched = make_scheduler(g_opt, args.lr_schedule, para.n_epoch, len(train_loader))
    d_sched = make_scheduler(d_opt, args.lr_schedule, para.n_epoch, len(train_loader))

    log_csv = os.path.join(para.log_path, tag + ".csv")
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "d", "adv", "l1", "stft", "cplx",
                                "g", "valid_l1", "valid_l1_ema", "lr", "sec",
                                "d_collapse_ratio"])

    best = float("inf")
    aug_gen = torch.Generator(device=device).manual_seed(para.seed)

    for epoch in range(1, para.n_epoch + 1):
        t0 = time.time()
        acc = dict(d=0., adv=0., l1=0., stft=0., cplx=0., g=0.)
        nstep = 0
        n_d_collapse = 0
        ref_stats = None
        last_d = torch.zeros((), device=device)
        last_real_out = None

        for clean, noisy in train_loader:
            clean = clean.to(device, non_blocking=True)
            noisy = noisy.to(device, non_blocking=True)
            clean, noisy = augment(clean, noisy, args.remix, args.gain_aug, aug_gen)

            if args.z_zero:
                z = torch.zeros(clean.size(0), G.z_ch, para.size_z[1], device=device)
            else:
                z = torch.randn(clean.size(0), G.z_ch, para.size_z[1], device=device)


            if ref_stats is None or nstep % args.ref_refresh == 0:
                ref_stats = D.compute_ref_stats(ref_batch)


            fake = G(noisy, z)
            real_pair = torch.cat([clean, noisy], 1)


            update_d = (nstep % args.d_every == 0)
            if update_d:
                D.zero_grad(set_to_none=True)
                real_out = D(real_pair, ref_stats=ref_stats)
                fake_out_d = D(torch.cat([fake.detach(), noisy], 1), ref_stats=ref_stats)
                d_loss = adv.d_loss(real_out, fake_out_d)
                d_loss.backward()
                nn.utils.clip_grad_norm_(D.parameters(), para.grad_clip)
                d_opt.step()
                if d_sched: d_sched.step()
                last_d = d_loss.detach()
                last_real_out = real_out.detach()
            else:
                d_loss = last_d


            G.zero_grad(set_to_none=True)
            fake_out = D(torch.cat([fake, noisy], 1), ref_stats=ref_stats)

            if not need_real_for_g:
                real_out_d = None
            elif args.reuse_real_out and last_real_out is not None:
                real_out_d = last_real_out
            else:
                with torch.no_grad():
                    real_out_d = D(real_pair, ref_stats=ref_stats)

            g_adv = para.w_adv * adv.g_loss(real_out_d, fake_out)
            g_l1 = para.w_l1 * torch.mean(torch.abs(fake - clean))
            g_stft = para.w_stft * stft_loss(fake, clean) if stft_loss else torch.zeros((), device=device)
            g_cplx = args.w_cplx * cplx_loss(fake, clean) if cplx_loss else torch.zeros((), device=device)

            g_loss = g_adv + g_l1 + g_stft + g_cplx
            g_loss.backward()
            nn.utils.clip_grad_norm_(G.parameters(), para.grad_clip)
            g_opt.step()
            if g_sched: g_sched.step()
            if ema: ema.update(G)

            acc["d"] += d_loss.item(); acc["adv"] += g_adv.item()
            acc["l1"] += g_l1.item(); acc["stft"] += g_stft.detach().item()
            acc["cplx"] += g_cplx.detach().item(); acc["g"] += g_loss.item()
            nstep += 1
            if d_loss.item() < 0.05:
                n_d_collapse += 1

            if nstep % 100 == 0:
                print("  ep%d step%d d=%.4f g=%.4f (adv %.3f l1 %.3f stft %.3f cplx %.3f)" %
                      (epoch, nstep, d_loss.item(), g_loss.item(),
                       g_adv.item(), g_l1.item(), g_stft.detach().item(), g_cplx.detach().item()))

        for k in acc: acc[k] /= max(nstep, 1)
        d_collapse_ratio = n_d_collapse / max(nstep, 1)
        do_val = (epoch % args.val_every == 0) or epoch == para.n_epoch
        if do_val:
            v = validate(G, valid_loader, device, para)
            v_ema = validate(ema.shadow, valid_loader, device, para) if ema else float("nan")
        else:
            v = v_ema = float("nan")
        lr_now = g_opt.param_groups[0]["lr"]
        dt = time.time() - t0
        print("[ep %d/%d] d=%.4f g=%.4f valid=%.5f ema=%.5f lr=%.2e %.1fs  D贴零%.0f%%" %
              (epoch, para.n_epoch, acc["d"], acc["g"], v, v_ema, lr_now, dt,
               d_collapse_ratio * 100))
        if d_collapse_ratio > 0.7 and epoch >= 3:
            print("       [警告] 判别器 %.0f%% 的步数上 d_loss<0.05，对抗训练可能已失效。"
                  % (d_collapse_ratio * 100))
            print("              建议降低 --lr_D（如 5e-5）或减小 --w_adv")

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch] + ["%.5f" % acc[k] for k in
                                              ("d", "adv", "l1", "stft", "cplx", "g")] +
                                   ["%.6f" % v, "%.6f" % v_ema, "%.3e" % lr_now,
                                    "%.1f" % dt, "%.4f" % d_collapse_ratio])

        torch.save(G.state_dict(), os.path.join(save_dir, "G_last.pkl"))
        if ema:
            torch.save(ema.state_dict(), os.path.join(save_dir, "G_ema_last.pkl"))
        score = min(v, v_ema) if ema else v
        if do_val and score == score and score < best:
            best = score
            torch.save(G.state_dict(), os.path.join(save_dir, "G_best.pkl"))
            if ema:
                torch.save(ema.state_dict(), os.path.join(save_dir, "G_ema_best.pkl"))
            print("       -> new best %.6f" % best)

    print("[done] best=%.6f | %s" % (best, save_dir))
    if ema:
        print("       评测时优先试 G_ema_best.pkl，通常略优于 G_best.pkl")


if __name__ == "__main__":
    main()

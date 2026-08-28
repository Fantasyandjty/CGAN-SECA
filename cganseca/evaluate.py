import os
import csv
import glob
import argparse

import numpy as np
import torch
import soundfile as sf
import librosa

from hparams import hparams
from dataset import emphasis
from model import Generator
import metrics as M


def enhance_overlap_add(model, noisy, para, device, overlap=0.5, seed=0,
                        z_mode="random", z_avg=4):
    win = para.win_len
    hop = int(win * (1 - overlap)) if overlap > 0 else win
    L = len(noisy)

    n_frames = max(1, int(np.ceil(max(L - win, 0) / hop)) + 1)
    total = (n_frames - 1) * hop + win
    padded = np.pad(np.asarray(noisy, dtype=np.float32), (0, total - L), mode="constant")

    w = np.hanning(win).astype(np.float32) if overlap > 0 else np.ones(win, dtype=np.float32)
    out = np.zeros(total, dtype=np.float64)
    wsum = np.zeros(total, dtype=np.float64)

    g = torch.Generator(device="cpu").manual_seed(seed)
    model.eval()
    with torch.no_grad():
        for i in range(n_frames):
            s = i * hop
            seg = padded[s:s + win]
            seg_e = emphasis(seg, para.emph_coeff, pre=True)
            t = torch.from_numpy(seg_e.astype(np.float32)).view(1, 1, -1).to(device)
            if z_mode == "zero":
                z = torch.zeros(1, model.z_ch, para.size_z[1], device=device)
                y = model(t, z).cpu().numpy()[0, 0]
            elif z_mode == "avg":
                ys = []
                for _ in range(z_avg):
                    z = torch.randn(1, model.z_ch, para.size_z[1], generator=g).to(device)
                    ys.append(model(t, z).cpu().numpy()[0, 0])
                y = np.mean(ys, axis=0)
            else:
                z = torch.randn(1, model.z_ch, para.size_z[1], generator=g).to(device)
                y = model(t, z).cpu().numpy()[0, 0]
            y = emphasis(y, para.emph_coeff, pre=False)
            out[s:s + win] += y * w
            wsum[s:s + win] += w

    out = out / np.maximum(wsum, 1e-8)
    return out[:L].astype(np.float64)


def baseline_signal(mode, clean, noisy, fs=16000):
    n_fft, hop = 512, 128
    C = M.stft(clean, n_fft, hop, n_fft)
    N = M.stft(noisy, n_fft, hop, n_fft)
    k = min(C.shape[1], N.shape[1])
    C, N = C[:, :k], N[:, :k]

    if mode == "noisy":
        return np.asarray(noisy, dtype=np.float64)
    if mode == "oracle_mag_noisy_phase":
        X = np.abs(C) * np.exp(1j * np.angle(N))
    elif mode == "oracle_mag_clean_phase":
        X = np.abs(C) * np.exp(1j * np.angle(C))
    elif mode == "wiener_noisy_phase":
        Pn = np.maximum(np.abs(N) ** 2 - np.abs(C) ** 2, 1e-12)
        G = np.abs(C) ** 2 / (np.abs(C) ** 2 + Pn + 1e-12)
        X = G * np.abs(N) * np.exp(1j * np.angle(N))
    else:
        raise ValueError(mode)
    y = M.istft(X, n_fft, hop, n_fft)
    L = len(clean)
    return np.pad(y, (0, max(0, L - len(y))))[:L].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--mode", type=str, default="model",
                    choices=["model", "noisy", "oracle_mag_noisy_phase",
                             "oracle_mag_clean_phase", "wiener_noisy_phase"])
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--no_seca", action="store_true")
    ap.add_argument("--bottleneck", type=str, default="none",
                    choices=["none", "gru", "lstm"])
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--z_mode", type=str, default="random",
                    choices=["random", "zero", "avg"],
                    help="潜变量模式；默认 random，即每个语音片段独立采样 z~N(0,I)，与论文一致")
    ap.add_argument("--z_avg", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234,
                    help="随机潜变量的基础种子；第 i 条语音使用 seed+i，便于复现实验")
    ap.add_argument("--clean_dir", type=str, default=None)
    ap.add_argument("--noisy_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--save_wav", type=int, default=0, help="保存前 N 条增强音频")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="仅评测前 N 条（调试用，正式跑设 0）")
    args = ap.parse_args()

    para = hparams()
    clean_dir = args.clean_dir or para.test_clean_dir
    noisy_dir = args.noisy_dir or para.test_noisy_dir
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = None
    if args.mode == "model":
        assert args.ckpt, "--mode model 必须提供 --ckpt"
        model = Generator(use_seca=not args.no_seca, bottleneck=args.bottleneck,
                          width=args.width, se_init=para.se_init).to(device)
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print("[load] %s" % args.ckpt)

    files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
    if args.limit:
        files = files[:args.limit]
    assert files, "测试集为空：%s" % clean_dir
    print("[eval] %s | %d 条语音 | 全测试集，无 SNR 筛选" % (args.tag, len(files)))
    if args.mode == "model":
        print("[eval] z_mode=%s | z~N(0,I) when random | base_seed=%d" %
              (args.z_mode, args.seed))

    rows = []
    wav_dir = os.path.join(args.out_dir, "wav_" + args.tag)
    if args.save_wav:
        os.makedirs(wav_dir, exist_ok=True)

    for i, cf in enumerate(files):
        name = os.path.basename(cf)
        nf = os.path.join(noisy_dir, name)
        if not os.path.isfile(nf):
            print("  [skip] 缺少含噪对应文件 %s" % name)
            continue

        clean, _ = librosa.load(cf, sr=para.fs, mono=True)
        noisy, _ = librosa.load(nf, sr=para.fs, mono=True)
        n = min(len(clean), len(noisy))
        clean, noisy = clean[:n].astype(np.float64), noisy[:n].astype(np.float64)

        if args.mode == "model":
            enh = enhance_overlap_add(model, noisy, para, device,
                                      overlap=para.eval_overlap, seed=args.seed + i,
                                      z_mode=args.z_mode, z_avg=args.z_avg)
        else:
            enh = baseline_signal(args.mode, clean, noisy, para.fs)

        r = M.evaluate_all(clean, enh, para.fs)
        r["file"] = name
        rows.append(r)

        if args.save_wav and i < args.save_wav:
            sf.write(os.path.join(wav_dir, "enh-" + name), enh, para.fs)
            sf.write(os.path.join(wav_dir, "noisy-" + name), noisy, para.fs)
            sf.write(os.path.join(wav_dir, "clean-" + name), clean, para.fs)

        if (i + 1) % 50 == 0:
            print("  %d/%d  PESQ=%.3f STOI=%.4f SISDR=%.2f" %
                  (i + 1, len(files), r["PESQ"], r["STOI"], r["SISDR"]))

    keys = [k for k in rows[0].keys() if k != "file"]
    csv_path = os.path.join(args.out_dir, "per_utt_%s.csv" % args.tag)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file"] + keys)
        for r in rows:
            w.writerow([r["file"]] + ["%.6f" % r[k] if not np.isnan(r[k]) else "" for k in keys])

    print("\n===== %s | N=%d =====" % (args.tag, len(rows)))
    summary = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[~np.isnan(v)]
        mu, sd = v.mean(), v.std(ddof=1)
        se = sd / np.sqrt(len(v))
        summary[k] = (mu, sd, se)
        arrow = "↑" if M.METRIC_HIGHER_BETTER.get(k, True) else "↓"
        print("  %-7s %s  %8.4f ± %.4f   (95%%CI %.4f~%.4f)" %
              (k, arrow, mu, sd, mu - 1.96 * se, mu + 1.96 * se))

    with open(os.path.join(args.out_dir, "summary_%s.csv" % args.tag),
              "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std", "sem", "ci95_lo", "ci95_hi", "N"])
        for k, (mu, sd, se) in summary.items():
            w.writerow([k, "%.6f" % mu, "%.6f" % sd, "%.6f" % se,
                        "%.6f" % (mu - 1.96 * se), "%.6f" % (mu + 1.96 * se), len(rows)])
    print("\n[out] %s" % csv_path)


if __name__ == "__main__":
    main()

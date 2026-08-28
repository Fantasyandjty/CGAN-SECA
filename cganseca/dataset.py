import os
import numpy as np
import torch
from scipy.signal import lfilter
from torch.utils.data import Dataset


def emphasis(signal, emph_coeff=0.95, pre=True):
    signal = np.asarray(signal, dtype=np.float64)
    if pre:
        out = lfilter([1.0, -emph_coeff], [1.0], signal)
    else:
        out = lfilter([1.0], [1.0, -emph_coeff], signal)
    return out.astype(np.float32)


def _norm_path(p):
    return p.replace('\\', '/').strip()


def _load_scp(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "找不到训练列表: %s\n"
            "请先运行 data_geneation.py 生成 .npy 切片与该 scp 文件，"
            "并确认 hparams.train_scp 指向正确路径。" % os.path.abspath(path))

    clean, noisy, bad = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                bad.append(ln)
                continue
            clean.append(_norm_path(parts[0]))
            noisy.append(_norm_path(parts[1]))

    if bad:
        raise ValueError("scp 第 %s 行格式错误（每行需为「干净路径 含噪路径」）" %
                         ", ".join(map(str, bad[:10])))
    if not clean:
        raise ValueError("scp 为空: %s" % os.path.abspath(path))


    for p in (clean[:3] + noisy[:3]):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                "scp 中的文件不存在: %s\n"
                "常见原因：scp 里是 Windows 绝对路径（如 D:/...），"
                "在当前机器上不成立。请重新生成 scp 或批量替换路径前缀。" % p)

    return clean, noisy


class SEGAN_Dataset(Dataset):
    def __init__(self, para, split="train"):
        self.para = para
        self.emph = getattr(para, "emph_coeff", 0.95)

        clean, noisy = _load_scp(para.train_scp)


        rng = np.random.default_rng(20240101)
        idx = rng.permutation(len(clean))
        n_val = int(len(clean) * getattr(para, "valid_ratio", 0.0))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        use = tr_idx if split == "train" else val_idx

        self.clean_files = [clean[i] for i in use]
        self.noisy_files = [noisy[i] for i in use]
        self.split = split

        if split == "train" and len(self.clean_files) == 0:
            raise ValueError("训练集为空，请检查 valid_ratio 是否设得过大")

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        c = emphasis(np.load(self.clean_files[idx]), self.emph)
        n = emphasis(np.load(self.noisy_files[idx]), self.emph)
        c = torch.from_numpy(np.ascontiguousarray(c, dtype=np.float32)).reshape(1, -1)
        n = torch.from_numpy(np.ascontiguousarray(n, dtype=np.float32)).reshape(1, -1)
        return c, n

    def ref_batch(self, batch_size, seed=None):
        rng = np.random.default_rng(seed)
        index = rng.choice(len(self.clean_files), batch_size).tolist()
        cc = np.stack([emphasis(np.load(self.clean_files[i]), self.emph) for i in index])
        nn_ = np.stack([emphasis(np.load(self.noisy_files[i]), self.emph) for i in index])
        batch = np.stack([cc, nn_], axis=1).astype(np.float32)
        return torch.from_numpy(batch)


if __name__ == "__main__":
    from hparams import hparams
    para = hparams()
    tr = SEGAN_Dataset(para, "train")
    va = SEGAN_Dataset(para, "valid")
    print("train %d  valid %d" % (len(tr), len(va)))

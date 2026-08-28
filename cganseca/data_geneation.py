import os
import numpy as np
import librosa


def wav_split(wav, win_length, strid):
    slices = []
    if len(wav) > win_length:
        for idx_end in range(win_length, len(wav), strid):
            slices.append(wav[idx_end - win_length:idx_end])
        slices.append(wav[-win_length:])
    return slices


def save_slices(slices, name):
    name_list = []
    for i, slice_wav in enumerate(slices):
        name_slice = "%s_%d.npy" % (name, i)
        if not os.path.exists(name_slice):
            np.save(name_slice, slice_wav.astype(np.float32))
        name_list.append(name_slice.replace("\\", "/"))
    return name_list


if __name__ == "__main__":

    clean_wav_path = "voicedemand_16k/train/clean"
    noisy_wav_path = "voicedemand_16k/train/noisy"

    catch_train_clean = "cache/clean"
    catch_train_noisy = "cache/noisy"

    scp_path = "scp/train_segan.scp"


    for p in (clean_wav_path, noisy_wav_path):
        if not os.path.isdir(p):
            raise FileNotFoundError(
                "找不到目录: %s\n请确认 voicedemand_16k 与本脚本在同一目录下。"
                % os.path.abspath(p))

    os.makedirs(catch_train_clean, exist_ok=True)
    os.makedirs(catch_train_noisy, exist_ok=True)
    os.makedirs(os.path.dirname(scp_path) or ".", exist_ok=True)

    win_length = 16384
    strid = win_length // 2

    wav_files = sorted(f for f in os.listdir(clean_wav_path) if f.lower().endswith(".wav"))
    print("[data] 训练集共 %d 条 wav" % len(wav_files))
    if len(wav_files) == 0:
        raise RuntimeError("目录里没有 wav 文件: %s" % os.path.abspath(clean_wav_path))

    n_ok, n_skip, n_slice = 0, 0, 0
    with open(scp_path, "wt", encoding="utf-8") as f:
        for k, name in enumerate(wav_files, 1):
            file_clean = os.path.join(clean_wav_path, name)
            file_noisy = os.path.join(noisy_wav_path, name)

            if not os.path.exists(file_noisy):
                print("  [跳过] 缺少含噪对应文件: %s" % name)
                n_skip += 1
                continue

            clean_data, _ = librosa.load(file_clean, sr=16000, mono=True)
            noisy_data, _ = librosa.load(file_noisy, sr=16000, mono=True)

            if len(clean_data) != len(noisy_data):
                n = min(len(clean_data), len(noisy_data))
                clean_data, noisy_data = clean_data[:n], noisy_data[:n]

            clean_names = save_slices(wav_split(clean_data, win_length, strid),
                                      os.path.join(catch_train_clean, name))
            noisy_names = save_slices(wav_split(noisy_data, win_length, strid),
                                      os.path.join(catch_train_noisy, name))

            for c, n_ in zip(clean_names, noisy_names):
                f.write("%s %s\n" % (c, n_))
                n_slice += 1

            n_ok += 1
            if k % 200 == 0 or k == len(wav_files):
                print("  进度 %d/%d  已生成切片 %d" % (k, len(wav_files), n_slice))

    print("\n[完成] 处理 %d 条，跳过 %d 条，共 %d 个切片" % (n_ok, n_skip, n_slice))
    print("[完成] scp 写入 %s" % os.path.abspath(scp_path))

    with open(scp_path, encoding="utf-8") as f:
        first = f.readline().strip()
    print("[校验] scp 首行: %s" % first)
    p0 = first.split()[0]
    print("[校验] 首个切片存在: %s" % os.path.isfile(p0))

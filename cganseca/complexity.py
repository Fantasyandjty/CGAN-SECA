import argparse
import time
import json

import numpy as np
import torch

from model import Generator, Discriminator

try:
    from thop import profile as thop_profile
except ImportError:
    thop_profile = None


FS = 16000
WIN = 16384


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def model_size_mb(m):
    return sum(p.numel() * p.element_size() for p in m.parameters()) / 1024 ** 2


def macs_per_call(model, device):
    if thop_profile is None:
        return float("nan")
    x = torch.randn(1, 1, WIN, device=device)
    z = torch.randn(1, model.z_ch, 8, device=device)
    macs, _ = thop_profile(model, inputs=(x, z), verbose=False)
    return float(macs)


@torch.no_grad()
def measure_rtf(model, device, n_warm=3, n_run=20):
    model.eval().to(device)
    x = torch.randn(1, 1, WIN, device=device)
    z = torch.randn(1, model.z_ch, 8, device=device)
    for _ in range(n_warm):
        model(x, z)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_run):
        model(x, z)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_run
    audio_sec = WIN / FS
    return dt, dt / audio_sec


@torch.no_grad()
def peak_memory_mb(model, device):
    if device.type != "cuda":
        return float("nan")
    model.to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.randn(1, 1, WIN, device=device)
    z = torch.randn(1, model.z_ch, 8, device=device)
    model(x, z)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


def analyze(name, model, device_cpu, device_gpu):
    row = dict(model=name)
    row["params_M"] = count_params(model) / 1e6
    row["size_MB"] = model_size_mb(model)
    mc = macs_per_call(model, device_cpu)
    row["GMACs_per_1.024s"] = mc / 1e9
    row["GMACs_per_sec_audio"] = mc / 1e9 / (WIN / FS)

    model.to(device_cpu)
    lat_c, rtf_c = measure_rtf(model, device_cpu, n_run=5)
    row["cpu_latency_ms"] = lat_c * 1e3
    row["cpu_RTF"] = rtf_c

    if device_gpu is not None:
        model.to(device_gpu)
        lat_g, rtf_g = measure_rtf(model, device_gpu, n_run=20)
        row["gpu_latency_ms"] = lat_g * 1e3
        row["gpu_RTF"] = rtf_g
        row["gpu_peak_mem_MB"] = peak_memory_mb(model, device_gpu)
    else:
        row["gpu_latency_ms"] = float("nan")
        row["gpu_RTF"] = float("nan")
        row["gpu_peak_mem_MB"] = float("nan")
    return row


LITERATURE = [
    dict(model="SEGAN (Pascual 2017)", params_M=97.47, note="时域GAN，本文同族基线"),
    dict(model="MetricGAN+ (Fu 2021)", params_M=2.70, note="时频域，指标驱动训练"),
    dict(model="CleanUNet (Kong 2022)", params_M=46.07, note="时域，Transformer瓶颈"),
    dict(model="CMGAN (Cao 2022)", params_M=1.83, note="时频域复数谱，Conformer"),
    dict(model="DeepFilterNet2 (2022)", params_M=2.31, note="低复杂度实时"),
    dict(model="GTCRN (2024)", params_M=0.024, note="超轻量实时"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=float, nargs="*", default=[1.0])
    ap.add_argument("--out", type=str, default="results/complexity.json")
    args = ap.parse_args()

    dev_cpu = torch.device("cpu")
    dev_gpu = torch.device("cuda:0") if torch.cuda.is_available() else None
    print("[env] torch %s | CUDA %s" % (torch.__version__, torch.cuda.is_available()))
    if dev_gpu:
        print("[env] GPU: %s" % torch.cuda.get_device_name(0))

    rows = []
    for w in args.widths:
        for seca in (True, False):
            nm = "CGAN-SECA(G, width=%.2f)" % w if seca else "CGAN w/o SECA(G, width=%.2f)" % w
            rows.append(analyze(nm, Generator(use_seca=seca, width=w), dev_cpu, dev_gpu))


    D = Discriminator()
    d_row = dict(model="Discriminator (仅训练期)", params_M=count_params(D) / 1e6,
                 size_MB=model_size_mb(D))
    rows.append(d_row)

    hdr = ["model", "params_M", "size_MB", "GMACs_per_sec_audio",
           "cpu_RTF", "gpu_RTF", "gpu_peak_mem_MB"]
    print("\n%-34s %9s %9s %11s %9s %9s %11s" %
          ("模型", "参数(M)", "大小(MB)", "GMACs/s音频", "CPU RTF", "GPU RTF", "峰值显存(MB)"))
    print("-" * 100)
    for r in rows:
        print("%-34s %9.2f %9.1f %11s %9s %9s %11s" % (
            r["model"], r.get("params_M", float("nan")), r.get("size_MB", float("nan")),
            "%.2f" % r["GMACs_per_sec_audio"] if "GMACs_per_sec_audio" in r else "-",
            "%.3f" % r["cpu_RTF"] if "cpu_RTF" in r else "-",
            "%.4f" % r["gpu_RTF"] if "gpu_RTF" in r and r["gpu_RTF"] == r["gpu_RTF"] else "-",
            "%.1f" % r["gpu_peak_mem_MB"] if "gpu_peak_mem_MB" in r and
                     r["gpu_peak_mem_MB"] == r["gpu_peak_mem_MB"] else "-"))

    print("\n【文献对照参数量】(需在论文中标注来源，不可写成本机实测)")
    for r in LITERATURE:
        print("  %-26s %7.3f M   %s" % (r["model"], r["params_M"], r["note"]))

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dict(measured=rows, literature=LITERATURE), f,
                  ensure_ascii=False, indent=2)
    print("\n[out] %s" % args.out)


if __name__ == "__main__":
    main()

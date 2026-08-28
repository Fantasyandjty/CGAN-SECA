import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import get_window

try:
    from pesq import pesq as _pesq
except ImportError:
    _pesq = None
try:
    from pystoi import stoi as _stoi
except ImportError:
    _stoi = None

EPS = 1e-10


def _align(x, y):
    n = min(len(x), len(y))
    return np.asarray(x[:n], dtype=np.float64), np.asarray(y[:n], dtype=np.float64)


def stft(x, n_fft=512, hop=128, win_len=512):
    win = get_window("hann", win_len, fftbins=True)
    pad = n_fft - win_len
    if pad > 0:
        win = np.pad(win, (pad // 2, pad - pad // 2))
    n_frames = 1 + (len(x) - n_fft) // hop if len(x) >= n_fft else 0
    if n_frames <= 0:
        return np.zeros((n_fft // 2 + 1, 0), dtype=complex)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * win[None, :]
    return np.fft.rfft(frames, n=n_fft, axis=1).T


def istft(X, n_fft=512, hop=128, win_len=512):
    win = get_window("hann", win_len, fftbins=True)
    pad = n_fft - win_len
    if pad > 0:
        win = np.pad(win, (pad // 2, pad - pad // 2))
    frames = np.fft.irfft(X.T, n=n_fft, axis=1)
    n_frames = frames.shape[0]
    out = np.zeros((n_frames - 1) * hop + n_fft)
    wsum = np.zeros_like(out)
    for i in range(n_frames):
        out[i * hop:i * hop + n_fft] += frames[i] * win
        wsum[i * hop:i * hop + n_fft] += win ** 2
    return out / np.maximum(wsum, EPS)


def wrap_to_pi(p):
    return np.angle(np.exp(1j * p))


def pesq_wb(clean, enh, fs=16000):
    if _pesq is None:
        return np.nan
    c, e = _align(clean, enh)
    try:
        return float(_pesq(fs, c, e, 'wb'))
    except Exception:
        return np.nan


def stoi_score(clean, enh, fs=16000, extended=False):
    if _stoi is None:
        return np.nan
    c, e = _align(clean, enh)
    try:
        return float(_stoi(c, e, fs, extended=extended))
    except Exception:
        return np.nan


def si_sdr(clean, enh):
    c, e = _align(clean, enh)
    c = c - c.mean()
    e = e - e.mean()
    alpha = np.dot(e, c) / (np.dot(c, c) + EPS)
    target = alpha * c
    noise = e - target
    return float(10 * np.log10((np.dot(target, target) + EPS) / (np.dot(noise, noise) + EPS)))


def seg_snr(clean, enh, fs=16000, frame_ms=30, lo=-10.0, hi=35.0):
    c, e = _align(clean, enh)
    N = int(fs * frame_ms / 1000)
    hop = N // 2
    n = 1 + (len(c) - N) // hop if len(c) >= N else 0
    vals = []
    for i in range(n):
        cs = c[i * hop:i * hop + N]
        es = e[i * hop:i * hop + N]
        sig = np.sum(cs ** 2)
        nse = np.sum((cs - es) ** 2)
        if sig < EPS:
            continue
        v = 10 * np.log10((sig + EPS) / (nse + EPS))
        vals.append(np.clip(v, lo, hi))
    return float(np.mean(vals)) if vals else np.nan


def _lpc(frame, order):
    r = np.correlate(frame, frame, mode='full')[len(frame) - 1:len(frame) + order]
    if r[0] < EPS:
        return np.concatenate(([1.0], np.zeros(order)))
    try:
        a = np.linalg.solve(toeplitz(r[:order]), r[1:order + 1])
    except np.linalg.LinAlgError:
        return np.concatenate(([1.0], np.zeros(order)))
    return np.concatenate(([1.0], -a))


def llr(clean, enh, fs=16000, frame_ms=30, order=16):
    c, e = _align(clean, enh)
    N = int(fs * frame_ms / 1000)
    hop = N // 2
    win = np.hanning(N)
    n = 1 + (len(c) - N) // hop if len(c) >= N else 0
    vals = []
    for i in range(n):
        cs = c[i * hop:i * hop + N] * win
        es = e[i * hop:i * hop + N] * win
        if np.sum(cs ** 2) < EPS:
            continue
        ac, ae = _lpc(cs, order), _lpc(es, order)
        rc = np.correlate(cs, cs, mode='full')[N - 1:N + order]
        Rc = toeplitz(rc[:order + 1])
        num = ae @ Rc @ ae
        den = ac @ Rc @ ac
        if den < EPS or num < EPS:
            continue
        vals.append(np.clip(np.log(num / den), 0, 2))
    if not vals:
        return np.nan
    vals = np.sort(vals)
    return float(np.mean(vals[:max(1, int(round(len(vals) * 0.95)))]))


_CENT_FREQ = np.array([
    50.0000, 120.000, 190.000, 260.000, 330.000, 400.000, 470.000, 540.000,
    617.372, 703.378, 798.717, 904.128, 1020.38, 1148.30, 1288.72, 1442.54,
    1610.70, 1794.16, 1993.93, 2211.08, 2446.71, 2701.97, 2978.04, 3276.17,
    3597.63])
_BANDWIDTH = np.array([
    70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 77.3724,
    86.0056, 95.3398, 105.411, 116.256, 127.914, 140.423, 153.823, 168.154,
    183.457, 199.776, 217.153, 235.631, 255.255, 276.072, 298.126, 321.465,
    346.136])

_CRIT_CACHE = {}


def _crit_filter_bank(fs, n_fft):
    key = (fs, n_fft)
    if key in _CRIT_CACHE:
        return _CRIT_CACHE[key]
    n_half = n_fft // 2
    max_freq = fs / 2.0
    bw_min = _BANDWIDTH[0]
    min_factor = np.exp(-30.0 / (2.0 * 2.303))
    j = np.arange(n_half, dtype=np.float64)
    bank = np.zeros((len(_CENT_FREQ), n_half))
    for i in range(len(_CENT_FREQ)):
        f0 = np.floor((_CENT_FREQ[i] / max_freq) * n_half)
        bw = (_BANDWIDTH[i] / max_freq) * n_half
        norm_factor = np.log(bw_min) - np.log(_BANDWIDTH[i])
        v = np.exp(-11.0 * (((j - f0) / bw) ** 2) + norm_factor)
        bank[i] = np.where(v > min_factor, v, 0.0)
    _CRIT_CACHE[key] = bank
    return bank


def _loc_peak(energy, slope):
    n_crit = len(energy)
    out = np.zeros(n_crit - 1)
    for i in range(n_crit - 1):
        if slope[i] > 0:
            n = i
            while n < n_crit - 1 and slope[n] > 0:
                n += 1
            out[i] = energy[n]
        else:
            n = i
            while n >= 0 and slope[n] <= 0:
                n -= 1
            out[i] = energy[n + 1]
    return out


def wss(clean, enh, fs=16000, frame_ms=30):
    c, e = _align(clean, enh)
    N = int(fs * frame_ms / 1000)
    skip = N // 4
    n_fft = 512
    n_half = n_fft // 2
    Kmax, Klocmax = 20.0, 1.0
    win = np.hanning(N + 2)[1:-1]
    bank = _crit_filter_bank(fs, n_fft)

    n = 1 + (len(c) - N) // skip if len(c) >= N else 0
    vals = []
    for i in range(n):
        cs = c[i * skip:i * skip + N] * win
        es = e[i * skip:i * skip + N] * win
        if np.sum(cs ** 2) < EPS:
            continue

        Sc = np.abs(np.fft.fft(cs, n_fft)[:n_half]) ** 2
        Se = np.abs(np.fft.fft(es, n_fft)[:n_half]) ** 2

        bc = 10 * np.log10(np.maximum(bank @ Sc, 1e-10))
        be = 10 * np.log10(np.maximum(bank @ Se, 1e-10))

        sc = bc[1:] - bc[:-1]
        se = be[1:] - be[:-1]

        lc = _loc_peak(bc, sc)
        le = _loc_peak(be, se)

        Wc = (Kmax / (Kmax + bc.max() - bc[:-1])) * (Klocmax / (Klocmax + lc - bc[:-1]))
        We = (Kmax / (Kmax + be.max() - be[:-1])) * (Klocmax / (Klocmax + le - be[:-1]))
        W = (Wc + We) / 2.0

        denom = np.sum(W)
        if denom < EPS:
            continue
        vals.append(float(np.sum(W * (sc - se) ** 2) / denom))

    if not vals:
        return np.nan
    vals = np.sort(vals)
    return float(np.mean(vals[:max(1, int(round(len(vals) * 0.95)))]))


def composite(clean, enh, fs=16000):
    p = pesq_wb(clean, enh, fs)
    l = llr(clean, enh, fs)
    w = wss(clean, enh, fs)
    s = seg_snr(clean, enh, fs)
    if np.isnan(p) or np.isnan(l) or np.isnan(w):
        return dict(CSIG=np.nan, CBAK=np.nan, COVL=np.nan)
    csig = np.clip(3.093 - 1.029 * l + 0.603 * p - 0.009 * w, 1, 5)
    cbak = np.clip(1.634 + 0.478 * p - 0.007 * w + 0.063 * (0 if np.isnan(s) else s), 1, 5)
    covl = np.clip(1.594 + 0.805 * p - 0.512 * l - 0.007 * w, 1, 5)
    return dict(CSIG=float(csig), CBAK=float(cbak), COVL=float(covl))


def phase_distance(clean, enh, fs=16000, n_fft=512, hop=128):
    c, e = _align(clean, enh)
    C = stft(c, n_fft, hop, n_fft)
    E = stft(e, n_fft, hop, n_fft)
    n = min(C.shape[1], E.shape[1])
    if n == 0:
        return np.nan
    C, E = C[:, :n], E[:, :n]
    Mc = np.abs(C)
    dphi = np.abs(wrap_to_pi(np.angle(E) - np.angle(C)))
    w = Mc / (Mc.sum() + EPS)
    return float(np.sum(w * dphi))


def group_delay_deviation(clean, enh, fs=16000, n_fft=512, hop=128):
    c, e = _align(clean, enh)
    C, E = stft(c, n_fft, hop, n_fft), stft(e, n_fft, hop, n_fft)
    n = min(C.shape[1], E.shape[1])
    if n == 0:
        return np.nan
    C, E = C[:, :n], E[:, :n]
    gc = wrap_to_pi(np.diff(np.angle(C), axis=0))
    ge = wrap_to_pi(np.diff(np.angle(E), axis=0))
    w = np.abs(C[1:, :])
    w = w / (w.sum() + EPS)
    return float(np.sum(w * np.abs(wrap_to_pi(ge - gc))))


def instantaneous_freq_deviation(clean, enh, fs=16000, n_fft=512, hop=128):
    c, e = _align(clean, enh)
    C, E = stft(c, n_fft, hop, n_fft), stft(e, n_fft, hop, n_fft)
    n = min(C.shape[1], E.shape[1])
    if n < 2:
        return np.nan
    C, E = C[:, :n], E[:, :n]
    ic = wrap_to_pi(np.diff(np.angle(C), axis=1))
    ie = wrap_to_pi(np.diff(np.angle(E), axis=1))
    w = np.abs(C[:, 1:])
    w = w / (w.sum() + EPS)
    return float(np.sum(w * np.abs(wrap_to_pi(ie - ic))))


def complex_spec_err(clean, enh, fs=16000, n_fft=512, hop=128):
    c, e = _align(clean, enh)
    C, E = stft(c, n_fft, hop, n_fft), stft(e, n_fft, hop, n_fft)
    n = min(C.shape[1], E.shape[1])
    if n == 0:
        return np.nan
    C, E = C[:, :n], E[:, :n]
    num = np.sum(np.abs(C - E) ** 2)
    den = np.sum(np.abs(C) ** 2)
    return float(10 * np.log10((num + EPS) / (den + EPS)))


def mag_lsd(clean, enh, fs=16000, n_fft=512, hop=128):
    c, e = _align(clean, enh)
    C, E = stft(c, n_fft, hop, n_fft), stft(e, n_fft, hop, n_fft)
    n = min(C.shape[1], E.shape[1])
    if n == 0:
        return np.nan
    lc = 10 * np.log10(np.abs(C[:, :n]) ** 2 + 1e-8)
    le = 10 * np.log10(np.abs(E[:, :n]) ** 2 + 1e-8)
    return float(np.mean(np.sqrt(np.mean((lc - le) ** 2, axis=0))))


def stft_consistency(x, n_fft=512, hop=128):
    X = stft(np.asarray(x, dtype=np.float64), n_fft, hop, n_fft)
    if X.shape[1] == 0:
        return np.nan
    xr = istft(X, n_fft, hop, n_fft)
    Xr = stft(xr, n_fft, hop, n_fft)
    m = min(X.shape[1], Xr.shape[1])
    num = np.sum(np.abs(X[:, :m] - Xr[:, :m]) ** 2)
    den = np.sum(np.abs(X[:, :m]) ** 2)
    return float(10 * np.log10((num + EPS) / (den + EPS)))


def evaluate_all(clean, enh, fs=16000, with_composite=True, with_phase=True):
    r = dict(
        PESQ=pesq_wb(clean, enh, fs),
        STOI=stoi_score(clean, enh, fs, extended=False),
        ESTOI=stoi_score(clean, enh, fs, extended=True),
        SISDR=si_sdr(clean, enh),
        SegSNR=seg_snr(clean, enh, fs),
    )
    if with_composite:
        r.update(composite(clean, enh, fs))
    if with_phase:
        r.update(
            PhaseD=phase_distance(clean, enh, fs),
            GDD=group_delay_deviation(clean, enh, fs),
            IFD=instantaneous_freq_deviation(clean, enh, fs),
            cSER=complex_spec_err(clean, enh, fs),
            LSD=mag_lsd(clean, enh, fs),
            Cons=stft_consistency(enh),
        )
    return r


METRIC_HIGHER_BETTER = {
    "PESQ": True, "STOI": True, "ESTOI": True, "SISDR": True, "SegSNR": True,
    "CSIG": True, "CBAK": True, "COVL": True,
    "PhaseD": False, "GDD": False, "IFD": False, "cSER": False, "LSD": False, "Cons": False,
}


if __name__ == "__main__":
    fs = 16000
    rng = np.random.default_rng(0)
    from scipy.signal import butter, lfilter
    b, a = butter(4, [200 / (fs / 2), 3800 / (fs / 2)], btype="band")
    exc = rng.standard_normal(fs) * (1 + np.sin(2 * np.pi * 4 * np.arange(fs) / fs))
    clean = lfilter(b, a, exc)
    clean = 0.5 * clean / (np.abs(clean).max() + 1e-9)
    noise = 0.08 * rng.standard_normal(fs)
    noisy = clean + noise


    C, Nz = stft(clean), stft(noisy)
    m = min(C.shape[1], Nz.shape[1])
    oracle_mag_noisy_phase = istft(np.abs(C[:, :m]) * np.exp(1j * np.angle(Nz[:, :m])))

    for name, sig in [("noisy", noisy), ("oracleMag+noisyPhase", oracle_mag_noisy_phase)]:
        r = evaluate_all(clean, sig, fs)
        print(name, {k: (round(v, 4) if isinstance(v, float) and not np.isnan(v) else v)
                     for k, v in r.items()})

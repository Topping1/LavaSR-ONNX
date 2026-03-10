import math
import yaml
import numpy as np
import soundfile as sf
import onnxruntime as ort
from scipy import signal
from numpy.lib.stride_tricks import sliding_window_view


###########################################################
# AUDIO
###########################################################

def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.astype(np.float32, copy=False)
    return x.mean(axis=1).astype(np.float32)


def _resample_poly_1d(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    g = math.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    y = signal.resample_poly(x, up, down)
    return y.astype(np.float32)


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    wav, sr = sf.read(path, always_2d=False)
    wav = _to_mono(np.asarray(wav))
    wav = _resample_poly_1d(wav, int(sr), int(target_sr))
    return wav[None, :].astype(np.float32)


###########################################################
# LINKWITZ MERGE
###########################################################

class FastLRMerge:
    def __init__(self, sample_rate=48000, cutoff=4000, transition_bins=256):
        self.sample_rate = sample_rate
        self.cutoff = cutoff
        self.transition_bins = transition_bins

    def _fade(self, n: int) -> np.ndarray:
        x = np.linspace(-1.0, 1.0, num=n, dtype=np.float32)
        t = (x + 1.0) / 2.0
        return (3 * t**2 - 2 * t**3).astype(np.float32)

    def __call__(self, enhanced: np.ndarray, original: np.ndarray) -> np.ndarray:
        out = []
        for a, b in zip(enhanced, original):
            spec1 = np.fft.rfft(a)
            spec2 = np.fft.rfft(b)
            n_bins = spec1.shape[-1]
            cutoff_bin = int((self.cutoff / (self.sample_rate / 2.0)) * n_bins)
            mask = np.ones(n_bins, dtype=np.float32)
            half = self.transition_bins // 2
            start = max(0, cutoff_bin - half)
            end = min(n_bins, cutoff_bin + half)
            mask[:start] = 0.0
            if end > start:
                mask[start:end] = self._fade(end - start)
            mask[end:] = 1.0
            spec = spec2 + (spec1 - spec2) * mask.astype(np.complex64)
            out.append(np.fft.irfft(spec, n=a.shape[-1]).astype(np.float32))
        return np.stack(out, axis=0)


###########################################################
# DSP HELPERS
###########################################################

def _hann_periodic(win_len: int) -> np.ndarray:
    return np.hanning(win_len + 1)[:-1].astype(np.float32)


def _frame_signal_same_reflect(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    pad = (frame_length - hop_length) // 2
    xpad = np.pad(x, (pad, pad), mode='reflect')
    if xpad.shape[0] < frame_length:
        xpad = np.pad(xpad, (0, frame_length - xpad.shape[0]))
    frames = sliding_window_view(xpad, frame_length)[::hop_length]
    return frames.astype(np.float32, copy=False)


def _stft_same_reflect(x: np.ndarray, n_fft: int, hop_length: int, win_length: int) -> np.ndarray:
    """Match MelSpectrogramFeatures(padding="same") in vocos.py."""
    pad = (win_length - hop_length) // 2
    xpad = np.pad(x, (pad, pad), mode="reflect")
    if xpad.shape[0] < win_length:
        xpad = np.pad(xpad, (0, win_length - xpad.shape[0]))
    frames = sliding_window_view(xpad, win_length)[::hop_length]
    window = _hann_periodic(win_length)
    frames = frames * window[None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    return spec.T.astype(np.complex64)  # [F, T]


def _istft_same(spec: np.ndarray, n_fft: int, hop_length: int, win_length: int, target_len: int | None = None) -> np.ndarray:
    """Match vocos.ISTFT(padding="same") in vocos.py."""
    window = _hann_periodic(win_length)
    pad = (win_length - hop_length) // 2

    ifft = np.fft.irfft(spec, n=n_fft, axis=0).astype(np.float32)[:win_length, :]
    ifft *= window[:, None]

    T = ifft.shape[1]
    output_size = (T - 1) * hop_length + win_length
    y = np.zeros(output_size, dtype=np.float32)
    wenv = np.zeros(output_size, dtype=np.float32)
    win_sq = window ** 2
    for t in range(T):
        start = t * hop_length
        y[start:start + win_length] += ifft[:, t]
        wenv[start:start + win_length] += win_sq

    y = y[pad:-pad]
    wenv = wenv[pad:-pad]
    y = y / np.clip(wenv, 1e-8, None)

    if target_len is not None:
        if y.shape[0] < target_len:
            y = np.pad(y, (0, target_len - y.shape[0]))
        else:
            y = y[:target_len]

    return y.astype(np.float32)


def _stft_center_reflect(x: np.ndarray, n_fft: int, hop_length: int, win_length: int) -> np.ndarray:
    """Match torch.stft(..., center=True, pad_mode="reflect") used by ULUNAS."""
    pad = n_fft // 2
    xpad = np.pad(x, (pad, pad), mode="reflect")
    if xpad.shape[0] < win_length:
        xpad = np.pad(xpad, (0, win_length - xpad.shape[0]))
    frames = sliding_window_view(xpad, win_length)[::hop_length]
    window = _hann_periodic(win_length)
    frames = frames * window[None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    return spec.T.astype(np.complex64)  # [F, T]


def stft_ri_batch_center_reflect(wav_batch: np.ndarray, n_fft: int, hop_len: int, win_len: int):
    specs = []
    for wav in wav_batch:
        z = _stft_center_reflect(wav, n_fft, hop_len, win_len)
        specs.append(z)
    spec = np.stack(specs, axis=0)  # [B, F, T]
    spec_ri = np.stack([spec.real, spec.imag], axis=1)  # [B, 2, F, T]
    spec_ri = np.transpose(spec_ri, (0, 1, 3, 2))       # [B, 2, T, F]
    return spec_ri.astype(np.float32)


def istft_ri_batch_center_reflect(spec_ri: np.ndarray, n_fft: int, hop_len: int, win_len: int, target_len: int):
    outs = []
    spec_ri = np.transpose(spec_ri, (0, 1, 3, 2))  # [B, 2, F, T]
    pad = n_fft // 2
    for s in spec_ri:
        spec = s[0] + 1j * s[1]  # [F, T]
        y = _istft_same(spec, n_fft, hop_len, win_len, target_len=None)
        # _istft_same crops (win_len-hop)/2 each side; center=True needs n_fft//2 crop.
        # So rebuild with center crop directly for ULUNAS.
        window = _hann_periodic(win_len)
        ifft = np.fft.irfft(spec, n=n_fft, axis=0).astype(np.float32)[:win_len, :]
        ifft *= window[:, None]
        T = ifft.shape[1]
        output_size = (T - 1) * hop_len + win_len
        raw = np.zeros(output_size, dtype=np.float32)
        wenv = np.zeros(output_size, dtype=np.float32)
        win_sq = window ** 2
        for t in range(T):
            start = t * hop_len
            raw[start:start + win_len] += ifft[:, t]
            wenv[start:start + win_len] += win_sq
        raw = raw[pad:-pad]
        wenv = wenv[pad:-pad]
        raw = raw / np.clip(wenv, 1e-8, None)
        if raw.shape[0] < target_len:
            raw = np.pad(raw, (0, target_len - raw.shape[0]))
        else:
            raw = raw[:target_len]
        outs.append(raw.astype(np.float32))
    return np.stack(outs, axis=0).astype(np.float32)
# Slaney mel helpers matching torchaudio/librosa style more closely than HTK
_F_SP = 200.0 / 3
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = np.log(6.4) / 27.0


def _hz_to_mel_slaney(f):
    f = np.asanyarray(f, dtype=np.float32)
    m = f / _F_SP
    log_t = f >= _MIN_LOG_HZ
    if np.any(log_t):
        m = m.copy()
        m[log_t] = _MIN_LOG_MEL + np.log(f[log_t] / _MIN_LOG_HZ) / _LOGSTEP
    return m


def _mel_to_hz_slaney(m):
    m = np.asanyarray(m, dtype=np.float32)
    f = _F_SP * m
    log_t = m >= _MIN_LOG_MEL
    if np.any(log_t):
        f = f.copy()
        f[log_t] = _MIN_LOG_HZ * np.exp(_LOGSTEP * (m[log_t] - _MIN_LOG_MEL))
    return f


def build_mel_filterbank_slaney(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float):
    n_freqs = n_fft // 2 + 1
    fftfreqs = np.linspace(0.0, sr / 2.0, n_freqs, dtype=np.float32)
    m_min = _hz_to_mel_slaney(np.array([fmin], dtype=np.float32))[0]
    m_max = _hz_to_mel_slaney(np.array([fmax], dtype=np.float32))[0]
    m_pts = np.linspace(m_min, m_max, n_mels + 2, dtype=np.float32)
    f_pts = _mel_to_hz_slaney(m_pts)
    fdiff = np.diff(f_pts)
    ramps = f_pts[:, None] - fftfreqs[None, :]

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        lower = -ramps[i] / max(fdiff[i], 1e-12)
        upper = ramps[i + 2] / max(fdiff[i + 1], 1e-12)
        fb[i] = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney area normalization
    enorm = 2.0 / np.maximum(f_pts[2:n_mels + 2] - f_pts[:n_mels], 1e-12)
    fb *= enorm[:, None]
    return fb.astype(np.float32)


class MelSpectrogramFrontend:
    def __init__(self, sample_rate: int, n_fft: int, hop_length: int, n_mels: int, padding='same', fmin=0.0, fmax=None):
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(n_fft)
        self.n_mels = int(n_mels)
        self.padding = padding
        self.fmin = float(fmin)
        self.fmax = float(fmax if fmax is not None else sample_rate / 2.0)
        self.mel_fb = build_mel_filterbank_slaney(self.sample_rate, self.n_fft, self.n_mels, self.fmin, self.fmax)

    def __call__(self, wav_batch: np.ndarray) -> np.ndarray:
        feats = []
        for wav in wav_batch:
            z = _stft_same_reflect(wav, self.n_fft, self.hop_length, self.win_length)  # [F,T]
            mag = np.abs(z).astype(np.float32)
            mel = self.mel_fb @ mag
            mel = np.log(np.clip(mel, 1e-5, None)).astype(np.float32)
            feats.append(mel)
        return np.stack(feats, axis=0).astype(np.float32)  # [B, 80, T]


class ISTFTReconstructor:
    def __init__(self, n_fft: int, hop_length: int, win_length: int = None, padding='same'):
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length if win_length is not None else n_fft)
        self.padding = padding

    def __call__(self, spec: np.ndarray, target_len: int = None) -> np.ndarray:
        # spec: [B, F, T] complex64
        outs = []
        window = _hann_periodic(self.win_length)
        pad = (self.win_length - self.hop_length) // 2
        for z in spec:
            ifft = np.fft.irfft(z, n=self.n_fft, axis=0).astype(np.float32)[:self.win_length, :]
            ifft *= window[:, None]
            T = ifft.shape[1]
            output_size = (T - 1) * self.hop_length + self.win_length
            y = np.zeros(output_size, dtype=np.float32)
            wenv = np.zeros(output_size, dtype=np.float32)
            win_sq = window ** 2
            for t in range(T):
                start = t * self.hop_length
                y[start:start + self.win_length] += ifft[:, t]
                wenv[start:start + self.win_length] += win_sq
            y = y[pad:-pad]
            wenv = wenv[pad:-pad]
            y = y / np.clip(wenv, 1e-8, None)
            if target_len is not None:
                if y.shape[0] < target_len:
                    y = np.pad(y, (0, target_len - y.shape[0]))
                else:
                    y = y[:target_len]
            outs.append(y.astype(np.float32))
        return np.stack(outs, axis=0).astype(np.float32)


###########################################################
# HYBRID ONNX ULUNAS DENOISER
###########################################################

class LavaDenoiser:
    def __init__(
        self,
        denoiser_onnx_path="denoiser_core_legacy_fixed63.onnx",
        ort_providers=None,
        ort_intra_op_num_threads=1,
        ort_inter_op_num_threads=1,
        n_fft=512,
        hop_len=256,
        win_len=512,
        chunk_frames=63,
        chunk_hop_frames=21,
    ):
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.chunk_frames = chunk_frames
        self.chunk_hop_frames = chunk_hop_frames

        if ort_providers is None:
            ort_providers = ["CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.intra_op_num_threads = ort_intra_op_num_threads
        so.inter_op_num_threads = ort_inter_op_num_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        self.session = ort.InferenceSession(
            denoiser_onnx_path,
            sess_options=so,
            providers=ort_providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        w = signal.windows.hann(chunk_frames, sym=True).astype(np.float32)
        w = np.clip(w ** 2, 1e-4, None)
        self.chunk_weight = w.reshape(1, 1, chunk_frames, 1)

    def _run_chunk_ort(self, chunk_ri: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: chunk_ri.astype(np.float32)})[0]

    def _run_overlap_chunks(self, spec_ri: np.ndarray) -> np.ndarray:
        B, C, T, F = spec_ri.shape
        L = self.chunk_frames
        H = self.chunk_hop_frames

        if T <= L:
            pad_right = L - T
            chunk = np.pad(spec_ri, ((0, 0), (0, 0), (0, pad_right), (0, 0)))
            out = self._run_chunk_ort(chunk)
            return out[:, :, :T, :]

        starts = list(range(0, max(1, T - L + 1), H))
        if starts[-1] != T - L:
            starts.append(T - L)

        acc = np.zeros((B, C, T, F), dtype=np.float32)
        wacc = np.zeros((1, 1, T, 1), dtype=np.float32)

        for start in starts:
            end = start + L
            chunk = spec_ri[:, :, start:end, :]
            out_chunk = self._run_chunk_ort(chunk)
            acc[:, :, start:end, :] += out_chunk * self.chunk_weight
            wacc[:, :, start:end, :] += self.chunk_weight

        return acc / np.clip(wacc, 1e-6, None)

    def infer(self, wav_batch: np.ndarray) -> np.ndarray:
        spec_ri = stft_ri_batch_center_reflect(wav_batch, self.n_fft, self.hop_len, self.win_len)
        spec_enh_ri = self._run_overlap_chunks(spec_ri)
        out = istft_ri_batch_center_reflect(spec_enh_ri, self.n_fft, self.hop_len, self.win_len, target_len=wav_batch.shape[1])
        return out.astype(np.float32)


###########################################################
# HYBRID ONNX VOCOS ENHANCER
###########################################################

class LavaEnhancer:
    def __init__(
        self,
        config_path,
        enhancer_backbone_onnx="enhancer_backbone.onnx",
        enhancer_spec_head_onnx="enhancer_spec_head.onnx",
        ort_providers=None,
        ort_intra_op_num_threads=1,
        ort_inter_op_num_threads=1,
    ):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        feat_cfg = config['feature_extractor']['init_args']
        head_cfg = config['head']['init_args']

        self.feature_extractor = MelSpectrogramFrontend(
            sample_rate=feat_cfg['sample_rate'],
            n_fft=feat_cfg['n_fft'],
            hop_length=feat_cfg['hop_length'],
            n_mels=feat_cfg['n_mels'],
            padding=feat_cfg.get('padding', 'same'),
            fmin=feat_cfg.get('f_min', 0.0),
            # Match vocos.MelSpectrogramFeatures defaults exactly.
            # In the uploaded vocos.py, omitted f_max defaults to 8000 Hz,
            # not Nyquist.
            fmax=feat_cfg.get('f_max', 8000.0),
        )
        self.istft = ISTFTReconstructor(
            n_fft=head_cfg['n_fft'],
            hop_length=head_cfg['hop_length'],
            win_length=head_cfg['n_fft'],
            padding=head_cfg.get('padding', 'same'),
        )
        self.merge = FastLRMerge(sample_rate=48000)

        if ort_providers is None:
            ort_providers = ["CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.intra_op_num_threads = ort_intra_op_num_threads
        so.inter_op_num_threads = ort_inter_op_num_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.backbone_session = ort.InferenceSession(
            enhancer_backbone_onnx,
            sess_options=so,
            providers=ort_providers,
        )
        self.spec_head_session = ort.InferenceSession(
            enhancer_spec_head_onnx,
            sess_options=so,
            providers=ort_providers,
        )

        self.backbone_input_name = self.backbone_session.get_inputs()[0].name
        self.backbone_output_name = self.backbone_session.get_outputs()[0].name
        self.spec_head_input_name = self.spec_head_session.get_inputs()[0].name
        spec_outputs = self.spec_head_session.get_outputs()
        self.spec_head_output_names = [spec_outputs[0].name, spec_outputs[1].name]

    def infer(self, wav_batch: np.ndarray) -> np.ndarray:
        features = self.feature_extractor(wav_batch)
        hidden = self.backbone_session.run([self.backbone_output_name], {self.backbone_input_name: features.astype(np.float32)})[0]
        real, imag = self.spec_head_session.run(self.spec_head_output_names, {self.spec_head_input_name: hidden.astype(np.float32)})
        spec = real.astype(np.float32) + 1j * imag.astype(np.float32)
        enhanced = self.istft(spec, target_len=wav_batch.shape[1])
        enhanced = self.merge(enhanced[:, :wav_batch.shape[1]], wav_batch[:, :enhanced.shape[1]])
        return enhanced.astype(np.float32)


###########################################################
# MAIN ENGINE
###########################################################

class LavaSR:
    def __init__(
        self,
        config,
        denoiser_onnx="denoiser_core_legacy_fixed63.onnx",
        enhancer_backbone_onnx="enhancer_backbone.onnx",
        enhancer_spec_head_onnx="enhancer_spec_head.onnx",
        ort_providers=None,
        ort_intra_op_num_threads=1,
        ort_inter_op_num_threads=1,
    ):
        self.denoiser = LavaDenoiser(
            denoiser_onnx_path=denoiser_onnx,
            ort_providers=ort_providers,
            ort_intra_op_num_threads=ort_intra_op_num_threads,
            ort_inter_op_num_threads=ort_inter_op_num_threads,
        )
        self.enhancer = LavaEnhancer(
            config_path=config,
            enhancer_backbone_onnx=enhancer_backbone_onnx,
            enhancer_spec_head_onnx=enhancer_spec_head_onnx,
            ort_providers=ort_providers,
            ort_intra_op_num_threads=ort_intra_op_num_threads,
            ort_inter_op_num_threads=ort_inter_op_num_threads,
        )

    def load_audio(self, path):
        return load_audio(path)

    def enhance(self, wav_batch: np.ndarray, apply_denoise=False) -> np.ndarray:
        wav = np.asarray(wav_batch, dtype=np.float32)
        if wav.ndim == 1:
            wav = wav[None, :]

        if apply_denoise:
            wav = self.denoiser.infer(wav)

        resampled = []
        for w in wav:
            resampled.append(_resample_poly_1d(w, 16000, 48000))
        wav_48k = np.stack(resampled, axis=0).astype(np.float32)

        enhanced = self.enhancer.infer(wav_48k)
        return enhanced.squeeze().astype(np.float32)

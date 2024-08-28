import os
import random
import torch
import torchaudio
import torchaudio.functional as F

from typing import Union, Callable, Optional

from ariacl.config import load_config


def get_wav_segments(
    audio_path: str,
    stride_factor: int,
):
    assert os.path.isfile(audio_path), "Audio file not found"
    config = load_config()
    sample_rate = config["audio"]["sample_rate"]
    chunk_len = config["audio"]["chunk_len"]

    stream = torchaudio.io.StreamReader(audio_path)
    chunk_samples = int(sample_rate * chunk_len)
    stride_samples = int(chunk_samples // stride_factor)
    assert chunk_samples % stride_samples == 0, "Invalid stride"

    stream.add_basic_audio_stream(
        frames_per_chunk=stride_samples,
        stream_index=0,
        sample_rate=sample_rate,
    )

    buffer = torch.tensor([], dtype=torch.float32)
    for stride_seg in stream.stream():
        seg_chunk = stride_seg[0].mean(1)

        # Pad seg_chunk if required
        if seg_chunk.shape[0] < stride_samples:
            seg_chunk = torch.nn.functional.pad(
                seg_chunk,
                (0, stride_samples - seg_chunk.shape[0]),
                mode="constant",
                value=0.0,
            )

        if buffer.shape[0] < chunk_samples:
            buffer = torch.cat((buffer, seg_chunk), dim=0)
        else:
            buffer = torch.cat((buffer[stride_samples:], seg_chunk), dim=0)

        if buffer.shape[0] == chunk_samples:
            yield buffer


@torch.no_grad()
def get_wav_segments_b(
    audio_path: str,
    stride_factor: int,
    device: str = "cuda",
):
    assert os.path.isfile(audio_path), "Audio file not found"
    config = load_config()
    sample_rate = config["audio"]["sample_rate"]
    chunk_len = config["audio"]["chunk_len"]
    chunk_samples = int(sample_rate * chunk_len)
    stride_samples = int(chunk_samples // stride_factor)
    assert chunk_samples % stride_samples == 0, "Invalid stride"

    wav, orig_sample_rate = torchaudio.load(audio_path)
    if orig_sample_rate != sample_rate:
        wav = torchaudio.functional.resample(
            waveform=wav, orig_freq=orig_sample_rate, new_freq=sample_rate
        ).to(device)
    else:
        wav = wav.to(device)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav.squeeze(0)

    return wav.unfold(0, chunk_samples, stride_samples)


def _amplitude_to_db(
    S: torch.Tensor,
    ref: Union[float, Callable] = 1.0,
    amin: float = 1e-5,
    top_db: Optional[float] = 80.0,
):
    magnitude = torch.abs(S)
    amin = torch.tensor(amin, device=magnitude.device, dtype=magnitude.dtype)

    if callable(ref):
        ref_value = ref(magnitude)
    else:
        ref_value = torch.tensor(ref)

    log_spec = 20.0 * torch.log10(torch.maximum(amin, magnitude))
    log_spec -= 20.0 * torch.log10(torch.maximum(amin, ref_value))

    if top_db is not None:
        log_spec = torch.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def signal_to_frame_nonsilent(
    y: torch.Tensor,
    frame_length: int = 2048,
    hop_length: int = 512,
    top_db: float = 60.0,
    silence_threshold: float = -20.0,
    ref: Union[float, Callable] = torch.max,
    aggregate: Callable = torch.max,
):
    """Note that silence_threshold in in units of dBFS."""
    # Ensure there is a batch_dim
    if y.dim() == 1:
        y = y.unsqueeze(0)

    spec = torchaudio.transforms.Spectrogram(
        n_fft=frame_length,
        hop_length=hop_length,
        power=2,
        center=True,
    ).to(y.device)(y)
    rms = torch.sqrt(spec.mean(dim=1))
    db = _amplitude_to_db(rms, ref=ref, top_db=None)

    if db.dim() > 2:
        db = aggregate(db, dim=1)
    if top_db is not None:
        db = torch.max(db, db.max() - top_db)

    return db > silence_threshold


def get_audio_intervals(
    wav: torch.Tensor,
    min_window_s: float,
    threshold_db: int,
    detect_silent_intervals: bool = False,
):
    """Note that threshold_db is in units of dBFS"""

    config = load_config()
    sample_rate = config["audio"]["sample_rate"]
    frame_len = config["data"]["piano_detection"]["frame_len"]
    hop_len = config["data"]["piano_detection"]["hop_len"]
    min_window_steps = (sample_rate // hop_len) * min_window_s + 1
    ms_per_hop = int((hop_len * 1e3) / sample_rate)

    non_silent = signal_to_frame_nonsilent(
        wav,
        frame_length=frame_len,
        hop_length=hop_len,
        silence_threshold=threshold_db,
        ref=1.0,
    )

    if detect_silent_intervals is True:
        non_silent = ~non_silent

    batch_intervals = []
    for non_silent_item in non_silent:
        padded = torch.cat(
            [
                torch.tensor([False], device=wav.device),
                non_silent_item,
                torch.tensor([False], device=wav.device),
            ]
        )
        edges = torch.diff(padded.int())
        starts = torch.where(edges == 1)[0]
        ends = torch.where(edges == -1)[0]
        lengths = ends - starts
        valid = lengths > min_window_steps

        intervals = [
            (int(start.item() * ms_per_hop), int((end.item() - 1) * ms_per_hop))
            for start, end, vl in zip(starts, ends, valid)
            if vl.item()
        ]
        batch_intervals.append(intervals)

    return batch_intervals


class AudioTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = load_config()

        self.sample_rate = config["audio"]["sample_rate"]
        self.n_fft = config["audio"]["n_fft"]
        self.hop_len = config["audio"]["hop_len"]
        self.chunk_len = config["audio"]["chunk_len"]
        self.n_mels = config["audio"]["n_mels"]
        self.samples_per_chunk = self.sample_rate * self.chunk_len

        self.noise_ratio = config["aug"]["noise_ratio"]
        self.min_noise_snr = config["aug"]["min_noise_snr"]
        self.max_noise_snr = config["aug"]["max_noise_snr"]
        self.reverb_ratio = config["aug"]["reverb_ratio"]
        self.bandpass_ratio = config["aug"]["bandpass_ratio"]
        self.distortion_ratio = config["aug"]["distortion_ratio"]
        self.min_distortion_gain = config["aug"]["min_distortion_gain"]
        self.max_distortion_gain = config["aug"]["max_distortion_gain"]
        self.pitch_shift_ratio = config["aug"]["pitch_shift_ratio"]
        self.max_pitch_shift = config["aug"]["max_pitch_shift"]

        # Audio aug
        impulse_paths = self._get_paths(
            os.path.join(os.path.dirname(__file__), "assets", "impulse")
        )
        noise_paths = self._get_paths(
            os.path.join(os.path.dirname(__file__), "assets", "noise")
        )

        # Register impulses and noises as buffers
        self.num_impulse = 0
        for i, impulse in enumerate(self._get_impulses(impulse_paths)):
            self.register_buffer(f"impulse_{i}", impulse)
            self.num_impulse += 1

        self.num_noise = 0
        for i, noise in enumerate(self._get_noise(noise_paths)):
            self.register_buffer(f"noise_{i}", noise)
            self.num_noise += 1

        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_len,
        )
        self.mel_transform = torchaudio.transforms.MelScale(
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            n_stft=self.n_fft // 2 + 1,
            f_min=20,
            f_max=self.sample_rate // 2,
        )

    def get_params(self):
        return {
            "noise_ratio": self.noise_ratio,
            "min_noise_snr": self.min_noise_snr,
            "max_noise_snr": self.max_noise_snr,
            "reverb_ratio": self.reverb_ratio,
            "bandpass_ratio": self.bandpass_ratio,
            "distortion_ratio": self.distortion_ratio,
            "min_distortion_gain": self.min_distortion_gain,
            "max_distortion_gain": self.max_distortion_gain,
            "pitch_shift_ratio": self.pitch_shift_ratio,
            "max_pitch_shift": self.max_pitch_shift,
        }

    def _get_paths(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)

        return [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, f))
        ]

    def _get_impulses(self, impulse_paths: list):
        impulses = [torchaudio.load(path) for path in impulse_paths]
        impulses = [
            F.resample(
                waveform=wav, orig_freq=sr, new_freq=self.sample_rate
            ).mean(0, keepdim=True)[:, : self.samples_per_chunk]
            for wav, sr in impulses
        ]
        return [
            (wav) / (torch.linalg.vector_norm(wav, ord=2)) for wav in impulses
        ]

    def _get_noise(self, noise_paths: list):
        noises = [torchaudio.load(path) for path in noise_paths]
        noises = [
            F.resample(
                waveform=wav, orig_freq=sr, new_freq=self.sample_rate
            ).mean(0, keepdim=True)[:, : self.samples_per_chunk]
            for wav, sr in noises
        ]

        for wav in noises:
            assert (
                wav.shape[-1] == self.samples_per_chunk
            ), "noise wav too short"

        return noises

    def apply_reverb(self, wav: torch.Tensor):
        # wav: (bz, L)
        batch_size, _ = wav.shape

        reverb_strength = (
            torch.Tensor([random.uniform(0, 1) for _ in range(batch_size)])
            .unsqueeze(-1)
            .to(wav.device)
        )
        reverb_type = random.randint(0, self.num_impulse - 1)
        impulse = getattr(self, f"impulse_{reverb_type}")
        reverb = F.fftconvolve(wav, impulse, mode="full")[
            :, : self.samples_per_chunk
        ]
        res = (reverb_strength * reverb) + ((1 - reverb_strength) * wav)

        return res

    def apply_noise(self, wav: torch.Tensor):
        batch_size, _ = wav.shape

        snr_dbs = torch.tensor(
            [
                random.randint(self.min_noise_snr, self.max_noise_snr)
                for _ in range(batch_size)
            ]
        ).to(wav.device)
        noise_type = random.randint(0, self.num_noise - 1)
        noise = getattr(self, f"noise_{noise_type}")

        return F.add_noise(waveform=wav, noise=noise, snr=snr_dbs)

    def apply_bandpass(self, wav: torch.Tensor):
        central_freq = random.randint(1000, 3500)
        Q = random.uniform(0.707, 1.41)

        return torchaudio.functional.bandpass_biquad(
            wav, self.sample_rate, central_freq, Q
        )

    def apply_distortion(self, wav: torch.Tensor):
        gain = random.randint(
            self.min_distortion_gain, self.max_distortion_gain
        )
        colour = random.randint(5, 95)

        return F.overdrive(wav, gain=gain, colour=colour)

    def distortion_aug_cpu(self, wav: torch.Tensor):
        # This function should run on the cpu (i.e. in the dataloader collate
        # function) in order to not be a bottlekneck

        if random.random() < self.distortion_ratio:
            wav = self.apply_distortion(wav)

        return wav

    def shift_spec(self, spec: torch.Tensor, shift_st: int | float):
        # Spec (bz, n_mels, L) ??
        if shift_st == 0:
            return spec

        freq_mult = 2 ** (shift_st / 12.0)
        _, num_bins, L = spec.shape
        new_num_bins = int(num_bins * freq_mult)

        # Interpolate expects extra channel dim
        spec = spec.unsqueeze(1)
        shifted_specs = torch.nn.functional.interpolate(
            spec, size=(new_num_bins, L), mode="bilinear", align_corners=False
        )
        shifted_specs = shifted_specs.squeeze(1)

        if shift_st > 0:
            shifted_specs = shifted_specs[:, :num_bins, :]
        else:
            padding = num_bins - shifted_specs.size(1)
            shifted_specs = torch.nn.functional.pad(
                shifted_specs, (0, 0, 0, padding), "constant", 0
            )

        return shifted_specs

    # TODO: Debug why noise is causing NaN loss
    def aug_wav(self, wav: torch.Tensor):
        # This function doesn't apply distortion. If distortion is desired it
        # should be run beforehand on the cpu with distortion_aug_cpu. Note
        # also that shifting is done to the spectrogram in log_mel, not the wav.

        # Noise
        if random.random() < self.noise_ratio:
            wav = self.apply_noise(wav)

        # Reverb
        if random.random() < self.reverb_ratio:
            wav = self.apply_reverb(wav)

        # EQ
        if random.random() < self.bandpass_ratio:
            wav = self.apply_bandpass(wav)

        return wav

    def norm_mel(self, mel_spec: torch.Tensor):
        # This norm formula is taken from Whisper
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        max_over_mels = log_spec.max(dim=1, keepdim=True)[0]
        max_log_spec = max_over_mels.max(dim=2, keepdim=True)[0]
        log_spec = torch.maximum(log_spec, max_log_spec - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec

    def log_mel(self, wav: torch.Tensor, shift: bool = False):
        spec = self.spec_transform(wav)[..., :-1]

        if random.random() < self.pitch_shift_ratio and shift == True:
            pitch_shift_st = random.uniform(
                -self.max_pitch_shift, self.max_pitch_shift
            )
            spec = self.shift_spec(
                spec=spec,
                shift_st=pitch_shift_st,
            )

        mel_spec = self.mel_transform(spec)
        log_mel = self.norm_mel(mel_spec)

        return log_mel

    def forward(self, wav: torch.Tensor):
        # Noise, and reverb
        wav = self.aug_wav(wav)

        # Spec, pitch shift
        log_mel = self.log_mel(wav, shift=True)

        return log_mel

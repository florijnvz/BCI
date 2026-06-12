"""
Batch feature extraction for the bi2015a (Brain Invaders) dataset.

Loads preprocessed .fif epoch files produced by 00_batch_preprocessing.py
and extracts feature sets per session, saving one .npz per session.

Feature sets
------------
  erp        : flattened post-stimulus waveform, all 32 ch, ds×16
  delta_wave : bandpass 0.5–3 Hz (butter order 4, filtfilt), DELTA_CHANNELS (9 ch), ds×16, flattened
  theta_wave : bandpass 3–7 Hz (butter order 4, filtfilt), THETA_CHANNELS (6 ch), ds×16, flattened
  dt_wave    : delta_wave + theta_wave concatenated

Usage
-----
    python 00_batch_feature_extraction.py

To verify with a subset first, set SUBJECTS = [3, 4] below, check output,
then restore the full subject list for the complete batch.

Excluded subjects
-----------------
  1, 27  — flagged in bi2015a paper (power-line interference / corrupted)
  7, 11, 18, 29 — all epochs dropped or < 20 surviving epochs after 100µV rejection
"""

import logging
import traceback
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, filtfilt

mne.set_log_level("WARNING")

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_ROOT = Path(__file__).parent / "data" / "preprocessed"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
log_path = OUTPUT_ROOT / "feature_extraction_log.txt"
logger   = logging.getLogger("batch_feature_extraction")
logger.setLevel(logging.DEBUG)

_fmt     = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
_console = logging.StreamHandler()
_console.setFormatter(_fmt)
_file    = logging.FileHandler(log_path, mode="a", encoding="utf-8")
_file.setFormatter(_fmt)
logger.addHandler(_console)
logger.addHandler(_file)

# ── Dataset constants ─────────────────────────────────────────────────────────
FLASH_DURATION_MS = {1: 110, 2: 80, 3: 50}
MIN_EPOCHS        = 20
SUBJECTS_TO_SKIP  = [1, 7, 11, 18, 27, 29]

SESSIONS = [1, 2, 3]
SUBJECTS = [s for s in range(2, 44) if s not in SUBJECTS_TO_SKIP]

# Fixed 32-channel order (same as preprocessing CSV column order)
ALL_CH_NAMES = [
    'FP1', 'FP2', 'AFz', 'F7',  'F3',  'F4',  'F8',
    'FC5', 'FC1', 'FC2', 'FC6', 'T7',  'C3',  'Cz',  'C4',  'T8',
    'CP5', 'CP1', 'CP2', 'CP6', 'P7',  'P3',  'Pz',  'P4',  'P8',
    'PO7', 'O1',  'Oz',  'O2',  'PO8', 'PO9', 'PO10',
]

# ── Channel subsets for bandpass waveform features ────────────────────────────
#
# Delta and theta band responses have distinct spatial distributions over the
# scalp, so we use different electrode subsets for each band rather than all 32.
# Using focused channel sets reduces feature dimensionality and noise from
# electrodes where the band is weak.
#
# Delta (0.5–3 Hz) — posterior / centro-parietal channels
#   The P300 slow-wave component is dominated by delta-band activity and is
#   maximal over parietal and central-parietal sites (Cz, Pz, CP electrodes).
#   Using posterior channels captures the attention-related late positivity that
#   distinguishes target from non-target trials.
#   Reference: Polich, J. (2007). Updating P300: An integrative theory of P3a
#   and P3b. Clinical Neurophysiology, 118(10), 2128–2148.
#
# Theta (3–7 Hz) — frontal / fronto-central channels
#   Theta power increases over frontal sites during working memory updating and
#   attentional gating — both of which occur when the subject detects a target.
#   Frontal theta is thought to reflect executive processes coordinated by the
#   anterior cingulate and prefrontal cortex.
#   Reference: Polich (2007), ibid.; also Bachman, M. D., & Bernat, E. M. (2015).
#   Independent contributions of theta and delta ERP activity to the N2 and P3
#   components. International Journal of Psychophysiology, 99, 1–11.
DELTA_CHANNELS = ['Cz', 'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'P8']
THETA_CHANNELS = ['AFz', 'FC1', 'FC2', 'F3', 'F4', 'Cz']

# Pre-compute the integer index of each channel name inside ALL_CH_NAMES so that
# numpy array slicing (which needs integer indices) is fast and unambiguous.
DELTA_INDICES  = [ALL_CH_NAMES.index(ch) for ch in DELTA_CHANNELS]
THETA_INDICES  = [ALL_CH_NAMES.index(ch) for ch in THETA_CHANNELS]

# ── Bandpass filter parameters ────────────────────────────────────────────────
#
# EEG frequency bands are defined by the dominant neural rhythms seen across
# different cognitive states.  A bandpass filter passes only the signal energy
# between a lower (low) and upper (high) frequency cutoff and attenuates
# everything outside that window, isolating a particular oscillation of interest.
#
# Delta band: 0.5–3 Hz
#   The lowest EEG rhythm.  In awake, attending subjects it is not prominent at
#   rest, but large-amplitude slow waves emerge in the ERP time-locked to
#   attended (target) stimuli.  Bachman & Bernat (2015) demonstrated that delta-
#   band decomposition of the ERP captures the P300 waveform more precisely than
#   the raw broadband signal, and that delta power over posterior electrodes is
#   specifically enhanced for target versus non-target events.
#   Lower bound 0.5 Hz (not 0 Hz) avoids DC drift artefacts from slow baseline
#   shifts while still capturing the full slow-wave envelope.
#
# Theta band: 3–7 Hz
#   A mid-range rhythm linked to frontal executive processes — especially working
#   memory encoding and attentional suppression of distractors.  In P300 tasks,
#   frontal theta increases for targets versus non-targets, making it a
#   complementary discriminative feature to the posterior delta wave.
#   Reference: Bachman & Bernat (2015), ibid.; Polich (2007), ibid.
DELTA_LOW  = 0.5
DELTA_HIGH = 3.0
THETA_LOW  = 3.0
THETA_HIGH = 7.0

# Butterworth filter order: higher order → steeper roll-off outside the passband,
# but also more numerical ringing near the cutoffs.  Order 4 is a standard
# compromise for EEG that gives ~80 dB/decade attenuation without instability.
BP_ORDER   = 4

# Original sampling rate of the bi2015a recordings after preprocessing.
# All time-axis computations must use this value to convert between samples
# and seconds correctly.
SFREQ      = 512.0


# ── Feature extraction helpers ────────────────────────────────────────────────
def extract_erp(epoch_data, times, downsample_factor=16, channel_indices=None):
    """Crop to the post-stimulus window, optionally restrict channels,
    downsample by downsample_factor, then flatten each epoch to a 1-D vector.

    Parameters
    ----------
    epoch_data : ndarray, shape (n_epochs, n_channels, n_times)
        Raw (or bandpass-filtered) EEG epochs as output by MNE.
    times : ndarray, shape (n_times,)
        Time axis in seconds for each sample in epoch_data, where t=0 is the
        stimulus onset.  Negative values are the pre-stimulus baseline period.
    downsample_factor : int
        Keep every Nth sample along the time axis.  With SFREQ=512 Hz and
        downsample_factor=16, the effective sample rate is 512/16 = 32 Hz.
        32 Hz is well above the Nyquist frequency for the highest band of
        interest (theta at 7 Hz requires > 14 Hz), so no information is lost.
        The 16× reduction shrinks each feature vector by the same factor,
        making classifier training much faster.
    channel_indices : list of int or None
        Integer indices (into the channel axis) to keep.  Pass None to keep
        all channels (used for the broadband ERP feature).

    Returns
    -------
    ndarray, shape (n_epochs, n_channels_kept * n_post_times_downsampled)
        One flattened feature vector per epoch.  Flattening converts the 2-D
        (channels × time) matrix into a 1-D array so that it can be passed
        directly to scikit-learn classifiers, which expect (n_samples, n_features).
    """
    # Step 1 — Crop to the post-stimulus window (t ≥ 0).
    # The pre-stimulus period is used only as a baseline reference during
    # preprocessing and contains no ERP response.  Discarding it prevents the
    # classifier from fitting to baseline fluctuations and halves the feature
    # length (the epoch is typically −0.2 s to +0.8 s, so roughly half is pre).
    mask    = times >= 0                   # boolean mask: True for every sample at or after stimulus onset
    cropped = epoch_data[:, :, mask]       # shape: (n_epochs, n_channels, n_post_times)

    # Step 2 — Optionally restrict to a channel subset.
    # For the broadband ERP we keep all 32 channels; for delta/theta features we
    # keep only the anatomically relevant subset defined above.
    if channel_indices is not None:
        cropped = cropped[:, channel_indices, :]   # shape: (n_epochs, n_selected_ch, n_post_times)

    # Step 3 — Downsample by taking every Nth sample along the time axis (axis=2),
    # then flatten the (channels, downsampled_times) matrix into a 1-D vector per
    # epoch with reshape(..., -1).  The -1 lets NumPy infer the total feature count
    # automatically.  Channel order is preserved: all time points for channel 0,
    # then all for channel 1, etc.
    return cropped[:, :, ::downsample_factor].reshape(len(epoch_data), -1)


def bandpass_filter(data, sfreq, low, high, order=4):
    """Zero-phase Butterworth bandpass filter applied along the time axis.

    A Butterworth filter is chosen because its passband is maximally flat
    (no amplitude ripple), which preserves the relative amplitudes of ERP
    components within the band.

    filtfilt (forward-backward filtering) is used instead of a one-pass filter.
    Applying the filter twice — once forward, once in reverse — cancels the
    phase delay that a single-pass filter would introduce.  Zero phase delay is
    critical for ERP research: we need peak latencies (e.g. the P300 at ~300 ms)
    to remain aligned with their true neural timing.

    Parameters
    ----------
    data  : ndarray, shape (n_epochs, n_channels, n_times)
        EEG data to filter.
    sfreq : float
        Sampling frequency in Hz.
    low   : float
        Lower edge of the passband in Hz.
    high  : float
        Upper edge of the passband in Hz.
    order : int
        Filter order (poles per pass direction; effective order is doubled by
        filtfilt, so order=4 gives an effective 8th-order filter).

    Returns
    -------
    ndarray, same shape as data
        Band-limited signal; all frequency content outside [low, high] is
        strongly attenuated.
    """
    # The Butterworth design function works in normalised frequency (0–1), where
    # 1.0 corresponds to the Nyquist frequency (half the sample rate).  Dividing
    # by nyq converts Hz → normalised units.
    nyq = sfreq / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype='band')

    # axis=2 tells filtfilt to filter along the time axis (the last dimension),
    # processing each epoch × channel combination independently.
    return filtfilt(b, a, data, axis=2)


# ── Core function ─────────────────────────────────────────────────────────────
def extract_session(subject, session, output_root):
    """Extract all feature sets for one subject/session and save to .npz.

    The function reads the preprocessed MNE epoch file (.fif), computes four
    complementary feature representations, and bundles them into a single
    NumPy archive (.npz).

    Why .npz?
    ---------
    np.savez stores multiple named arrays in a single compressed file.  It is
    fast to write and load (np.load returns a dict-like object), preserves
    exact floating-point values without rounding (unlike CSV), and keeps all
    features for a session together so downstream code only needs one file
    path per session.
    """
    flash_ms  = FLASH_DURATION_MS[session]
    fif_path  = output_root / f"subject_{subject:02d}_session_{session:02d}_epo.fif"
    save_path = output_root / f"subject_{subject:02d}_session_{session:02d}_features.npz"

    epochs = mne.read_epochs(fif_path, verbose=False)

    # ── Load the epoch array and metadata ─────────────────────────────────────
    # get_data(picks="eeg") returns a float64 array of shape
    # (n_epochs, n_eeg_channels, n_times).  We confirm the channel count is 32
    # by construction (ALL_CH_NAMES above matches the preprocessing output).
    data   = epochs.get_data(picks="eeg")          # (n_epochs, 32, n_times)
    times  = epochs.times                           # 1-D array of time values in seconds; t=0 = stimulus onset

    # Labels encode the subject's behavioural role for each trial:
    #   1 = "target"     — the row or column being flashed contains the character
    #                       the subject was attending to; a P300 is expected.
    #   0 = "non-target" — the flash was irrelevant to the subject's goal;
    #                       no P300 expected, only a small sensory response.
    # In the Brain Invaders paradigm a 6×6 symbol matrix is shown; one row and
    # one column flash per "round" are targets, the other 10 are non-targets.
    # The classifier trained on these labels learns to detect the P300 and
    # thereby identify which character the subject intended.
    labels = epochs.metadata["target"].values       # 1=target, 0=non-target

    # ── Feature set 1: broadband ERP waveform ─────────────────────────────────
    # All 32 channels, post-stimulus only, downsampled to 32 Hz and flattened.
    # This represents the full scalp topography of the averaged evoked response
    # without any frequency decomposition.
    features_erp = extract_erp(data, times)

    # ── Feature sets 2 & 3: band-limited waveforms ────────────────────────────
    # Apply bandpass filters to isolate the delta and theta oscillations embedded
    # in the broadband signal before extracting post-stimulus waveforms.
    # Filtering is done on the full epoch (including pre-stimulus) so that
    # filtfilt has enough signal on both sides to avoid edge artefacts.
    data_delta = bandpass_filter(data, SFREQ, DELTA_LOW, DELTA_HIGH, BP_ORDER)
    data_theta = bandpass_filter(data, SFREQ, THETA_LOW, THETA_HIGH, BP_ORDER)

    # For each filtered signal, extract_erp then crops to post-stimulus, selects
    # the anatomically appropriate channel subset, downsamples to 32 Hz, and
    # flattens — producing a compact 1-D feature vector per epoch.
    features_delta_wave = extract_erp(data_delta, times, channel_indices=DELTA_INDICES)
    features_theta_wave = extract_erp(data_theta, times, channel_indices=THETA_INDICES)

    # ── Feature set 4: combined delta + theta ─────────────────────────────────
    # Horizontal stack concatenates the delta and theta feature vectors into a
    # single longer vector.  This allows a classifier to exploit both the
    # posterior slow-wave (delta) and the frontal attention signal (theta)
    # simultaneously, often outperforming either band alone.
    features_dt_wave    = np.hstack([features_delta_wave, features_theta_wave])

    # ── Save all feature sets to a single compressed NumPy archive ────────────
    # Each keyword argument becomes a named array accessible via
    # np.load(path)[key].  Scalar metadata (flash_ms, subject, session) are
    # stored alongside the feature matrices so each file is self-contained.
    np.savez(
        save_path,
        erp        = features_erp,
        delta_wave = features_delta_wave,
        theta_wave = features_theta_wave,
        dt_wave    = features_dt_wave,
        labels     = labels,
        flash_ms   = flash_ms,
        subject    = subject,
        session    = session,
    )

    return dict(
        erp_shape        = features_erp.shape,
        delta_wave_shape = features_delta_wave.shape,
        theta_wave_shape = features_theta_wave.shape,
        n_target         = int(labels.sum()),
        n_nontarget      = int((labels == 0).sum()),
    )


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    total   = len(SUBJECTS) * len(SESSIONS)
    counter = 0

    n_processed      = 0
    n_already_exists = 0
    n_missing_fif    = 0
    n_low_epochs     = 0
    n_errors         = 0

    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info(f"Batch feature extraction started  {start_time:%Y-%m-%d %H:%M:%S}")
    logger.info(f"Subjects       : {SUBJECTS}")
    logger.info(f"Sessions       : {SESSIONS}")
    logger.info(f"Delta channels : {DELTA_CHANNELS}  band={DELTA_LOW}–{DELTA_HIGH} Hz  order={BP_ORDER}")
    logger.info(f"Theta channels : {THETA_CHANNELS}  band={THETA_LOW}–{THETA_HIGH} Hz  order={BP_ORDER}")
    logger.info(f"Output         : {OUTPUT_ROOT}")
    logger.info("=" * 60)

    for subject in SUBJECTS:
        for session in SESSIONS:
            counter  += 1
            flash_ms  = FLASH_DURATION_MS[session]
            tag       = f"[{counter:3d}/{total}] S{subject:02d} ses{session}"

            fif_path  = OUTPUT_ROOT / f"subject_{subject:02d}_session_{session:02d}_epo.fif"
            save_path = OUTPUT_ROOT / f"subject_{subject:02d}_session_{session:02d}_features.npz"

            if not fif_path.exists():
                logger.warning(f"{tag} — SKIPPED (.fif not found)")
                n_missing_fif += 1
                continue

            if save_path.exists():
                logger.info(f"{tag} — already exists, skipping")
                n_already_exists += 1
                continue

            try:
                epochs = mne.read_epochs(fif_path, verbose=False)
            except Exception:
                logger.error(f"{tag} — ERROR reading .fif\n{traceback.format_exc()}")
                n_errors += 1
                continue

            if len(epochs) == 0:
                logger.warning(f"{tag} — SKIPPED (0 epochs in .fif)")
                n_low_epochs += 1
                continue
            if len(epochs) < MIN_EPOCHS:
                logger.warning(f"{tag} — SKIPPED ({len(epochs)} epochs < {MIN_EPOCHS} threshold)")
                n_low_epochs += 1
                continue

            try:
                result = extract_session(subject, session, OUTPUT_ROOT)
                logger.info(
                    f"{tag} ({flash_ms} ms) — "
                    f"ERP:{result['erp_shape']}, "
                    f"DeltaWave:{result['delta_wave_shape']}, "
                    f"ThetaWave:{result['theta_wave_shape']}, "
                    f"Labels:{result['n_target']}T/{result['n_nontarget']}NT"
                )
                n_processed += 1

            except Exception:
                logger.error(f"{tag} — ERROR\n{traceback.format_exc()}")
                n_errors += 1

    elapsed   = datetime.now() - start_time
    n_skipped = n_already_exists + n_missing_fif + n_low_epochs + n_errors

    logger.info("")
    logger.info("=" * 60)
    logger.info("BATCH FEATURE EXTRACTION COMPLETE")
    logger.info(f"Elapsed   : {elapsed}")
    logger.info(f"Processed : {n_processed}")
    logger.info(
        f"Skipped   : {n_skipped}  "
        f"(already existed: {n_already_exists}, "
        f"missing .fif: {n_missing_fif}, "
        f"low epochs: {n_low_epochs}, "
        f"errors: {n_errors})"
    )
    logger.info(f"Output    : {OUTPUT_ROOT}")
    logger.info("=" * 60)

    print()
    print("=== BATCH FEATURE EXTRACTION COMPLETE ===")
    print(f"Processed : {n_processed}")
    print(f"Skipped   : {n_skipped}  (already existed: {n_already_exists}, errors: {n_errors}, low epochs: {n_low_epochs})")
    print(f"Output    : {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()

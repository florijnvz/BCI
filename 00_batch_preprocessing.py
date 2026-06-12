"""
Batch preprocessing for the bi2015a (Brain Invaders) dataset.

Runs the full MNE preprocessing pipeline on all subjects and sessions,
saving one .fif epochs file per session to OUTPUT_ROOT.

Usage
-----
    python 00_batch_preprocessing.py

To test with a subset first, set SUBJECTS = [3, 4] below, verify output,
then restore SUBJECTS = list(range(1, 44)) for the full batch.

Note: subjects 7, 11, 18 had 0 epochs in 2-3 sessions after artifact rejection.
Subject 29 had suspiciously low epoch counts (9-12 per session).
These are excluded at the classification stage, not preprocessing.
"""

import logging
import traceback
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import pandas as pd

# Suppress MNE's verbose output so our own log messages stay readable.
# MNE prints a lot of internal detail by default; WARNING level keeps only
# genuinely unexpected events.
mne.set_log_level("WARNING")

# ── Paths ─────────────────────────────────────────────────────────────────────
# RAW_ROOT: where the original CSV files downloaded from bi2015a live.
# OUTPUT_ROOT: where we write the cleaned, epoched .fif files.
# mkdir(parents=True, exist_ok=True) creates the folder (and any missing
# parent folders) the first time; does nothing if it already exists.
RAW_ROOT    = Path(__file__).parent / "data" / "raw"
OUTPUT_ROOT = Path(__file__).parent / "data" / "preprocessed"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
# We write the same log messages to two places: the terminal (so you see
# progress while the script runs) and a log file (so you can review what
# happened after the fact, even if the terminal was closed).
log_path = OUTPUT_ROOT / "preprocessing_log.txt"
logger   = logging.getLogger("batch_preprocess")
logger.setLevel(logging.DEBUG)   # capture all severity levels

# Formatter: every line will look like: "2024-01-01 12:00:00  INFO      message"
_fmt     = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")

# Console handler — prints to the terminal
_console = logging.StreamHandler()
_console.setFormatter(_fmt)

# File handler — appends to the log file (mode="a" keeps previous runs)
_file    = logging.FileHandler(log_path, mode="a", encoding="utf-8")
_file.setFormatter(_fmt)

logger.addHandler(_console)
logger.addHandler(_file)

# ── Dataset constants ─────────────────────────────────────────────────────────
# The bi2015a CSV files have no header row, so we supply the column names
# ourselves.  The order here must exactly match the column order in the files.
#
# Column layout:
#   Time      — timestamp (we don't actually use this; MNE tracks time via SFREQ)
#   FP1…PO10 — 32 EEG electrode readings, named by their position on the scalp
#               using the standard 10-05 system (FP = frontal-polar, O = occipital, etc.)
#   Trigger   — a number written by the stimulus PC each time a face/object flashed;
#               non-zero only at flash onset, so MNE can find where stimuli occurred
#   Target    — 1 if the flash was the target face the participant was watching for,
#               0 if it was a non-target (distractor)
COLUMNS = [
    "Time",
    "FP1", "FP2", "AFz", "F7",  "F3",  "F4",  "F8",
    "FC5", "FC1", "FC2", "FC6", "T7",  "C3",  "Cz",  "C4",  "T8",
    "CP5", "CP1", "CP2", "CP6", "P7",  "P3",  "Pz",  "P4",  "P8",
    "PO7", "O1",  "Oz",  "O2",  "PO8", "PO9", "PO10",
    "Trigger", "Target",
]

# Slice out only the 32 EEG electrode names (positions 1–32 in the list above).
EEG_CHANNELS = COLUMNS[1:33]

# The amplifier recorded at 512 samples per second.
SFREQ = 512.0

# Each session used a different stimulus flash duration (milliseconds).
# Session 1 is slowest (110 ms), session 3 is fastest (50 ms).
# This is stored per-session so we can log it and sanity-check results.
FLASH_DURATION_MS = {1: 110, 2: 80, 3: 50}

# ── Subject / session config ──────────────────────────────────────────────────
# SUBJECTS_TO_SKIP: subjects flagged in the original bi2015a paper as having
# hardware problems (power-line interference, corrupted recordings).  We skip
# these entirely rather than trying to salvage bad data.
SUBJECTS_TO_SKIP = [1, 27]

# SUBJECTS_TO_EXCLUDE_POST: subjects whose recordings survived preprocessing
# but ended up with too few clean epochs for reliable classification (< 20
# after the 100 µV artifact rejection step).  We still preprocess them here
# so the .fif files exist; the classification scripts skip them at load time.
SUBJECTS_TO_EXCLUDE_POST = [7, 11, 18, 29]

# Full subject list (1–43).  Change to a small list like [3, 4] to do a
# quick end-to-end test before committing to the full batch run.
SUBJECTS = list(range(1, 44))   # ← change to [3, 4] for a quick test
SESSIONS = [1, 2, 3]


# ── Core function ─────────────────────────────────────────────────────────────
def preprocess_session(subject, session, raw_root, output_root):
    """Run the full preprocessing pipeline for one subject/session.

    Returns
    -------
    n_epochs : int   — epochs surviving artefact rejection
    n_target : int   — target epochs
    n_nontarget : int — non-target epochs
    """
    # Build the path to this subject/session's CSV file.
    # Example: data/raw/subject_03_csv/subject_03_session_02.csv
    # The :02d format specifier zero-pads single-digit numbers (e.g. 3 → "03").
    csv_path = (raw_root
                / f"subject_{subject:02d}_csv"
                / f"subject_{subject:02d}_session_{session:02d}.csv")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    # header=None tells pandas there is no header row in the file.
    # names=COLUMNS assigns our column names in order.
    df = pd.read_csv(csv_path, header=None, names=COLUMNS)

    # Force these columns to integer type.  The CSV may store them as floats
    # (e.g. "1.0"), so we cast explicitly to avoid mismatches later when we
    # index into numpy arrays using these values.
    df["Trigger"] = df["Trigger"].astype(int)
    df["Target"]  = df["Target"].astype(int)

    # ── Build MNE RawArray ────────────────────────────────────────────────────
    # MNE expects EEG data in Volts, but the amplifier stored values in
    # microvolts (µV).  Dividing by 1,000,000 (1e6) converts µV → V.
    # This matters because MNE's internal thresholds (e.g. for artifact
    # rejection) are all expressed in Volts.
    eeg_data = df[EEG_CHANNELS].values.T / 1e6
    # .values gives a 2-D numpy array of shape (n_samples, 32).
    # .T transposes it to (32, n_samples) because MNE wants (channels, samples).

    # The Trigger column records a non-zero integer at the sample where each
    # stimulus appeared; all other samples are 0.  This is called a STIM
    # (stimulus) channel.  MNE's find_events() will later scan this channel
    # and return a list of (sample_index, 0, trigger_value) tuples so we know
    # exactly when each flash happened.
    # reshape(1, -1) turns a 1-D array of length N into shape (1, N) so it
    # can be stacked vertically with the 2-D EEG data.
    stim_data = df["Trigger"].values.reshape(1, -1).astype(float)

    # Stack EEG (32 rows) and STIM (1 row) into one (33, n_samples) array.
    all_data = np.vstack([eeg_data, stim_data])

    # Give each channel a name and a type so MNE can apply type-specific
    # processing (e.g. filters apply to "eeg" channels only, not "stim").
    # "STI014" is MNE's conventional name for a generic stimulus channel.
    ch_names = EEG_CHANNELS + ["STI014"]
    ch_types = ["eeg"] * 32 + ["stim"]

    # create_info bundles channel names, types, and sampling frequency into
    # a metadata object that MNE attaches to the data.
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types=ch_types)

    # RawArray wraps our numpy array in an MNE Raw object, which gives us
    # access to all of MNE's filtering, epoching, and visualization methods.
    raw = mne.io.RawArray(all_data, info, verbose=False)

    # Assign physical electrode positions using the standard 10-05 layout.
    # on_missing="ignore" silently skips any channel whose name is not in
    # the standard montage (e.g. PO9/PO10 are sometimes absent).
    # This is needed for source-level analysis and some visualization tools,
    # but won't affect our classification pipeline.
    raw.set_montage("standard_1005", on_missing="ignore", verbose=False)

    # ── Filtering ─────────────────────────────────────────────────────────────
    # Step 1 — Notch filter at 50 Hz.
    # Power lines in Europe oscillate at 50 Hz, which induces a strong
    # electrical artifact in any recording made near electrical wiring.
    # A notch filter removes a very narrow band around exactly 50 Hz while
    # leaving everything else intact.  (In the US you would use 60 Hz.)
    raw_notched = raw.copy().notch_filter(50.0, verbose=False)

    # Step 2 — Bandpass filter: keep only 1–30 Hz.
    # l_freq=1.0  removes very slow drifts (DC offset, sweat artifacts,
    #             electrode drift) that contaminate the baseline.
    # h_freq=30.0 removes high-frequency muscle noise (EMG) and residual
    #             electrical noise above what EEG brain signals reach.
    # The ERP components we care about (P300 peaks around 300 ms after a
    # stimulus) live well within 1–30 Hz, so we keep the signal of interest
    # and discard the noise outside that band.
    raw_filtered = raw_notched.copy().filter(l_freq=1.0, h_freq=30.0, verbose=False)

    # ── Epoching ──────────────────────────────────────────────────────────────
    # "Epoching" means cutting the continuous EEG recording into short
    # fixed-length segments (epochs), one per stimulus event.  Instead of
    # working with hours of continuous data, we end up with a 3-D array of
    # shape (n_epochs, n_channels, n_timepoints).
    #
    # find_events scans the STIM channel for transitions from 0 to a non-zero
    # value and returns an (n_events, 3) array where each row is
    # [sample_index, 0, trigger_value].
    events = mne.find_events(raw_filtered, stim_channel="STI014", verbose=False)

    epochs = mne.Epochs(
        raw_filtered,
        events,
        event_id=1,          # only cut epochs around trigger value 1 (every flash)
        tmin=-0.2,           # start each epoch 200 ms BEFORE the flash
                             # (gives us a clean pre-stimulus period to measure
                             #  the brain's "resting" state right before the event)
        tmax=1.0,            # end each epoch 1000 ms AFTER the flash
                             # (the P300 ERP component typically peaks at ~300 ms;
                             #  1 s gives us room to see the full response shape)
        baseline=(-0.2, 0),  # Baseline correction: subtract the mean amplitude
                             # of the pre-stimulus window (−200 ms to 0 ms) from
                             # every timepoint in the epoch.  This removes any
                             # slow DC offset that happened to be present before
                             # the flash, so all epochs start from the same "zero"
                             # and brain responses are comparable across trials.
        reject=dict(eeg=100e-6),  # Artifact rejection: discard any epoch in which
                                  # at least one EEG channel exceeds ±100 µV
                                  # (= 100e-6 V in MNE's Volt units).
                                  # Large voltage swings typically mean the participant
                                  # blinked, moved their eyes, or clenched their jaw —
                                  # all of which create electrical artifacts far larger
                                  # than genuine brain signals (~5–20 µV).
                                  # 100 µV is a standard conservative threshold for
                                  # P300 studies.
        preload=True,        # load all epoch data into RAM immediately so later
                             # operations (metadata, save) work without re-reading
        verbose=False,
    )

    # ── Align Target labels to surviving epochs ───────────────────────────────
    # events[:, 0] gives the sample index of every flash in the raw recording.
    #
    # raw_filtered.first_samp is MNE's internal sample offset.  When MNE loads
    # a file it may not start the internal counter at 0; first_samp holds the
    # offset so that "sample index in file" = "MNE's sample index" − first_samp.
    # Subtracting it converts the event sample indices into row indices into our
    # original DataFrame so we can look up the corresponding Target label.
    labels_all = df["Target"].values[events[:, 0] - raw_filtered.first_samp]

    # epochs.drop_log is a list with one entry per original event.
    # Each entry is an empty tuple () if the epoch survived, or a tuple of
    # strings describing why it was dropped (e.g. ('EOG',) or ('EEG',)).
    # We collect the indices of all epochs that were NOT dropped (empty tuple).
    kept_idx = [i for i, d in enumerate(epochs.drop_log) if len(d) == 0]

    # Keep only the Target labels for the epochs that survived rejection.
    labels_kept = labels_all[kept_idx]

    # ── Attach metadata ───────────────────────────────────────────────────────
    # epochs.metadata is a pandas DataFrame with one row per surviving epoch.
    # Storing the target label here keeps the label permanently attached to
    # the epoch data in the .fif file, so downstream scripts can load a single
    # file and immediately have both the EEG data and its label without needing
    # to re-parse the original CSV.
    epochs.metadata = pd.DataFrame(
        {"target": labels_kept},
        index=range(len(labels_kept)),
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    # .fif is MNE's native binary format.  It stores EEG data, channel info,
    # event information, and metadata in one file.
    # overwrite=True lets us re-run the script safely; it replaces any
    # previously written file for this subject/session.
    fif_path = output_root / f"subject_{subject:02d}_session_{session:02d}_epo.fif"
    epochs.save(fif_path, overwrite=True, verbose=False)

    # Return counts for logging: total epochs kept, how many were targets,
    # how many were non-targets, and how many timepoints each epoch has.
    # shape[2] is the time axis in MNE's (n_epochs, n_channels, n_times) layout.
    n_times = epochs.get_data(copy=False).shape[2]
    return len(epochs), int(labels_kept.sum()), int((labels_kept == 0).sum()), n_times


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    # Total number of subject/session combinations we will attempt.
    total   = len(SUBJECTS) * len(SESSIONS)
    counter = 0

    # Counters for the final summary — track what happened to each session.
    n_processed      = 0   # successfully preprocessed and saved
    n_already_exists = 0   # .fif already on disk, skipped to avoid re-work
    n_skipped_list   = 0   # in SUBJECTS_TO_SKIP (known bad hardware)
    n_skipped_csv    = 0   # CSV file not found (session may not exist)
    n_errors         = 0   # unexpected exception during processing

    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info(f"Batch preprocessing started  {start_time:%Y-%m-%d %H:%M:%S}")
    logger.info(f"Subjects : {SUBJECTS}")
    logger.info(f"Sessions : {SESSIONS}")
    logger.info(f"Output   : {OUTPUT_ROOT}")
    logger.info("=" * 60)

    for subject in SUBJECTS:
        for session in SESSIONS:
            counter  += 1
            flash_ms  = FLASH_DURATION_MS[session]   # for the log message only
            tag       = f"[{counter:3d}/{total}] S{subject:02d} ses{session}"

            # ── Skip: known-bad subjects ──────────────────────────────────────
            # Subjects in SUBJECTS_TO_SKIP have hardware-level problems noted in
            # the dataset paper.  No amount of filtering can fix corrupted data,
            # so we skip them entirely rather than wasting time and producing
            # unreliable output files.
            if subject in SUBJECTS_TO_SKIP:
                logger.warning(f"{tag} — SKIPPED (subject in SUBJECTS_TO_SKIP)")
                n_skipped_list += 1
                continue

            # ── Skip: missing CSV ─────────────────────────────────────────────
            # Not every subject completed every session; if the file doesn't
            # exist we log a warning and move on instead of crashing.
            csv_path = (RAW_ROOT
                        / f"subject_{subject:02d}_csv"
                        / f"subject_{subject:02d}_session_{session:02d}.csv")
            if not csv_path.exists():
                logger.warning(f"{tag} — SKIPPED (CSV not found: {csv_path})")
                n_skipped_csv += 1
                continue

            # ── Skip: already preprocessed ───────────────────────────────────
            # Preprocessing is slow (~10–30 s per session).  If we have already
            # produced the output file, skip re-processing so we can safely
            # re-run the script after a partial failure without wasting time.
            fif_path = OUTPUT_ROOT / f"subject_{subject:02d}_session_{session:02d}_epo.fif"
            if fif_path.exists():
                logger.info(f"{tag} — already exists, skipping")
                n_already_exists += 1
                continue

            # ── Run pipeline ──────────────────────────────────────────────────
            # Wrap in try/except so a single bad session does not abort the
            # entire batch.  The full Python traceback is logged so we can
            # diagnose failures after the run finishes.
            try:
                n_ep, n_t, n_nt, n_times = preprocess_session(
                    subject, session, RAW_ROOT, OUTPUT_ROOT
                )
                logger.info(
                    f"{tag} ({flash_ms} ms) — "
                    f"{n_ep} epochs ({n_t} target, {n_nt} non-target), "
                    f"shape (_, 32, {n_times})"
                )
                n_processed += 1

            except Exception:
                logger.error(
                    f"{tag} — ERROR\n{traceback.format_exc()}"
                )
                n_errors += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed  = datetime.now() - start_time
    n_skipped = n_skipped_list + n_skipped_csv + n_already_exists

    logger.info("")
    logger.info("=" * 60)
    logger.info("BATCH PREPROCESSING COMPLETE")
    logger.info(f"Elapsed   : {elapsed}")
    logger.info(f"Processed : {n_processed}")
    logger.info(
        f"Skipped   : {n_skipped}  "
        f"(already existed: {n_already_exists}, "
        f"skip list: {n_skipped_list}, "
        f"missing CSV: {n_skipped_csv}, "
        f"errors: {n_errors})"
    )
    logger.info(f"Output    : {OUTPUT_ROOT}")
    logger.info("=" * 60)

    print()
    print("=== BATCH PREPROCESSING COMPLETE ===")
    print(f"Processed : {n_processed}")
    print(f"Skipped   : {n_skipped}  (already existed: {n_already_exists}, errors: {n_errors})")
    print(f"Output    : {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()

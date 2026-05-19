from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
import matplotlib.pyplot as plt

import csv

PROJECT_DIR = Path(__file__).resolve().parent
sfreq       = 300

# =========================================================
# CHANNEL DEFINITIONS
# =========================================================

CH_MAP = {
    "S1:CZ":    "Cz",
    "S2:CP2":   "CP2",
    "S3:CP3":   "CP3",
    "S4:FC2":   "FC2",
    "S5:FC3":   "FC3",
    "S6:vEOGt": "vEOGt",
    "S7:vEOGb": "vEOGb",
}

EEG_CHANNELS        = ["S1:CZ", "S2:CP2", "S3:CP3", "S4:FC2", "S5:FC3"]
EOG_CHANNELS        = ["S6:vEOGt", "S7:vEOGb"]
ALL_SIGNAL_CHANNELS = EEG_CHANNELS + EOG_CHANNELS

MI_TMIN, MI_TMAX = 0, 4

# =========================================================
# CONFIGURATION — edit this section before running
# =========================================================

SUBJECTS  = ["Sub2_data"]   # list of subject folder names to process

EXCLUDE_SESSIONS = {
    "Sub1_data": {5, 6, 12},
    "Sub2_data": {8},
    "Sub3_data": {14, 15, 19},   # matches MARKED_SESSIONS in 06b
}
EXCLUDE_SESSIONS_DEFAULT = {5, 6, 12}

# Plot colours / markers
TRAIN_COLORS  = ["#E53935", "#1565C0", "#F9A825", "#AB47BC", "#FF7043"]   # red, blue, yellow, purple, orange
TEST_COLOR    = "black"
CLASS_MARKERS = {0: "o", 1: "^"}   # circle = MI Left, triangle = MI Right
CLASS_LABELS  = {0: "MI Left", 1: "MI Right"}

APPLY_ICA     = True
ICA_THRESHOLD = 0.3

# ---- Classifier selection ----
# "csp_lda" : CSP + shrinkage LDA with Riemannian Alignment
# "eegnet"  : EEGNet (PyTorch) with per-config z-scoring, no RA
CLASSIFIER = "csp_lda"

# CSP hyperparameters (only used when CLASSIFIER == "csp_lda")
CSP_COMPONENTS = 4   # upper bound; auto-capped to (n_remaining_channels - 1)

# EEGNet hyperparameters (only used when CLASSIFIER == "eegnet")
EEGNET_F1       = 8
EEGNET_D        = 2
EEGNET_F2       = 16
EEGNET_KERN_LEN = sfreq // 2   # 150 samples @ 300 Hz
EEGNET_DROPOUT  = 0.5
EEGNET_EPOCHS   = 300
EEGNET_BATCH    = 16
EEGNET_LR       = 1e-3
EEGNET_DEVICE       = "cuda"   # "auto" = cuda if available, else cpu; or force "cuda" / "cpu"
EEGNET_USE_VAL      = False     # True  = split last EEGNET_VAL_SESSIONS sessions for validation
                                # False = all sessions used for training, no validation tracking
EEGNET_VAL_SESSIONS = 2        # number of trailing sessions held out (only when EEGNET_USE_VAL=True)

# Order of EEG channels in X (must match CH_MAP / MNE pick order)
EEG_CH_NAMES = ["Cz", "CP2", "CP3", "FC2", "FC3"]

# =========================================================
# HELPER: PREPROCESS ONE SESSION → MI EPOCHS (EEG only)
# =========================================================

def load_session(file_path):
    df = pd.read_csv(file_path, comment="#", skipinitialspace=True)

    trigger_col = None
    for candidate in ["Manual trigger", "Trigger"]:
        if candidate in df.columns:
            temp = pd.to_numeric(df[candidate], errors="coerce").fillna(0).astype(int)
            if temp.sum() > 0:
                trigger_col = candidate
                df[candidate] = temp
                break

    if trigger_col is None:
        return None, None

    trigger = df[trigger_col].astype(int)

    baseline_rows   = trigger[trigger.isin([1, 2, 3, 4, 5])].index
    mi_onset_cutoff = int(baseline_rows[-1]) if len(baseline_rows) > 0 else 0

    data_cols = ALL_SIGNAL_CHANNELS + [trigger_col]
    ch_names  = [CH_MAP[c] for c in ALL_SIGNAL_CHANNELS] + ["STI"]
    ch_types  = ["eeg"] * 5 + ["eog"] * 2 + ["stim"]

    data = df[data_cols].values.T.astype(float)
    data[:-1] *= 1e-6
    data = np.nan_to_num(data)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw  = mne.io.RawArray(data, info, verbose=False)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")

    raw.filter(l_freq=6.0, h_freq=50.0, method="fir",
               fir_window="hamming", verbose=False)

    if APPLY_ICA:
        ica = mne.preprocessing.ICA(
            n_components=len(EEG_CHANNELS), random_state=42,
            max_iter="auto", verbose=False,
        )
        ica.fit(raw, picks="eeg", verbose=False)

        sources   = ica.get_sources(raw).get_data()
        eog_data  = raw.get_data(picks=["vEOGt", "vEOGb"])
        veog_diff = eog_data[0] - eog_data[1]

        corrs = np.array([
            np.abs(np.corrcoef(sources[i], veog_diff)[0, 1])
            for i in range(sources.shape[0])
        ])
        best = int(np.argmax(corrs))
        if corrs[best] > ICA_THRESHOLD:
            ica.exclude = [best]
            ica.apply(raw, verbose=False)

    all_events = mne.find_events(raw, stim_channel="STI",
                                 consecutive=True, min_duration=0.01,
                                 verbose=False)

    mi_events = all_events[
        np.isin(all_events[:, 2], [8, 9]) &
        (all_events[:, 0] >= mi_onset_cutoff)
    ]

    if len(mi_events) == 0:
        return None, None

    picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)
    epochs = mne.Epochs(
        raw, mi_events,
        event_id={"MI_Left": 8, "MI_Right": 9},
        tmin=MI_TMIN, tmax=MI_TMAX,
        baseline=None, picks=picks,
        preload=True, verbose=False,
    )

    X = epochs.get_data()
    y = (epochs.events[:, 2] == 9).astype(int)
    return X, y

# =========================================================
# RIEMANNIAN ALIGNMENT — per session (Zanini et al. 2018)
# =========================================================
# Identical operation to Euclidean Alignment (whitening by M^{-1/2}) but
# M is the Fréchet/geometric mean on the SPD manifold rather than the
# arithmetic mean.  This is more geometrically correct because covariance
# matrices live on a curved manifold, not a flat Euclidean space.

def _mat_pow(M, p):
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-12)
    return eigvecs @ np.diag(eigvals ** p) @ eigvecs.T

def _sym_logm(M):
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-12)
    return eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T

def _sym_expm(M):
    eigvals, eigvecs = np.linalg.eigh(M)
    return eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.T

def _riemannian_mean(covs, max_iter=50, tol=1e-8):
    """Fréchet mean on the SPD manifold via fixed-point iteration."""
    M = np.mean(covs, axis=0)
    for _ in range(max_iter):
        M_invsqrt = _mat_pow(M, -0.5)
        M_sqrt    = _mat_pow(M,  0.5)
        tangent   = np.mean([_sym_logm(M_invsqrt @ C @ M_invsqrt)
                             for C in covs], axis=0)
        M = M_sqrt @ _sym_expm(tangent) @ M_sqrt
        if np.linalg.norm(tangent) < tol:
            break
    return M

def riemannian_align(X):
    """Recentre each session by its geometric mean covariance M^{-1/2}."""
    T    = X.shape[2]
    covs = np.array([x @ x.T / T for x in X])
    M    = _riemannian_mean(covs)
    M_invsqrt = _mat_pow(M, -0.5)
    return np.stack([M_invsqrt @ x for x in X], axis=0)

# =========================================================
# CSP + SHRINKAGE LDA
# =========================================================

def train_csp_lda(X_tr, y_tr, X_te, y_te):
    n_csp = min(CSP_COMPONENTS, X_tr.shape[1] - 1)
    csp = CSP(n_components=n_csp, reg=None, log=False, norm_trace=False)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    X_tr_feat = np.nan_to_num(csp.fit_transform(X_tr, y_tr))
    X_te_feat = np.nan_to_num(csp.transform(X_te))

    lda.fit(X_tr_feat, y_tr)
    return (lda.predict(X_te_feat) == y_te).mean() * 100

# =========================================================
# EEGNET + Z-SCORE
# =========================================================

class _EEGNet(object):
    """Pure-PyTorch EEGNet (Lawhern et al. 2018). Lazy import of torch."""

    def __init__(self, n_ch, n_times):
        import torch
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                F1, D, F2 = EEGNET_F1, EEGNET_D, EEGNET_F2
                klen      = EEGNET_KERN_LEN
                pad_t     = klen // 2   # 'same' padding for temporal conv

                self.block1 = nn.Sequential(
                    nn.Conv2d(1, F1, (1, klen), padding=(0, pad_t), bias=False),
                    nn.BatchNorm2d(F1),
                    nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
                    nn.BatchNorm2d(F1 * D),
                    nn.ELU(),
                    nn.AvgPool2d((1, 4)),
                    nn.Dropout(EEGNET_DROPOUT),
                )
                self.block2 = nn.Sequential(
                    nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                              groups=F1 * D, bias=False),
                    nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
                    nn.BatchNorm2d(F2),
                    nn.ELU(),
                    nn.AvgPool2d((1, 8)),
                    nn.Dropout(EEGNET_DROPOUT),
                )
                with torch.no_grad():
                    dummy = torch.zeros(1, 1, n_ch, n_times)
                    out   = self.block2(self.block1(dummy))
                    flat  = out.numel()
                self.fc = nn.Linear(flat, 2)

            def forward(self, x):
                x = self.block2(self.block1(x))
                return self.fc(x.flatten(1))

        self._torch = torch
        self._nn    = nn
        if EEGNET_DEVICE == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(EEGNET_DEVICE)
        print(f"    [EEGNet] device: {self.device}")
        self.model  = _Model().to(self.device)

    def fit(self, X, y, X_val=None, y_val=None):
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        Xt = torch.tensor(X[:, np.newaxis], dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        loader  = DataLoader(TensorDataset(Xt, yt),
                             batch_size=EEGNET_BATCH, shuffle=True)
        opt     = torch.optim.Adam(self.model.parameters(), lr=EEGNET_LR)
        loss_fn = self._nn.CrossEntropyLoss()

        has_val = X_val is not None
        if has_val:
            Xv = torch.tensor(X_val[:, np.newaxis], dtype=torch.float32).to(self.device)
            yv = torch.tensor(y_val, dtype=torch.long).to(self.device)

        self.loss_history     = []
        self.val_loss_history = [] if has_val else None
        self.model.train()
        for _ in range(EEGNET_EPOCHS):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            self.loss_history.append(epoch_loss / len(loader))

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_loss = loss_fn(self.model(Xv), yv).item()
                self.val_loss_history.append(val_loss)
                self.model.train()

    def predict(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            Xt   = torch.tensor(X[:, np.newaxis], dtype=torch.float32).to(self.device)
            return self.model(Xt).argmax(dim=1).cpu().numpy()


def train_eegnet(X_tr, y_tr, X_te, y_te, X_val=None, y_val=None):
    mu    = X_tr.mean(axis=(0, 2), keepdims=True)
    sigma = X_tr.std(axis=(0, 2), keepdims=True)
    sigma = np.where(sigma < 1e-10, 1.0, sigma)
    X_tr_z = (X_tr - mu) / sigma
    X_te_z = (X_te - mu) / sigma
    X_val_z = (X_val - mu) / sigma if X_val is not None else None

    net = _EEGNet(X_tr.shape[1], X_tr.shape[2])
    net.fit(X_tr_z, y_tr, X_val=X_val_z, y_val=y_val)
    preds = net.predict(X_te_z)
    return (preds == y_te).mean() * 100, net.loss_history, net.val_loss_history

# =========================================================
# MAIN
# =========================================================

# (subject, lda_acc, n_sessions, n_train_ses, n_test_ses, n_train_trials, n_test_trials)
all_results = []

for subject in SUBJECTS:

    subject_dir  = PROJECT_DIR / subject
    filtered_dir = subject_dir / "Filtered_data"
    session_files = sorted(
        filtered_dir.glob("*.csv"),
        key=lambda f: int(m.group(1)) if (m := re.search(r"Ses(\d+)", f.stem, re.IGNORECASE)) else 0
    )

    if len(session_files) == 0:
        print(f"[{subject}] No filtered CSV files found — skipping.\n")
        continue

    print(f"{'=' * 60}")
    print(f"  Subject: {subject}  —  {len(session_files)} files found")
    print(f"{'=' * 60}")

    # ----------------------------------------------------------
    # Load, preprocess, reject bad epochs, zero bad channels.
    # Each session is kept as a separate (X, y) entry so we can
    # split chronologically: first 70 % → train, rest → test.
    # Sessions where ALL channels are bad are skipped.
    # ----------------------------------------------------------
    sessions_data: list[tuple] = []   # list of (ses_name, X, y) per session
    n_sessions_used = 0

    for f in session_files:
        ses_name = f.stem.replace("_filtered", "")
        ses_m    = re.search(r"Ses(\d+)", ses_name, re.IGNORECASE)
        ses_num  = int(ses_m.group(1)) if ses_m else None
        excluded = EXCLUDE_SESSIONS.get(subject, EXCLUDE_SESSIONS_DEFAULT)
        if ses_num in excluded:
            print(f"\n  Session: {ses_name}  — excluded (session {ses_num})")
            continue

        print(f"\n  Session: {ses_name}", flush=True)

        X, y = load_session(f)
        if X is None:
            print("    skipped (no MI epochs)")
            continue

        rej_file = (PROJECT_DIR / "outputs" / "epoch_rejection"
                    / subject / f"{ses_name}_bad_epochs.json")
        bad_ch: list[str] = []
        if rej_file.exists():
            with open(rej_file) as fh:
                bad = json.load(fh)
            good_mask = np.ones(len(X), dtype=bool)
            good_mask[bad["bad_indices"]] = False
            X, y = X[good_mask], y[good_mask]
            print(f"    Rejection file loaded: {bad['n_bad']} epochs dropped")
            bad_ch = [ch for ch in bad.get("bad_channels", []) if ch in EEG_CH_NAMES]

        if bad_ch:
            if len(bad_ch) >= len(EEG_CH_NAMES):
                print(f"    skipped (all channels bad)")
                continue
            ch_idx = [EEG_CH_NAMES.index(ch) for ch in bad_ch]
            X[:, ch_idx, :] = 0.0
            print(f"    Bad channels zeroed: {bad_ch}")

        if CLASSIFIER == "csp_lda":
            X = riemannian_align(X)
            print(f"    {len(X)} epochs (RA)  (L={int((y==0).sum())}  R={int((y==1).sum())})")
        else:
            print(f"    {len(X)} epochs       (L={int((y==0).sum())}  R={int((y==1).sum())})")

        sessions_data.append((ses_name, X, y))
        n_sessions_used += 1

    if len(sessions_data) == 0:
        print(f"\n  No usable sessions for {subject} — skipping.\n")
        continue

    all_results.append((subject, sessions_data))

out_dir = PROJECT_DIR / "outputs" / "results"
out_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# FIXED TEST: Session 19 — shrinking training window
# =========================================================

# Per-subject fixed test session (last session for each subject)
FIXED_TEST_SESSION = {
    "Sub1_data": 19,
    "Sub2_data": 10,
    "Sub3_data": 20,   # update if needed
}
FIXED_TEST_SESSION_DEFAULT = 19   # fallback for subjects not listed above

def _ses_num(name):
    m = re.search(r"Ses(\d+)", name, re.IGNORECASE)
    return int(m.group(1)) if m else None

print("\n\n" + "=" * 65)
print(f"  FIXED TEST — shrinking training window")
print("=" * 65)
print(f"  Classifier: {CLASSIFIER}")
print(f"  {'Subject':<20}  {'Test ses':>8}  {'Train sessions':<20}  {'N train':>8}  {'Acc %':>8}")
print("  " + "-" * 65)

fixed_results = []   # (subject, config_label, config, acc, X_tr, y_tr, X_te, y_te, te_name, ses_lookup)

for subject, sessions_data in all_results:
    ses_lookup = {_ses_num(sd[0]): sd for sd in sessions_data
                  if _ses_num(sd[0]) is not None}

    test_ses = FIXED_TEST_SESSION.get(subject, FIXED_TEST_SESSION_DEFAULT)

    if test_ses not in ses_lookup:
        print(f"  {subject:<20}  session {test_ses} not found — skipping")
        continue

    te_name, X_te, y_te = ses_lookup[test_ses]

    # Build shrinking configs: start from first available session,
    # drop one from the front each iteration; keep at least 2 training sessions
    train_nums = sorted(n for n in ses_lookup if n < test_ses)
    configs = [tuple(train_nums[i:]) for i in range(len(train_nums) - 1)]

    for config in configs:
        # Optionally split last EEGNET_VAL_SESSIONS sessions into a validation set
        use_val = (CLASSIFIER == "eegnet"
                   and EEGNET_USE_VAL
                   and len(config) > EEGNET_VAL_SESSIONS)
        if use_val:
            fit_cfg = config[:-EEGNET_VAL_SESSIONS]
            val_cfg = config[-EEGNET_VAL_SESSIONS:]
            X_tr = np.concatenate([ses_lookup[s][1] for s in fit_cfg], axis=0)
            y_tr = np.concatenate([ses_lookup[s][2] for s in fit_cfg], axis=0)
            X_val = np.concatenate([ses_lookup[s][1] for s in val_cfg], axis=0)
            y_val = np.concatenate([ses_lookup[s][2] for s in val_cfg], axis=0)
        else:
            X_tr = np.concatenate([ses_lookup[s][1] for s in config], axis=0)
            y_tr = np.concatenate([ses_lookup[s][2] for s in config], axis=0)
            X_val = y_val = None

        if CLASSIFIER == "eegnet":
            acc, loss_hist, val_loss_hist = train_eegnet(
                X_tr, y_tr, X_te, y_te, X_val=X_val, y_val=y_val)
        else:
            acc, loss_hist, val_loss_hist = train_csp_lda(X_tr, y_tr, X_te, y_te), None, None
        label = f"{config[0]}-{config[-1]}"
        print(f"  {subject:<20}  {test_ses:>8}  {label:<20}  {len(X_tr):>8}  {acc:>8.2f}")
        fixed_results.append((subject, test_ses, label, config, acc, X_tr, y_tr,
                               X_te, y_te, te_name, ses_lookup, loss_hist, val_loss_hist))

print("=" * 65)

# Save CSV
out_csv_fixed = out_dir / "fixed_test_results.csv"
with open(out_csv_fixed, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["subject", "train_sessions", "n_train_trials",
                     "test_session", "classifier", "acc_%"])
    for subject, test_ses, label, config, acc, X_tr, y_tr, X_te, y_te, te_name, ses_lookup, *_ in fixed_results:
        writer.writerow([subject, label, len(X_tr),
                         test_ses, CLASSIFIER, f"{acc:.2f}"])
print(f"\nFixed-test results saved to:\n  {out_csv_fixed}")

# CSP plots — one per training configuration (csp_lda only)
if CLASSIFIER != "csp_lda":
    print("\nCSP visualization skipped (not applicable for EEGNet).")

for subject, test_ses, label, config, acc, X_tr, y_tr, X_te, y_te, te_name, ses_lookup, *_ \
        in (fixed_results if CLASSIFIER == "csp_lda" else []):

    fig, ax = plt.subplots(figsize=(7, 6))

    # Fit shared CSP on all training data
    n_csp = min(2, X_tr.shape[1] - 1)
    csp = CSP(n_components=n_csp, reg=None, log=True, norm_trace=False)
    csp.fit(X_tr, y_tr)

    # Training sessions
    for k, s in enumerate(config):
        sname, Xs, ys = ses_lookup[s]
        feats = np.nan_to_num(csp.transform(Xs))
        for cls in [0, 1]:
            mask = ys == cls
            ax.scatter(feats[mask, 0], feats[mask, 1],
                       color=TRAIN_COLORS[k % len(TRAIN_COLORS)],
                       marker=CLASS_MARKERS[cls],
                       label=f"{sname} — {CLASS_LABELS[cls]}",
                       alpha=0.65, s=45,
                       edgecolors="black" if cls == 0 else "none",
                       linewidths=0.6)
            if mask.sum() > 0:
                mx, my = feats[mask, 0].mean(), feats[mask, 1].mean()
                ax.scatter(mx, my, marker="*", s=350, color="white",
                           edgecolors=TRAIN_COLORS[k % len(TRAIN_COLORS)],
                           linewidths=2.0, zorder=5)
                ax.scatter(mx, my, marker=CLASS_MARKERS[cls], s=160,
                           color=TRAIN_COLORS[k % len(TRAIN_COLORS)],
                           edgecolors="black", linewidths=1.2, zorder=6)

    # Test session
    feats_te = np.nan_to_num(csp.transform(X_te))
    for cls in [0, 1]:
        mask = y_te == cls
        ax.scatter(feats_te[mask, 0], feats_te[mask, 1],
                   color=TEST_COLOR, marker=CLASS_MARKERS[cls],
                   label=f"{te_name} (test) — {CLASS_LABELS[cls]}",
                   alpha=0.65, s=45,
                   edgecolors="black" if cls == 0 else "none",
                   linewidths=0.6)
        if mask.sum() > 0:
            mx, my = feats_te[mask, 0].mean(), feats_te[mask, 1].mean()
            ax.scatter(mx, my, marker="*", s=350, color="white",
                       edgecolors=TEST_COLOR, linewidths=2.0, zorder=5)
            ax.scatter(mx, my, marker=CLASS_MARKERS[cls], s=160,
                       color=TEST_COLOR, edgecolors="black",
                       linewidths=1.2, zorder=6)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("CSP Component 1 (log-var)", fontsize=10)
    ax.set_ylabel("CSP Component 2 (log-var)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.8)
    fig.suptitle(
        f"{subject}  |  Train: {label}  →  Test: Session {test_ses}\n"
        f"CSP+LDA accuracy: {acc:.1f}%  |  Riemannian Aligned",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()

    csp_path = out_dir / f"{subject}_csp_fixed_test{test_ses}_train{label}.png"
    fig.savefig(csp_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  CSP plot saved: {csp_path}")

# =========================================================
# TRAINING CURVES (EEGNet only)
# =========================================================

if CLASSIFIER == "eegnet":
    for subject in SUBJECTS:
        subj_results = [
            (test_ses, label, acc, loss_hist, val_loss_hist)
            for s, test_ses, label, _, acc, _, _,
                _, _, _, _, loss_hist, val_loss_hist
            in fixed_results if s == subject and loss_hist is not None
        ]

        if not subj_results:
            continue

        has_val = any(v is not None for _, _, _, _, v in subj_results)
        test_ses_subj = subj_results[0][0] if subj_results else "?"
        fig, axes = plt.subplots(1, 2 if has_val else 1,
                                 figsize=(14 if has_val else 10, 5), squeeze=False)
        ax_tr  = axes[0][0]
        ax_val = axes[0][1] if has_val else None
        cmap   = plt.get_cmap("tab20")
        n      = len(subj_results)

        for i, (_, label, acc, loss_hist, val_loss_hist) in enumerate(subj_results):
            color = cmap(i / max(n - 1, 1))
            ax_tr.plot(range(1, len(loss_hist) + 1), loss_hist,
                       color=color, linewidth=1.2,
                       label=f"{label}  ({acc:.1f}%)")
            if has_val and val_loss_hist is not None:
                ax_val.plot(range(1, len(val_loss_hist) + 1), val_loss_hist,
                            color=color, linewidth=1.2,
                            label=f"{label}  ({acc:.1f}%)")

        for ax, title in [(ax_tr, "Training loss"),
                          *(([(ax_val, "Validation loss")] if has_val else []))]:
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel("Cross-entropy loss", fontsize=11)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.legend(fontsize=7, loc="upper right", framealpha=0.8,
                      ncol=max(1, n // 10))
            ax.grid(alpha=0.3)

        val_note = f"  |  last {EEGNET_VAL_SESSIONS} sessions = validation" if has_val else ""
        fig.suptitle(
            f"{subject}  |  EEGNet training curves\n"
            f"Fixed test: Session {test_ses_subj}  —  shrinking training window{val_note}",
            fontsize=11, fontweight="bold"
        )
        fig.tight_layout()

        curves_path = out_dir / f"{subject}_eegnet_training_curves.png"
        fig.savefig(curves_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Training curves saved: {curves_path}")


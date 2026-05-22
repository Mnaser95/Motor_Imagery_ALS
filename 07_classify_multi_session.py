from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy.linalg
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

SUBJECTS  = ["Sub1_data", "Sub2_data", "Sub3_data"]   # list of subject folder names to process

EXCLUDE_SESSIONS = {
    "Sub1_data": {},
    "Sub2_data": {},
    "Sub3_data": {},   # matches MARKED_SESSIONS in 06b
}
EXCLUDE_SESSIONS_DEFAULT = {}

# Plot colours / markers
TRAIN_COLORS  = ["#E53935", "#1565C0", "#F9A825", "#AB47BC", "#FF7043"]   # red, blue, yellow, purple, orange
TEST_COLOR    = "black"
CLASS_MARKERS = {0: "o", 1: "^"}   # circle = MI Left, triangle = MI Right
CLASS_LABELS  = {0: "MI Left", 1: "MI Right"}


# ---- Classifier selection ----
# "csp_lda"          : CSP + shrinkage LDA with Riemannian Alignment
# "weighted_csp_lda" : impedance-weighted CSP + shrinkage LDA (no RA)
# "eegnet"           : EEGNet (PyTorch) with Euclidean Alignment + z-scoring
CLASSIFIER = "csp_lda"

# CSP hyperparameters (used when CLASSIFIER == "csp_lda" or "weighted_csp_lda")
CSP_COMPONENTS = 4   # upper bound; auto-capped to (n_remaining_channels - 1)

# Exponent for impedance quality score: q_i = (1/mean_Z_i)^k  (weighted_csp_lda only)
QUALITY_EXPONENT = 1.1

# EEGNet hyperparameters (only used when CLASSIFIER == "eegnet")
EEGNET_F1       = 8
EEGNET_D        = 2
EEGNET_F2       = 16
EEGNET_KERN_LEN = sfreq // 2   # 150 samples @ 300 Hz
EEGNET_DROPOUT  = 0.25         # reduced from 0.5 — small datasets need less aggressive dropout
EEGNET_EPOCHS   = 200
EEGNET_BATCH    = 16
EEGNET_LR       = 1e-3
EEGNET_WEIGHT_DECAY  = 1e-4    # L2 regularisation in Adam
EEGNET_LR_PATIENCE   = 20      # ReduceLROnPlateau: halve LR after this many stagnant epochs
EEGNET_NOISE_STD         = 0.05    # Gaussian noise augmentation std (0 = disabled)
EEGNET_USE_IMPEDANCE     = True    # True = weight loss by per-epoch impedance quality scores
EEGNET_SEED              = 42      # set to None to disable fixed seed
EEGNET_DEVICE            = "cuda"  # "auto" = cuda if available, else cpu; or force "cuda" / "cpu"
EEGNET_USE_VAL       = False   # True  = split last EEGNET_VAL_SESSIONS sessions for validation
                                # False = all sessions used for training, no validation tracking
EEGNET_VAL_SESSIONS  = 2       # number of trailing sessions held out (only when EEGNET_USE_VAL=True)

# Order of EEG channels in X (must match CH_MAP / MNE pick order)
EEG_CH_NAMES = ["Cz", "CP2", "CP3", "FC2", "FC3"]

# =========================================================
# HELPER: PREPROCESS ONE SESSION → MI EPOCHS (EEG only)
# =========================================================

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

def euclidean_align(X):
    """Whiten each session by arithmetic mean covariance M^{-1/2} (He & Wu 2020)."""
    T    = X.shape[2]
    covs = np.array([x @ x.T / T for x in X])
    M    = covs.mean(axis=0)
    M_invsqrt = _mat_pow(M, -0.5)
    return np.stack([M_invsqrt @ x for x in X], axis=0)

# =========================================================
# WEIGHTED CSP
# =========================================================

class WeightedCSP:
    """CSP with per-epoch impedance-quality-weighted class covariances."""

    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_     = None

    def fit(self, X, y, sample_weight=None):
        n_epochs, n_ch, n_times = X.shape
        if sample_weight is None:
            sample_weight = np.ones(n_epochs)

        covs = []
        for cls in [0, 1]:
            idx = (y == cls)
            w   = sample_weight[idx].copy()
            w  /= w.sum()
            C   = np.zeros((n_ch, n_ch))
            for wi, xi in zip(w, X[idx]):
                C += wi * (xi @ xi.T) / n_times
            covs.append(C)

        evals, evecs = scipy.linalg.eigh(covs[1], covs[0] + covs[1])
        n_low  = self.n_components // 2
        n_high = self.n_components - n_low
        idx_s  = np.argsort(evals)
        sel    = np.concatenate([idx_s[:n_low], idx_s[-n_high:]])
        self.filters_ = evecs[:, sel]
        return self

    def transform(self, X):
        return np.array([np.var(self.filters_.T @ x, axis=1) for x in X])

    def fit_transform(self, X, y, sample_weight=None):
        return self.fit(X, y, sample_weight).transform(X)


def load_impedance_table(z_path):
    if not z_path.exists():
        return None
    try:
        df = pd.read_csv(z_path, comment="#", skipinitialspace=True)
    except Exception:
        return None
    required = ["Time"] + EEG_CHANNELS
    if any(c not in df.columns for c in required):
        return None
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    for col in EEG_CHANNELS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[required].dropna(subset=["Time"]).reset_index(drop=True)


def epoch_quality(z_table, onset_sec, active_ch_cols):
    if z_table is None or not active_ch_cols:
        return 1.0
    idx   = (z_table["Time"] - onset_sec).abs().idxmin()
    z_row = z_table.loc[idx, active_ch_cols].values.astype(float)
    valid = z_row[np.isfinite(z_row) & (z_row > 0)]
    if len(valid) == 0:
        return 1.0
    return float(np.mean(valid)) ** (-QUALITY_EXPONENT)


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


def train_weighted_csp_lda(X_tr, y_tr, X_te, y_te, q_tr):
    n_csp = min(CSP_COMPONENTS, X_tr.shape[1] - 1)
    csp   = WeightedCSP(n_components=n_csp)
    lda   = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    X_tr_feat = np.nan_to_num(csp.fit_transform(X_tr, y_tr, sample_weight=q_tr))
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

    def fit(self, X, y, q=None, X_val=None, y_val=None):
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        Xt = torch.tensor(X[:, np.newaxis], dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        # Normalise quality weights so mean = 1 (preserves effective learning rate)
        if q is not None:
            qt = torch.tensor(q / q.mean(), dtype=torch.float32)
        else:
            qt = torch.ones(len(y), dtype=torch.float32)
        g = torch.Generator()
        if EEGNET_SEED is not None:
            g.manual_seed(EEGNET_SEED)
        loader  = DataLoader(TensorDataset(Xt, yt, qt),
                             batch_size=EEGNET_BATCH, shuffle=True, generator=g)
        opt      = torch.optim.Adam(self.model.parameters(),
                                    lr=EEGNET_LR,
                                    weight_decay=EEGNET_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=EEGNET_LR_PATIENCE)
        loss_fn = self._nn.CrossEntropyLoss(reduction="none")

        has_val = X_val is not None
        if has_val:
            Xv = torch.tensor(X_val[:, np.newaxis], dtype=torch.float32).to(self.device)
            yv = torch.tensor(y_val, dtype=torch.long).to(self.device)

        self.loss_history     = []
        self.val_loss_history = [] if has_val else None
        self.model.train()
        for _ in range(EEGNET_EPOCHS):
            epoch_loss = 0.0
            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(self.device), yb.to(self.device), wb.to(self.device)
                if EEGNET_NOISE_STD > 0:
                    xb = xb + torch.randn_like(xb) * EEGNET_NOISE_STD
                opt.zero_grad()
                loss = (loss_fn(self.model(xb), yb) * wb).mean()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            epoch_loss /= len(loader)
            self.loss_history.append(epoch_loss)
            scheduler.step(epoch_loss)

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_loss = loss_fn(self.model(Xv), yv).mean().item()
                self.val_loss_history.append(val_loss)
                self.model.train()

    def predict(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            Xt   = torch.tensor(X[:, np.newaxis], dtype=torch.float32).to(self.device)
            return self.model(Xt).argmax(dim=1).cpu().numpy()


def train_eegnet(X_tr, y_tr, X_te, y_te, q_tr=None, X_val=None, y_val=None):
    if EEGNET_SEED is not None:
        import random, torch as _torch
        random.seed(EEGNET_SEED)
        np.random.seed(EEGNET_SEED)
        _torch.manual_seed(EEGNET_SEED)
        _torch.cuda.manual_seed_all(EEGNET_SEED)
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark     = False

    mu    = X_tr.mean(axis=(0, 2), keepdims=True)
    sigma = X_tr.std(axis=(0, 2), keepdims=True)
    sigma = np.where(sigma < 1e-10, 1.0, sigma)
    X_tr_z = (X_tr - mu) / sigma
    X_te_z = (X_te - mu) / sigma
    X_val_z = (X_val - mu) / sigma if X_val is not None else None

    net = _EEGNet(X_tr.shape[1], X_tr.shape[2])
    net.fit(X_tr_z, y_tr, q=q_tr, X_val=X_val_z, y_val=y_val)
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
    z_dir        = subject_dir / "Z"
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

        preproc_file = (PROJECT_DIR / "outputs" / "00_manual_epoch_review"
                        / "preprocessed_epochs" / subject / f"{ses_name}_epo.fif")
        if not preproc_file.exists():
            print("    skipped (run 00_manual_epoch_review.py first)")
            continue
        _epo      = mne.read_epochs(preproc_file, verbose=False)
        eeg_picks = mne.pick_types(_epo.info, eeg=True, eog=False, stim=False)
        X         = _epo.get_data(picks=eeg_picks)
        y         = (_epo.events[:, 2] == 9).astype(int)
        onset_sec = _epo.events[:, 0] / sfreq

        rej_file = (PROJECT_DIR / "outputs" / "00_manual_epoch_review" / "epoch_rejection"
                    / subject / f"{ses_name}_bad_epochs.json")
        print(f"    Rejection file: {rej_file}")
        bad_ch: list[str] = []
        if rej_file.exists():
            with open(rej_file) as fh:
                bad = json.load(fh)
            good_mask = np.ones(len(X), dtype=bool)
            good_mask[bad["bad_indices"]] = False
            X, y, onset_sec = X[good_mask], y[good_mask], onset_sec[good_mask]
            print(f"    Rejection file loaded: {bad['n_bad']} epochs dropped  "
                  f"bad_indices={bad['bad_indices']}")
            bad_ch = [ch for ch in bad.get("bad_channels", []) if ch in EEG_CH_NAMES]
        else:
            print("    Rejection file NOT found — all epochs kept")

        active_eeg_cols = [c for c, name in zip(EEG_CHANNELS, EEG_CH_NAMES)
                           if name not in bad_ch]

        if bad_ch:
            if len(bad_ch) >= len(EEG_CH_NAMES):
                print(f"    skipped (all channels bad)")
                continue
            ch_idx = [EEG_CH_NAMES.index(ch) for ch in bad_ch]
            X[:, ch_idx, :] = 0.0
            print(f"    Bad channels zeroed: {bad_ch}")

        # --- Impedance quality scores (weighted_csp_lda and eegnet) ---
        if CLASSIFIER in ("weighted_csp_lda", "eegnet"):
            ses_match = re.search(r"Ses(\d+)", ses_name, re.IGNORECASE)
            z_table   = None
            if ses_match:
                z_path  = z_dir / f"{int(ses_match.group(1))}.csv"
                z_table = load_impedance_table(z_path)
            if z_table is not None:
                q = np.array([epoch_quality(z_table, t, active_eeg_cols)
                               for t in onset_sec])
                print(f"    Impedance quality: min={q.min():.3f}  max={q.max():.3f}")
            else:
                q = np.ones(len(X))
                print("    Impedance file not found — uniform quality scores")
        else:
            q = np.ones(len(X))

        if CLASSIFIER == "csp_lda":
            X = riemannian_align(X)
            print(f"    {len(X)} epochs (RA)  (L={int((y==0).sum())}  R={int((y==1).sum())})")
        elif CLASSIFIER == "eegnet":
            X = euclidean_align(X)
            print(f"    {len(X)} epochs (EA)  (L={int((y==0).sum())}  R={int((y==1).sum())})")
        else:
            print(f"    {len(X)} epochs       (L={int((y==0).sum())}  R={int((y==1).sum())})")

        sessions_data.append((ses_name, X, y, q))
        n_sessions_used += 1

    if len(sessions_data) == 0:
        print(f"\n  No usable sessions for {subject} — skipping.\n")
        continue

    all_results.append((subject, sessions_data))

out_dir = PROJECT_DIR / "outputs" / Path(__file__).stem / CLASSIFIER
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

    te_name, X_te, y_te, q_te = ses_lookup[test_ses]

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
            X_tr  = np.concatenate([ses_lookup[s][1] for s in fit_cfg], axis=0)
            y_tr  = np.concatenate([ses_lookup[s][2] for s in fit_cfg], axis=0)
            X_val = np.concatenate([ses_lookup[s][1] for s in val_cfg], axis=0)
            y_val = np.concatenate([ses_lookup[s][2] for s in val_cfg], axis=0)
            q_tr  = np.concatenate([ses_lookup[s][3] for s in fit_cfg], axis=0)
        else:
            X_tr  = np.concatenate([ses_lookup[s][1] for s in config], axis=0)
            y_tr  = np.concatenate([ses_lookup[s][2] for s in config], axis=0)
            q_tr  = np.concatenate([ses_lookup[s][3] for s in config], axis=0)
            X_val = y_val = None

        if CLASSIFIER == "eegnet":
            acc, loss_hist, val_loss_hist = train_eegnet(
                X_tr, y_tr, X_te, y_te,
                q_tr=q_tr if EEGNET_USE_IMPEDANCE else None,
                X_val=X_val, y_val=y_val)
        elif CLASSIFIER == "weighted_csp_lda":
            acc, loss_hist, val_loss_hist = train_weighted_csp_lda(
                X_tr, y_tr, X_te, y_te, q_tr), None, None
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

# CSP plots — one per training configuration (csp_lda and weighted_csp_lda only)
if CLASSIFIER == "eegnet":
    print("\nCSP visualization skipped (not applicable for EEGNet).")

for subject, test_ses, label, config, acc, X_tr, y_tr, X_te, y_te, te_name, ses_lookup, *_ \
        in (fixed_results if CLASSIFIER in ("csp_lda", "weighted_csp_lda") else []):

    fig, ax = plt.subplots(figsize=(7, 6))

    # Fit shared CSP on all training data
    n_csp = min(2, X_tr.shape[1] - 1)
    csp = CSP(n_components=n_csp, reg=None, log=True, norm_trace=False)
    csp.fit(X_tr, y_tr)

    # Training sessions
    for k, s in enumerate(config):
        sname, Xs, ys, _ = ses_lookup[s]
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
    method_label = ("Riemannian Aligned" if CLASSIFIER == "csp_lda"
                    else "Impedance-Weighted")
    fig.suptitle(
        f"{subject}  |  Train: {label}  →  Test: Session {test_ses}\n"
        f"CSP+LDA accuracy: {acc:.1f}%  |  {method_label}",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()

    csp_path = out_dir / f"{subject}_csp_fixed_test{test_ses}_train{label}.png"
    fig.savefig(csp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
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
        plt.close(fig)
        print(f"Training curves saved: {curves_path}")


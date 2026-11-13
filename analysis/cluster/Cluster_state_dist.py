import glob, os, sys
sys.path.insert(0, r"C:\path\to\project\_libs")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CL = r"response_letter\_workbench\clustering"
STATE_SYS = {"Pre-active-MSTN": ["5", "6"], "Traj-MSTN": ["7", "8"],
             "Traj-MSTN-ActRII": ["9", "10", "11", "12"],
             "APO-MSTN": ["1", "2"], "APO-MSTN-ActRII": ["4"]}
STATE = {}
for line in open(os.path.join(CL, "..", "landmarks", "state_matches.csv"), encoding="utf-8"):
    f = line.strip().split(",")
    if f[0].startswith("state") and f[1] != "angle1":
        STATE[f[0].split()[1]] = (float(f[1]), float(f[2]), float(f[3]))

def load_cv(s):
    proj = {}
    for f in sorted(glob.glob(r"_work\pca\proj_%s__*.dat" % s)):
        rep = os.path.basename(f).split("__")[1].replace(".dat", "")
        proj[rep] = np.loadtxt(f)
    return proj

rows = []
for s, states in STATE_SYS.items():
    proj = load_cv(s)
    A = np.vstack(list(proj.values()))
    cv_all = A[:, 11:14]
    mu, sd = cv_all.mean(0), cv_all.std(0)
    Zall = (cv_all - mu) / sd

    z = np.load(os.path.join(CL, "%s_frames.npz" % s), allow_pickle=True)
    lab, time, rep = z["labels"], z["time"], z["replica"]
    samp = []
    for L, t, r in zip(lab, time, rep):
        a = proj.get(str(r))
        if a is None:
            continue
        i = int(np.abs(a[:, 0] - float(t)).argmin())
        if abs(a[i, 0] - float(t)) <= 0.6:
            samp.append((int(L), a[i, 11:14]))
    samp = np.array([(L, *c) for L, c in samp])
    labels = samp[:, 0].astype(int)
    Cz = (samp[:, 1:4] - mu) / sd
    cent = {}
    for L in np.unique(labels):
        m = labels == L
        if m.sum() >= 5:
            cent[L] = Cz[m].mean(0)
    if not cent:
        continue
    cids = np.array(list(cent.keys()))
    C = np.array(list(cent.values()))
    # assign every 1 ns frame to nearest centroid
    d = ((Zall[:, None, :] - C[None]) ** 2).sum(2)
    assign = cids[d.argmin(1)]

    for st in states:
        ang, d1, d2 = STATE[st]
        best = min(cent.items(), key=lambda kv: ((kv[1][0] * sd[0] + mu[0] - ang) / 20) ** 2
                   + ((kv[1][1] * sd[1] + mu[1] - d1) / 3) ** 2
                   + ((kv[1][2] * sd[2] + mu[2] - d2) / 3) ** 2)
        k = best[0]
        sel = cv_all[assign == k]
        rows.append(dict(state=st, system=s, cluster=int(k), n=len(sel), cv=sel,
                         rep=(ang, d1, d2), mu=mu, sd=sd))
        print("state %-2s %-18s cluster %3d  n=%4d" % (st, s, k, len(sel)))

# figure
order = sorted(rows, key=lambda r: int(r["state"]))
fig, axes = plt.subplots(len(order), 3, figsize=(11, 1.35 * len(order) + 1.2), dpi=160)
cols = ["Angle-1 (deg)", "Distance-1 (A)", "Distance-2 (A)"]
for i, r in enumerate(order):
    for j in range(3):
        ax = axes[i, j]
        v = r["cv"][:, j]
        ax.hist(v, bins=28, color="#4C78A8", alpha=0.85, density=True)
        ax.axvline(r["rep"][j], color="crimson", ls="--", lw=1.6)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_title(cols[j], fontsize=9)
        if j == 0:
            ax.set_ylabel("state %s\n(%s, cl %d, n=%d)" % (r["state"], r["system"].replace("-", "\u2011"), r["cluster"], r["n"]),
                          fontsize=6.5, rotation=0, ha="right", va="center", labelpad=26)
fig.suptitle("Within-cluster distributions of the three collective variables (frames assigned to the nearest cluster centroid)\n"
             "red dashed line: the representative frame of the state", fontsize=9)
fig.tight_layout(rect=[0.10, 0.01, 0.99, 0.94])
out = r"_tmp3\Fig_S19_cluster_distributions.png"
fig.savefig(out, dpi=160, facecolor="white")
print("saved", out)

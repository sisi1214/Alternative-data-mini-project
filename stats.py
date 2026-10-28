"""Stats for: Luk Fook SSSG ~ C&SD jewellery retail yoy (+ gold yoy). Small-n by construction: everything here is descriptive/indicative.
PRIMARY (declared before looking at results): lag 0, all quarters with complete data, incremental value of C&SD over gold:
   nested F-test of SSSG ~ gold + C&SD vs SSSG ~ gold, with a Freedman-Lane permutation p-value. All other specs are sensitivity checks, unadjusted."""
import pandas as pd, numpy as np, json
from scipy import stats as st
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = '/mnt/user-data/outputs'
P = pd.read_csv(f'{OUT}/quarterly_panel.csv', parse_dates=['start'])
P = P.rename(columns={'sssg_pct': 'S', 'csd_yoy_3m': 'C', 'csd_yoy_2m': 'C2', 'gold_hkd_yoy_pct': 'G'})
P['prevC'] = P.C.shift(1); P['prevG'] = P.G.shift(1); P['prev_contam'] = P.covid_contaminated.shift(1).fillna(True).astype(bool)
P['overseas'] = P.overseas_tag_row10.eq('Overseas')
RNG = np.random.default_rng(12345); B = 20000

def ols(y, X):
    y = np.asarray(y, float); X = np.column_stack([np.ones(len(y)), np.asarray(X, float).reshape(len(y), -1)])
    n, k = X.shape; XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b
    rss = e @ e; tss = ((y - y.mean()) ** 2).sum(); s2 = rss / (n - k); se = np.sqrt(np.diag(s2 * XtXi))
    h = np.einsum('ij,jk,ik->i', X, XtXi, X); u = e / (1 - h)
    se3 = np.sqrt(np.diag(XtXi @ (X.T * u ** 2) @ X @ XtXi))
    t = b / se; p = 2 * st.t.sf(abs(t), n - k); p3 = 2 * st.t.sf(abs(b / se3), n - k)
    return dict(b=b, se=se, p=p, p_hc3=p3, rss=rss, r2=1 - rss / tss, adj=1 - (rss / (n - k)) / (tss / (n - 1)), n=n, k=k, e=e, h=h, fit=X @ b,
                dw=(np.diff(e) ** 2).sum() / (e @ e))

def loo(y, X):
    y = np.asarray(y, float); X = np.asarray(X, float).reshape(len(y), -1); pr = np.zeros(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i; f = ols(y[m], X[m]); pr[i] = f['b'][0] + X[i] @ f['b'][1:]
    return pr

def incr_test(y, Xr, Xf):
    r, f = ols(y, Xr), ols(y, Xf); q = f['k'] - r['k']; dfd = f['n'] - f['k']
    F = ((r['rss'] - f['rss']) / q) / (f['rss'] / dfd); pF = st.f.sf(F, q, dfd)
    fit_r, er = r['fit'], r['e']; cnt = 0
    for _ in range(B):
        ys = fit_r + RNG.permutation(er); rs, fs = ols(ys, Xr), ols(ys, Xf)
        Fs = ((rs['rss'] - fs['rss']) / q) / (fs['rss'] / dfd); cnt += Fs >= F - 1e-12
    return dict(F=F, p_F=pF, p_perm=(1 + cnt) / (B + 1), d_r2=f['r2'] - r['r2'], d_adj=f['adj'] - r['adj'])

def corr(a, b):
    n = len(a); r, p = st.pearsonr(a, b); rho, ps = st.spearmanr(a, b)
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3) if n > 3 else np.nan
    return dict(n=n, pearson=r, p_pearson=p, ci_lo=np.tanh(z - 1.96 * se), ci_hi=np.tanh(z + 1.96 * se), spearman=rho, p_spearman=ps)

def sample(lag, covid_excl, overseas_excl=False):
    d = P.copy()
    d['x_C'] = d.C if lag == 0 else d.prevC; d['x_G'] = d.G if lag == 0 else d.prevG
    d['bad'] = d.covid_contaminated | (d.prev_contam if lag == 1 else False)
    d = d.dropna(subset=['S', 'x_C', 'x_G'])
    if covid_excl: d = d[~d.bad]
    if overseas_excl: d = d[~d.overseas]
    return d.reset_index(drop=True)

rows, loo_rows = [], []
SPECS = [('lag0_all (PRIMARY)', 0, False, False), ('lag0_exCOVID', 0, True, False), ('lag1_all', 1, False, False),
         ('lag1_exCOVID', 1, True, False), ('lag0_all_exOverseasTag', 0, False, True)]
for name, lag, cx, ox in SPECS:
    d = sample(lag, cx, ox); y = d.S.values; C = d.x_C.values; G = d.x_G.values
    c1, c2 = corr(y, C), corr(y, G)
    mC, mG, mCG = ols(y, C), ols(y, G), ols(y, np.column_stack([G, C]))
    inc_C = incr_test(y, G, np.column_stack([G, C])); inc_G = incr_test(y, C, np.column_stack([G, C]))
    rm = {k: np.sqrt(np.mean((y - loo(y, X)) ** 2)) for k, X in [('C', C), ('G', G), ('G+C', np.column_stack([G, C]))]}
    r_cg = np.corrcoef(C, G)[0, 1]
    rows.append(dict(spec=name, n=len(d), quarters=' '.join(d.fq_label), r_S_C=c1['pearson'], r_S_C_ci=f"[{c1['ci_lo']:.2f},{c1['ci_hi']:.2f}]", p_S_C=c1['p_pearson'],
        rho_S_C=c1['spearman'], r_S_G=c2['pearson'], p_S_G=c2['p_pearson'], rho_S_G=c2['spearman'], r_C_G=r_cg, vif=1 / (1 - r_cg ** 2),
        R2_C=mC['r2'], R2_G=mG['r2'], R2_GC=mCG['r2'], adjR2_G=mG['adj'], adjR2_GC=mCG['adj'],
        beta_C_in_GC=mCG['b'][2], se_C=mCG['se'][2], p_C_t=mCG['p'][2], p_C_hc3=mCG['p_hc3'][2], beta_G_in_GC=mCG['b'][1], p_G_t=mCG['p'][1],
        incrC_dR2=inc_C['d_r2'], incrC_F=inc_C['F'], incrC_p_F=inc_C['p_F'], incrC_p_perm=inc_C['p_perm'],
        incrG_dR2=inc_G['d_r2'], incrG_p_F=inc_G['p_F'], incrG_p_perm=inc_G['p_perm'],
        loocv_rmse_C=rm['C'], loocv_rmse_G=rm['G'], loocv_rmse_GC=rm['G+C'], dw_GC=mCG['dw']))
    # leave-one-out influence on the C&SD coefficient in S ~ G + C, and on the incremental R2
    for i in range(len(d)):
        m = np.arange(len(d)) != i; yy, Xf = y[m], np.column_stack([G, C])[m]
        f = ols(yy, Xf); rr = ols(yy, G[m])
        loo_rows.append(dict(spec=name, omitted=d.fq_label[i], n_fit=int(m.sum()), beta_C=f['b'][2], se_C=f['se'][2], p_C=f['p'][2],
                             beta_G=f['b'][1], dR2_C=f['r2'] - rr['r2'], r_S_C=np.corrcoef(yy, C[m])[0, 1], cook_like_hat=ols(y, np.column_stack([G, C]))['h'][i]))
R = pd.DataFrame(rows); L = pd.DataFrame(loo_rows)
R.to_csv(f'{OUT}/regression_results.csv', index=False); L.to_csv(f'{OUT}/leave_one_out_results.csv', index=False)

# LOO summary per spec
summ = L.groupby('spec').agg(beta_C_min=('beta_C', 'min'), beta_C_max=('beta_C', 'max'), minp_C=('p_C', 'min'), maxp_C=('p_C', 'max'), dR2_min=('dR2_C', 'min'), dR2_max=('dR2_C', 'max'),
                              rS_C_min=('r_S_C', 'min'), rS_C_max=('r_S_C', 'max')).reset_index()
summ['sign_flips_beta_C'] = L.groupby('spec').apply(lambda g: int((np.sign(g.beta_C) != np.sign(R.loc[R.spec == g.name, 'beta_C_in_GC'].iloc[0])).sum()), include_groups=False).values
summ.to_csv(f'{OUT}/leave_one_out_summary.csv', index=False)

# ---------------- NOWCAST: first two months only ----------------
N = P.dropna(subset=['S', 'C2']).sort_values('start').reset_index(drop=True)           # includes 2Q2026 (Jul-Aug only)
Ng = N.dropna(subset=['G']).reset_index(drop=True)                                      # excludes 2Q2026 (no Q3-26 gold average sourced)
def exp_window(d, cols, min_train=5):
    out = []
    for k in range(min_train, len(d)):
        tr, te = d.iloc[:k], d.iloc[k]
        f = ols(tr.S, tr[cols].values) if cols else None
        pred = f['b'][0] + te[cols].values.astype(float) @ f['b'][1:] if cols else tr.S.mean()
        out.append((te.fq_label, te.S, pred))
    return pd.DataFrame(out, columns=['fq', 'actual', 'pred'])
def metrics(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return dict(n=len(a), rmse=np.sqrt(np.mean((a - p) ** 2)), mae=np.mean(np.abs(a - p)), sign_hit=np.mean((a > 0) == (p > 0)), corr=np.corrcoef(a, p)[0, 1] if len(a) > 2 else np.nan)
nrows = []; preds = {}
def add(label, d, a, p, kind):
    m = metrics(a, p); m.update(model=label, scheme=kind); nrows.append(m); preds[(label, kind)] = (list(d), np.asarray(a), np.asarray(p))
# common sample = Ng (10 quarters, excl. 2Q2026)
y = Ng.S.values
for label, cols in [('S~C2m', ['C2']), ('S~C2m+Gold(qtr avg, mild look-ahead)', ['C2', 'G']), ('S~Gold only', ['G'])]:
    add(label, Ng.fq_label, y, loo(y, Ng[cols].values), 'LOO')
    ew = exp_window(Ng, cols); add(label, ew.fq, ew.actual, ew.pred, 'ExpandingWindow')
add('Pass-through S=C2m (no fit)', Ng.fq_label, y, Ng.C2.values, 'LOO')
ew = exp_window(Ng, []); add('Training-mean benchmark', ew.fq, ew.actual, ew.pred, 'ExpandingWindow')
lastS = Ng.S.shift(1); mk = lastS.notna()
add('Last available SSSG (naive)', Ng.fq_label[mk], y[mk], lastS[mk], 'LOO')
Nw = pd.DataFrame(nrows)
# fair comparison for expanding window: benchmarks on same test set -> already identical (Ng rows k>=5)
# live 2Q2026 nowcast: fit on the 10 complete quarters, predict from Jul-Aug C&SD only
tr = Ng; f = ols(tr.S, tr[['C2']].values); live = N[N.fq_label == '2Q2026'].iloc[0]
pred_live = f['b'][0] + f['b'][1] * live.C2
trx = Ng[~Ng.covid_contaminated]; fx = ols(trx.S, trx[['C2']].values); pred_live_x = fx['b'][0] + fx['b'][1] * live.C2
live_res = dict(C2_JulAug=live.C2, actual_SSSG=live.S, pred_all=pred_live, pred_exCOVID=pred_live_x, passthrough=live.C2, last_available=Ng.S.iloc[-1], train_mean=Ng.S.mean(),
                slope_all=f['b'][1], int_all=f['b'][0])
Nw.to_csv(f'{OUT}/nowcast_results.csv', index=False)
pd.DataFrame([live_res]).to_csv(f'{OUT}/nowcast_2Q2026_live.csv', index=False)
# how close is the 2-month signal to the 3-month signal?
c23 = P.dropna(subset=['C', 'C2']); cc = dict(corr_C2_C=np.corrcoef(c23.C, c23.C2)[0, 1], mae_C2_vs_C=np.mean(np.abs(c23.C - c23.C2)), max_gap=np.max(np.abs(c23.C - c23.C2)), n=len(c23))
corr_nowcast = corr(N.S.values, N.C2.values)

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.float_format', lambda v: f'{v:,.3f}')
print(R.drop(columns=['quarters']).T.to_string()); print(summ.to_string()); print(Nw.to_string()); print(live_res); print(cc); print('corr S vs C2m (n=%d):' % len(N), corr_nowcast)
print(L[L.spec.str.startswith('lag0_all (')].round(3).to_string())
json.dump(dict(live=live_res, cc=cc, corr_nowcast=corr_nowcast), open('/home/claude/nowcast_misc.json', 'w'), default=float)

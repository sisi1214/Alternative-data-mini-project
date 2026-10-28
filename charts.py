import pandas as pd, numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('/home/claude/stats.py').read().split('rows, loo_rows = [], []')[0])  # reuse helpers + panel (no re-run of the heavy tests)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
BLUE, ORG, GRY, RED = '#1f5fa8', '#d9731a', '#8c8c8c', '#b3261e'
Pl = P[P.start >= '2023-04-01'].reset_index(drop=True)   # window where SSSG exists
x = np.arange(len(Pl))

# Fig 1: time series
fig, ax = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})
for i, r in Pl.iterrows():
    if r.covid_contaminated: ax[0].axvspan(i - .5, i + .5, color='#eeeeee', zorder=0); ax[1].axvspan(i - .5, i + .5, color='#eeeeee', zorder=0)
ax[0].plot(x, Pl.C, '-o', color=BLUE, label='C&SD jewellery retail sales, yoy (3 months)', ms=4)
ax[0].plot(x, Pl.C2, ':', color=BLUE, alpha=.8, label='C&SD, first two months only (nowcast input)')
ok = Pl.S.notna(); ax[0].plot(x[ok], Pl.S[ok], 's', color=ORG, label='Luk Fook SSSG (as supplied; NA/blank quarters not plotted)', ms=6)
for i in x[Pl.overseas.values]: ax[0].plot(i, Pl.S[i], 's', mfc='none', mec=RED, ms=11)
ax[0].plot([], [], 's', mfc='none', mec=RED, label='SSSG cell carries an "Overseas" tag (unresolved)')
ax[0].axhline(0, color=GRY, lw=.6); ax[0].set_ylabel('yoy, %'); ax[0].legend(frameon=False, fontsize=7.5, loc='upper left')
ax[0].set_title('Luk Fook SSSG vs C&SD jewellery retail sales (HK), by fiscal quarter (Apr-Mar FY)', loc='left', fontsize=10)
ax[1].bar(x, Pl.G, color='#c9a227', width=.6); ax[1].set_ylabel('Gold, yoy %\n(HKD ~ USD; fx const.)'); ax[1].axhline(0, color=GRY, lw=.6)
ax[1].set_xticks(x); ax[1].set_xticklabels(Pl.fq_label + '\n' + Pl.start.dt.strftime('%b') + '–' + (Pl.start + pd.DateOffset(months=2)).dt.strftime('%b'), fontsize=6.5, rotation=0)
ax[1].text(len(x) - 1, 3, 'no Q3-26\ngold avg', ha='center', fontsize=6.5, color=GRY)
fig.text(.01, .005, 'Grey = quarter or its prior-year base touches the COVID reopening (<= Feb-2023). 2Q2026 C&SD = Jul-Aug only (Aug provisional).', fontsize=6.5, color=GRY)
plt.tight_layout(rect=(0, .02, 1, 1)); plt.savefig(f'{OUT}/fig1_timeseries.png', dpi=170); plt.close()

# Fig 2: scatters + added-variable plot (lag 0, primary sample)
d = sample(0, False); y = d.S.values; C = d.x_C.values; G = d.x_G.values
fig, ax = plt.subplots(1, 3, figsize=(11, 3.7))
def sc(a, xv, yv, xl, yl, ttl):
    a.scatter(xv, yv, c=[RED if c else BLUE for c in d.covid_contaminated], s=28)
    for xi, yi, l in zip(xv, yv, d.fq_label): a.annotate(l, (xi, yi), fontsize=6.5, xytext=(3, 3), textcoords='offset points')
    f = ols(yv, xv); xs = np.linspace(min(xv), max(xv), 20); a.plot(xs, f['b'][0] + f['b'][1] * xs, color=GRY, lw=1)
    a.set_xlabel(xl); a.set_ylabel(yl); a.set_title(ttl, fontsize=8.5, loc='left'); a.axhline(0, color='#dddddd', lw=.6); a.axvline(0, color='#dddddd', lw=.6)
    return f
f1 = sc(ax[0], C, y, 'C&SD jewellery yoy, %', 'Luk Fook SSSG, %', f'SSSG vs C&SD: r={np.corrcoef(C, y)[0, 1]:.2f}, n={len(y)}')
f2 = sc(ax[1], G, y, 'Gold yoy, %', 'Luk Fook SSSG, %', f'SSSG vs gold: r={np.corrcoef(G, y)[0, 1]:.2f}')
ry = ols(y, G)['e']; rc = ols(C, G)['e']
sc(ax[2], rc, ry, 'C&SD yoy | gold (residual)', 'SSSG | gold (residual)', 'Added-variable plot: C&SD beyond gold')
fig.text(.01, .005, 'Red = COVID-contaminated quarter (own or base window <= Feb-2023). Lag 0, all complete quarters. Points are not independent (serial regime swings).', fontsize=6.5, color=GRY)
plt.tight_layout(rect=(0, .03, 1, 1)); plt.savefig(f'{OUT}/fig2_scatter_added_variable.png', dpi=170); plt.close()

# Fig 3: leave-one-out
Lr = pd.read_csv(f'{OUT}/leave_one_out_results.csv'); Rr = pd.read_csv(f'{OUT}/regression_results.csv')
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'width_ratios': [1.35, 1]})
for spec, col, off in [('lag0_all (PRIMARY)', BLUE, -.12), ('lag1_all', ORG, .12)]:
    g = Lr[Lr.spec == spec]; full = Rr[Rr.spec == spec].beta_C_in_GC.iloc[0]
    xs = np.arange(len(g)) + off
    ax[0].errorbar(xs, g.beta_C, yerr=1.96 * g.se_C, fmt='o', color=col, ms=4, capsize=2, label=f'{spec}: full-sample beta={full:.2f}')
    ax[0].axhline(full, color=col, lw=.7, ls='--')
ax[0].axhline(0, color=GRY, lw=.7); ax[0].set_xticks(range(len(Lr[Lr.spec == 'lag1_all']))); ax[0].set_xticklabels(Lr[Lr.spec == 'lag1_all'].omitted, rotation=45, fontsize=7, ha='right')
ax[0].set_ylabel('beta on C&SD yoy (SSSG ~ gold + C&SD)'); ax[0].legend(frameon=False, fontsize=7); ax[0].set_title('Leave-one-out: beta (+/-1.96 se) when each quarter is omitted\n(x-axis labels = omitted quarter; lag0 has no 2Q2026)', fontsize=8.5, loc='left')
specs = ['lag0_all (PRIMARY)', 'lag0_exCOVID', 'lag1_all', 'lag1_exCOVID']; w = .26; xx = np.arange(len(specs))
for j, (c, lab, col) in enumerate([('loocv_rmse_G', 'gold only', '#c9a227'), ('loocv_rmse_C', 'C&SD only', BLUE), ('loocv_rmse_GC', 'gold + C&SD', ORG)]):
    ax[1].bar(xx + (j - 1) * w, [Rr[Rr.spec == s][c].iloc[0] for s in specs], w, label=lab, color=col)
ax[1].set_xticks(xx); ax[1].set_xticklabels([s.replace(' (PRIMARY)', '*').replace('_', '\n') for s in specs], fontsize=7); ax[1].set_ylabel('Leave-one-out CV RMSE (SSSG pp)'); ax[1].legend(frameon=False, fontsize=7)
ax[1].set_title('Out-of-sample error (LOO-CV), n = 8-11', fontsize=8.5, loc='left')
plt.tight_layout(); plt.savefig(f'{OUT}/fig3_leave_one_out.png', dpi=170); plt.close()

# Fig 4: nowcast
N = P.dropna(subset=['S','C2']).sort_values('start').reset_index(drop=True)
Ng2 = N.dropna(subset=['G']).reset_index(drop=True); y = Ng2.S.values
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'width_ratios': [1.5, 1]})
pa = loo(y, Ng2[['C2']].values); pg = loo(y, Ng2[['G']].values); xs = np.arange(len(Ng2))
ax[0].plot(xs, y, 's-', color=ORG, label='Reported SSSG'); ax[0].plot(xs, pa, 'o--', color=BLUE, label='LOO nowcast: C&SD first-2-months')
ax[0].plot(xs, pg, 'x:', color='#c9a227', label='LOO nowcast: gold only')
lv = pd.read_csv(f'{OUT}/nowcast_2Q2026_live.csv').iloc[0]; xl = len(Ng2)
ax[0].plot(xl, lv.actual_SSSG, 's', color=ORG); ax[0].plot(xl, lv.pred_all, 'o', color=BLUE); ax[0].plot(xl, lv.passthrough, 'D', mfc='none', color=BLUE)
ax[0].annotate('2Q2026 (live):\nactual %.0f%%, model %.0f%%,\npass-through %.0f%%' % (lv.actual_SSSG, lv.pred_all, lv.passthrough), (xl, lv.pred_all), xytext=(-60, 55), textcoords='offset points', fontsize=7, arrowprops=dict(arrowstyle='-', color=GRY))
ax[0].set_xticks(list(xs) + [xl]); ax[0].set_xticklabels(list(Ng2.fq_label) + ['2Q2026'], rotation=45, fontsize=7, ha='right'); ax[0].axhline(0, color=GRY, lw=.6)
ax[0].set_ylabel('SSSG, %'); ax[0].legend(frameon=False, fontsize=7, loc='lower right'); ax[0].set_title('Nowcast using only months 1-2 of each quarter (LOO, n=10)', fontsize=8.5, loc='left')
Nw = pd.read_csv(f'{OUT}/nowcast_results.csv'); e = Nw[Nw.scheme == 'ExpandingWindow']; l = Nw[Nw.scheme == 'LOO']
lab = ['S~C2m', 'S~Gold only', 'Training-mean benchmark']; vals = [e[e.model == k].rmse.iloc[0] for k in lab]
ax[1].bar(range(3), vals, color=[BLUE, '#c9a227', GRY]); ax[1].set_xticks(range(3)); ax[1].set_xticklabels(['C&SD\n2-month', 'Gold\nonly', 'Training\nmean'], fontsize=8)
ax[1].set_ylabel('RMSE, pp'); ax[1].set_title('Expanding-window RMSE (only 5 test quarters)', fontsize=8.5, loc='left')
for i, v in enumerate(vals): ax[1].text(i, v + .5, f'{v:.1f}', ha='center', fontsize=8)
plt.tight_layout(); plt.savefig(f'{OUT}/fig4_nowcast.png', dpi=170); plt.close()
print('charts ok')

"""
app.py  --  S&P500 급등종목 스크리너 (Streamlit)

파이프라인:
  1) S&P500 일봉 수집·캐시  ->  data/raw/*.parquet
  2) 시점정합(point-in-time) 파생변수 패널 생성
  3) 워크포워드 LightGBM 로 '이후 H거래일 내 종가 +T% 급등' 확률 예측
  4) 통계검증: 날짜블록 부트스트랩 / 동일자 랜덤픽 순열검정 / Newey-West t / 연도 부호검정
  5) 전체 재학습 후 '오늘자' 급등확률 상위 종목 스코어링

실행:  streamlit run app.py
"""
import os
import time
import json
import glob
import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb
from scipy import stats

import FinanceDataReader as fdr

# ----------------------------------------------------------------------------
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
START = "2010-01-01"
MIN_DOLLAR_VOL = 5e6
META_COLS = {"symbol", "Date", "close", "label", "fwd_max_ret", "trade_ret", "hit_day"}
os.makedirs(RAW_DIR, exist_ok=True)

st.set_page_config(page_title="S&P500 급등종목 스크리너", layout="wide", page_icon="📈")

LGB_PARAMS = dict(
    objective="binary", n_estimators=600, learning_rate=0.02, num_leaves=63,
    min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, n_jobs=-1, verbose=-1,
)

WALKFWD_FOLDS = [
    ("2017-12-31", "2018-01-01", "2019-06-30"),
    ("2019-06-30", "2019-07-01", "2020-12-31"),
    ("2020-12-31", "2021-01-01", "2022-06-30"),
    ("2022-06-30", "2022-07-01", "2023-12-31"),
    ("2023-12-31", "2024-01-01", "2027-12-31"),
]


# ============================ 1. 데이터 수집 =================================
@st.cache_data(show_spinner=False)
def load_listing():
    path = os.path.join(DATA_DIR, "sp500_listing.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    lst = fdr.StockListing("S&P500")
    lst.to_parquet(path)
    return lst


def _download_symbol(symbol):
    fp = os.path.join(RAW_DIR, f"{symbol.replace('/', '-')}.parquet")
    if os.path.exists(fp):
        return "cached"
    for attempt in range(3):
        try:
            df = fdr.DataReader(symbol, START).dropna()
            if df is None or len(df) < 250:
                return "short"
            df.to_parquet(fp)
            return "ok"
        except Exception:
            time.sleep(1.0)
    return "err"


def download_universe(progress=None):
    """S&P500 전체 일봉 수집. 이미 캐시된 종목은 건너뜀."""
    lst = load_listing()
    syms = list(lst["Symbol"])
    # 지수
    idx_path = os.path.join(DATA_DIR, "index_sp500.parquet")
    if not os.path.exists(idx_path):
        for tkr in ("US500", "S&P500", "^GSPC"):
            try:
                idx = fdr.DataReader(tkr, START).dropna()
                if len(idx) > 250:
                    idx.to_parquet(idx_path)
                    break
            except Exception:
                pass
    done = 0
    for i, s in enumerate(syms):
        _download_symbol(s)
        done += 1
        if progress is not None:
            progress.progress(done / len(syms), text=f"수집 {done}/{len(syms)}  ({s})")
    return count_cached()


def count_cached():
    return len(glob.glob(os.path.join(RAW_DIR, "*.parquet")))


# ============================ 2. 파생변수 패널 ==============================
def _rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _streak(sign_series):
    s = np.sign(sign_series.fillna(0).values)
    out = np.zeros(len(s))
    run = 0
    prev = 0
    for i, v in enumerate(s):
        if v == prev and v != 0:
            run += v
        elif v != 0:
            run = v
        else:
            run = 0
        out[i] = run
        prev = v if v != 0 else prev
    return pd.Series(out, index=sign_series.index)


def _features_one(symbol, df, horizon, target):
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    ret1 = c.pct_change()
    f = pd.DataFrame(index=df.index)

    for k in (1, 3, 5, 10, 20, 60, 120):
        f[f"ret_{k}"] = c.pct_change(k)
    f["price_accel"] = c.pct_change(5) - c.pct_change(5).shift(5)
    f["ret_5_z"] = (c.pct_change(5) - c.pct_change(5).rolling(120).mean()) / c.pct_change(5).rolling(120).std()

    for k in (10, 20, 50, 200):
        f[f"close_sma{k}"] = c / c.rolling(k).mean() - 1
    f["sma20_slope5"] = c.rolling(20).mean() / c.rolling(20).mean().shift(5) - 1
    f["sma50_slope10"] = c.rolling(50).mean() / c.rolling(50).mean().shift(10) - 1
    f["sma20_over_sma50"] = c.rolling(20).mean() / c.rolling(50).mean() - 1

    f["vol_20"] = ret1.rolling(20).std()
    f["vol_60"] = ret1.rolling(60).std()
    f["vol_ratio"] = ret1.rolling(20).std() / ret1.rolling(60).std()
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct_14"] = tr.rolling(14).mean() / c
    std20 = c.rolling(20).std()
    f["bb_pos_20"] = (c - c.rolling(20).mean()) / (2 * std20)
    f["hilo_range_20"] = (h.rolling(20).max() - l.rolling(20).min()) / c

    f["rsi_14"] = _rsi(c, 14)
    f["rsi_2"] = _rsi(c, 2)

    f["dist_52w_high"] = c / c.rolling(252).max() - 1
    f["dist_52w_low"] = c / c.rolling(252).min() - 1
    f["drawdown_20"] = c / c.rolling(20).max() - 1
    f["drawdown_60"] = c / c.rolling(60).max() - 1
    f["new_high_20"] = (c >= c.rolling(20).max()).astype(int)
    f["updown_streak"] = _streak(ret1)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

    f["vol_ratio_20"] = v / v.rolling(20).mean()
    f["vol_ratio_60"] = v / v.rolling(60).mean()
    f["obv_slope_10"] = (np.sign(ret1).fillna(0) * v).cumsum().pct_change(10)
    dollar_vol = (c * v).rolling(20).mean()
    f["log_dollar_vol"] = np.log(dollar_vol.replace(0, np.nan))

    f["gap_1"] = o / c.shift() - 1
    f["intraday_1"] = c / o - 1

    close_np = c.values
    n = len(close_np)
    fwd_max_ret = np.full(n, np.nan)
    trade_ret = np.full(n, np.nan)
    hit_day = np.full(n, np.nan)
    for i in range(n):
        end = min(i + horizon, n - 1)
        if end <= i:
            break
        window = close_np[i + 1:end + 1]
        entry = close_np[i]
        fwd_max_ret[i] = window.max() / entry - 1
        tgt = entry * (1 + target)
        hit = np.where(window >= tgt)[0]
        if len(hit):
            trade_ret[i] = window[hit[0]] / entry - 1
            hit_day[i] = hit[0] + 1
        else:
            trade_ret[i] = window[-1] / entry - 1
            hit_day[i] = len(window)
    f["fwd_max_ret"] = fwd_max_ret
    f["label"] = (fwd_max_ret >= target).astype(float)
    f["trade_ret"] = trade_ret
    f["hit_day"] = hit_day
    if n > horizon:
        f.iloc[n - horizon:, f.columns.get_indexer(["label", "fwd_max_ret", "trade_ret", "hit_day"])] = np.nan

    f["symbol"] = symbol
    f["close"] = c
    f["_dollar_vol"] = dollar_vol
    return f.reset_index().rename(columns={f.index.name or "index": "Date"})


@st.cache_data(show_spinner=False)
def build_panel(target: float, horizon: int, _raw_count: int):
    """파라미터별 패널 캐시. 디스크에도 저장."""
    tag = f"T{int(round(target*100))}_H{horizon}"
    out = os.path.join(DATA_DIR, f"panel_{tag}.parquet")
    if os.path.exists(out):
        return pd.read_parquet(out)

    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.parquet")))
    parts = []
    prog = st.progress(0.0, text="파생변수 생성 중...")
    for j, fp in enumerate(files):
        sym = os.path.splitext(os.path.basename(fp))[0]
        try:
            df = pd.read_parquet(fp)
            if len(df) >= 300:
                parts.append(_features_one(sym, df, horizon, target))
        except Exception:
            pass
        if (j + 1) % 20 == 0:
            prog.progress((j + 1) / len(files), text=f"파생변수 {j+1}/{len(files)}")
    prog.empty()
    panel = pd.concat(parts, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])

    idx_path = os.path.join(DATA_DIR, "index_sp500.parquet")
    if os.path.exists(idx_path):
        idx = pd.read_parquet(idx_path).sort_index()
        ic = idx["Close"]
        mkt = pd.DataFrame(index=idx.index)
        mkt["mkt_ret_5"] = ic.pct_change(5)
        mkt["mkt_ret_20"] = ic.pct_change(20)
        mkt["mkt_above_sma200"] = (ic > ic.rolling(200).mean()).astype(int)
        mkt["mkt_vol_20"] = ic.pct_change().rolling(20).std()
        mkt["mkt_drawdown"] = ic / ic.rolling(60).max() - 1
        mkt = mkt.reset_index()
        mkt.columns = ["Date"] + list(mkt.columns[1:])
        mkt["Date"] = pd.to_datetime(mkt["Date"])
        panel = panel.merge(mkt, on="Date", how="left")

    for col in ("ret_20", "ret_60", "ret_120"):
        panel[f"rs_{col}"] = panel.groupby("Date")[col].rank(pct=True)

    panel = panel[panel["_dollar_vol"] >= MIN_DOLLAR_VOL].drop(columns=["_dollar_vol"])
    panel.to_parquet(out)
    return panel


def feature_cols(panel, drop_market):
    feats = [c for c in panel.columns if c not in META_COLS and panel[c].dtype != "O"]
    if drop_market:
        feats = [c for c in feats if not c.startswith("mkt_")]
    return feats


# ============================ 3. 워크포워드 ================================
@st.cache_data(show_spinner=False)
def run_walkforward(target, horizon, drop_market, max_per_day, regime_filter,
                    signal_rate, _raw_count):
    panel = build_panel(target, horizon, _raw_count)
    feats = feature_cols(panel, drop_market)
    p = panel.dropna(subset=["label"]).sort_values("Date").reset_index(drop=True)
    embargo = pd.Timedelta(days=int(round((horizon + 2) * 1.6)))

    all_oos, imps = [], []
    prog = st.progress(0.0, text="워크포워드 학습 중...")
    for k, (tr_end, te_s, te_e) in enumerate(WALKFWD_FOLDS):
        tr_end, te_s, te_e = map(pd.Timestamp, (tr_end, te_s, te_e))
        tr = p[p["Date"] <= tr_end - embargo]
        te = p[(p["Date"] >= te_s) & (p["Date"] <= te_e)]
        if len(te) == 0 or len(tr) < 20000:
            prog.progress((k + 1) / len(WALKFWD_FOLDS))
            continue
        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(tr[feats], tr["label"].astype(int))
        thr = float(np.quantile(m.predict_proba(tr[feats])[:, 1], 1 - signal_rate))
        pte = m.predict_proba(te[feats])[:, 1]
        o = te[["symbol", "Date", "close", "label", "fwd_max_ret", "trade_ret", "hit_day"]].copy()
        o["proba"] = pte
        o["signal"] = (pte >= thr).astype(int)
        if regime_filter and "mkt_above_sma200" in te.columns:
            o.loc[te["mkt_above_sma200"].values != 1, "signal"] = 0
        if max_per_day > 0:
            rk = o.groupby("Date")["proba"].rank(ascending=False, method="first")
            o.loc[rk > max_per_day, "signal"] = 0
        o["fold"] = f"{te_s.date()}~{te_e.date()}"
        all_oos.append(o)
        imps.append(pd.Series(m.feature_importances_, index=feats))
        prog.progress((k + 1) / len(WALKFWD_FOLDS))
    prog.empty()
    oos = pd.concat(all_oos, ignore_index=True)
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    return oos, imp


# ============================ 4. 통계검증 =================================
def _newey_west_t(x, lag):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 5:
        return np.nan, np.nan, n
    e = x - x.mean()
    s = (e @ e) / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        s += 2 * w * (e[k:] @ e[:-k]) / n
    se = np.sqrt(s / n)
    return x.mean(), x.mean() / se, n


@st.cache_data(show_spinner=False)
def validate(target, horizon, drop_market, max_per_day, regime_filter, signal_rate,
             _raw_count, nsim=2500):
    panel = build_panel(target, horizon, _raw_count)
    oos, _ = run_walkforward(target, horizon, drop_market, max_per_day, regime_filter,
                             signal_rate, _raw_count)
    sig = oos[oos["signal"] == 1].copy()
    base = float(oos["label"].mean())
    if len(sig) < 20:
        return {"n": len(sig), "insufficient": True, "base_rate": base}
    prec = float(sig["label"].mean())
    ev = float(sig["trade_ret"].mean())

    rng = np.random.default_rng(42)

    # ── 날짜블록 부트스트랩 (벡터화): 날짜 단위 복원추출 후 시그널 풀링 ──
    g = sig.groupby("Date")
    d_sum_tr = g["trade_ret"].sum().values
    d_sum_lb = g["label"].sum().values
    d_cnt = g.size().values.astype(float)
    D = len(d_cnt)
    bi = rng.integers(0, D, size=(nsim, D))
    tot_cnt = d_cnt[bi].sum(1)
    b_ev = d_sum_tr[bi].sum(1) / tot_cnt
    b_pr = d_sum_lb[bi].sum(1) / tot_cnt

    # ── 동일자 랜덤픽 순열검정 (날짜별 벡터화) ──
    day_u = {d: v["trade_ret"].dropna().values for d, v in panel.groupby("Date")}
    day_l = {d: v["label"].dropna().values for d, v in panel.groupby("Date")}
    cnts = sig.groupby("Date").size()
    sum_ev = np.zeros(nsim)
    sum_pr = np.zeros(nsim)
    tot = 0
    for d, cc in cnts.items():
        u = day_u.get(d)
        if u is None or len(u) == 0:
            continue
        ix = rng.integers(0, len(u), size=(nsim, cc))
        sum_ev += u[ix].sum(1)
        sum_pr += day_l[d][ix].sum(1)
        tot += cc
    n_ev = sum_ev / tot
    n_pr = sum_pr / tot

    daily = sig.groupby("Date")["trade_ret"].mean()
    mu, tstat, nn = _newey_west_t(daily.values, horizon)

    sig["year"] = sig["Date"].dt.year
    yr = sig.groupby("year").agg(n=("label", "size"), precision=("label", "mean"),
                                 ev=("trade_ret", "mean"),
                                 win=("trade_ret", lambda x: float((x > 0).mean())))
    pos = int((yr["ev"] > 0).sum())
    sign_p = stats.binomtest(pos, len(yr), 0.5, alternative="greater").pvalue
    bt = stats.binomtest(int(sig["label"].sum()), len(sig), base, alternative="greater").pvalue

    return {
        "n": int(len(sig)), "n_days": int(sig["Date"].nunique()),
        "period": f"{sig['Date'].min().date()} ~ {sig['Date'].max().date()}",
        "base_rate": base, "precision": prec, "ev": ev,
        "win_rate": float((sig["trade_ret"] > 0).mean()),
        "median_trade": float(sig["trade_ret"].median()),
        "avg_hold": float(sig["hit_day"].mean()),
        "trade_sharpe": float(ev / sig["trade_ret"].std()),
        "lift": prec / base,
        "binom_p": float(bt),
        "boot_ev_ci": [float(np.percentile(b_ev, 2.5)), float(np.percentile(b_ev, 97.5))],
        "boot_pr_ci": [float(np.percentile(b_pr, 2.5)), float(np.percentile(b_pr, 97.5))],
        "boot_p_ev_le_0": float((b_ev <= 0).mean()),
        "boot_p_pr_le_base": float((b_pr <= base).mean()),
        "perm_null_ev_mean": float(n_ev.mean()),
        "perm_null_pr_mean": float(n_pr.mean()),
        "perm_p_ev": float((n_ev >= ev).mean()),
        "perm_p_pr": float((n_pr >= prec).mean()),
        "nw_mean": float(mu), "nw_t": float(tstat), "nw_days": int(nn),
        "year_pos": pos, "year_total": int(len(yr)), "sign_p": float(sign_p),
        "by_year": yr.reset_index(),
    }


# ============================ 5. 오늘자 스크리닝 ===========================
@st.cache_resource(show_spinner=False)
def train_final(target, horizon, drop_market, signal_rate, _raw_count):
    panel = build_panel(target, horizon, _raw_count)
    feats = feature_cols(panel, drop_market)
    labeled = panel.dropna(subset=["label"])
    max_date = panel["Date"].max()
    embargo = pd.Timedelta(days=int(round((horizon + 2) * 1.6)))
    train = labeled[labeled["Date"] <= max_date - embargo]
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(train[feats], train["label"].astype(int))
    thr = float(np.quantile(m.predict_proba(train[feats])[:, 1], 1 - signal_rate))
    return m, feats, thr, max_date


def screen_today(target, horizon, drop_market, regime_filter, signal_rate, _raw_count, topn=30):
    panel = build_panel(target, horizon, _raw_count)
    m, feats, thr, max_date = train_final(target, horizon, drop_market, signal_rate, _raw_count)
    latest = panel.sort_values("Date").groupby("symbol").tail(1).copy()
    latest = latest[latest["Date"] >= max_date - pd.Timedelta(days=7)]
    latest["proba"] = m.predict_proba(latest[feats])[:, 1]
    regime_ok = True
    if regime_filter and "mkt_above_sma200" in latest.columns and latest["mkt_above_sma200"].notna().any():
        regime_ok = bool(latest.sort_values("Date")["mkt_above_sma200"].dropna().iloc[-1] == 1)
    latest = latest.sort_values("proba", ascending=False)
    lst = load_listing()
    nmap = dict(zip(lst["Symbol"], lst["Name"])) if "Name" in lst.columns else {}
    latest["name"] = latest["symbol"].map(lambda s: nmap.get(s, s))
    latest["signal"] = (latest["proba"] >= thr) & regime_ok
    return latest.head(topn), thr, max_date, regime_ok


# ================================ UI ======================================
st.title("📈 S&P500 급등종목 스크리너")
st.caption("이후 H거래일 이내 종가가 +T% 이상 상승할 종목을 머신러닝으로 선별하고, 통계적으로 검증합니다.")

with st.sidebar:
    st.header("⚙️ 설정")
    n_cached = count_cached()
    st.metric("수집된 종목 수", n_cached)
    if st.button("🔄 데이터 수집 / 갱신 (느림)"):
        pr = st.progress(0.0, text="다운로드 시작...")
        c = download_universe(pr)
        pr.empty()
        st.success(f"완료: {c} 종목 캐시됨")
        st.cache_data.clear()
        st.rerun()

    st.divider()
    target = st.select_slider("급등 기준 T (종가 상승률)", options=[0.05, 0.07, 0.10, 0.15], value=0.07,
                              format_func=lambda x: f"{x:.0%}")
    horizon = st.slider("보유 상한 H (거래일)", 3, 15, 10)
    signal_rate = st.select_slider("시그널 비율 (학습셋 상위 분위)", options=[0.002, 0.005, 0.01, 0.02, 0.05],
                                   value=0.02, format_func=lambda x: f"{x:.1%}")
    drop_market = st.checkbox("시장(mkt_*) 피처 제거 — 순수 종목선택력", value=True)
    max_per_day = st.slider("하루 최대 시그널 수 (0=무제한)", 0, 10, 3)
    regime_filter = st.checkbox("약세장 회피 (S&P500 > 200일선 인 날만 매수)", value=True)
    st.divider()
    st.caption("변경 후 아래 각 탭의 실행 버튼을 누르세요. 워크포워드 학습은 수 분 걸립니다.")

if n_cached < 50:
    st.warning("수집된 데이터가 부족합니다. 좌측 사이드바에서 **데이터 수집 / 갱신**을 먼저 실행하세요 "
               "(S&P500 약 500종목, 5~15분 소요).")
    st.stop()

rawc = n_cached
tab_bt, tab_val, tab_today, tab_info = st.tabs(
    ["🧪 백테스트", "📊 통계검증", "🎯 오늘의 급등후보", "ℹ️ 방법론·주의"])

# ---- 백테스트 탭 ----
with tab_bt:
    st.subheader("워크포워드 아웃오브샘플 성과")
    if st.button("백테스트 실행", key="run_bt", type="primary"):
        with st.spinner("패널 생성 + 워크포워드 학습 중... (수 분)"):
            oos, imp = run_walkforward(target, horizon, drop_market, max_per_day,
                                       regime_filter, signal_rate, rawc)
        st.session_state["oos"] = oos
        st.session_state["imp"] = imp

    if "oos" in st.session_state:
        oos = st.session_state["oos"]
        imp = st.session_state["imp"]
        sig = oos[oos["signal"] == 1]
        base = oos["label"].mean()
        if len(sig) == 0:
            st.error("현재 설정으로는 시그널이 발생하지 않았습니다. 시그널 비율을 높이거나 필터를 완화하세요.")
        else:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("시그널 수", f"{len(sig):,}", f"{sig['Date'].nunique()}일")
            c2.metric("정밀도(급등적중)", f"{sig['label'].mean():.1%}", f"기저 {base:.1%}")
            c3.metric("Lift", f"{sig['label'].mean()/base:.2f}x")
            c4.metric("트레이드 EV", f"{sig['trade_ret'].mean():+.2%}", f"승률 {(sig['trade_ret']>0).mean():.0%}")
            c5.metric("평균 보유", f"{sig['hit_day'].mean():.1f}일",
                      f"샤프 {sig['trade_ret'].mean()/sig['trade_ret'].std():.2f}")

            sig2 = sig.copy()
            sig2["year"] = sig2["Date"].dt.year
            yr = sig2.groupby("year").agg(시그널수=("label", "size"),
                                          정밀도=("label", "mean"),
                                          EV=("trade_ret", "mean"),
                                          승률=("trade_ret", lambda x: (x > 0).mean()))
            st.markdown("**연도별**")
            st.dataframe(yr.style.format({"정밀도": "{:.1%}", "EV": "{:+.2%}", "승률": "{:.0%}"}),
                         use_container_width=True)
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**연도별 EV**")
                st.bar_chart(yr["EV"])
            with cc2:
                st.markdown("**누적 트레이드 수익 (시그널 순차 진입 가정)**")
                eq = sig.sort_values("Date")["trade_ret"].cumsum()
                eq.index = sig.sort_values("Date")["Date"].values
                st.line_chart(eq)

            st.markdown("**피처 중요도 (상위 20)**")
            st.bar_chart(imp.head(20))
    else:
        st.info("설정을 확인하고 **백테스트 실행**을 누르세요.")

# ---- 통계검증 탭 ----
with tab_val:
    st.subheader("시그널이 우연이 아닌지 검증")
    nsim = st.select_slider("시뮬레이션 횟수", options=[1000, 2500, 5000], value=2500)
    if st.button("통계검증 실행", key="run_val", type="primary"):
        with st.spinner("부트스트랩 / 순열검정 중..."):
            r = validate(target, horizon, drop_market, max_per_day, regime_filter,
                         signal_rate, rawc, nsim)
        st.session_state["val"] = r

    if "val" in st.session_state:
        r = st.session_state["val"]
        if r.get("insufficient"):
            st.error(f"시그널이 {r['n']}건으로 너무 적어 검증 불가. 설정을 완화하세요.")
        else:
            st.write(f"**시그널 {r['n']:,}건 / 고유거래일 {r['n_days']} / 기간 {r['period']}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("정밀도", f"{r['precision']:.1%}", f"기저 {r['base_rate']:.1%}")
            c2.metric("트레이드 EV", f"{r['ev']:+.2%}", f"중앙값 {r['median_trade']:+.2%}")
            c3.metric("Newey-West t", f"{r['nw_t']:.2f}", f"{r['nw_days']}일")

            rows = [
                ["① 이항검정 (정밀도 > 기저율, 독립가정)", f"p = {r['binom_p']:.2e}",
                 "✅" if r['binom_p'] < 0.01 else "⚠️"],
                ["② 날짜블록 부트스트랩 — P(EV ≤ 0)", f"{r['boot_p_ev_le_0']:.4f}  "
                 f"(EV 95%CI [{r['boot_ev_ci'][0]:+.2%}, {r['boot_ev_ci'][1]:+.2%}])",
                 "✅" if r['boot_p_ev_le_0'] < 0.01 else "⚠️"],
                ["② 날짜블록 부트스트랩 — P(정밀도 ≤ 기저율)", f"{r['boot_p_pr_le_base']:.4f}",
                 "✅" if r['boot_p_pr_le_base'] < 0.01 else "⚠️"],
                ["③ 동일자 랜덤픽 순열검정 — EV", f"관측 {r['ev']:+.2%}  vs  null {r['perm_null_ev_mean']:+.2%}   "
                 f"p = {r['perm_p_ev']:.4f}", "✅" if r['perm_p_ev'] < 0.05 else "⚠️"],
                ["③ 동일자 랜덤픽 순열검정 — 정밀도", f"관측 {r['precision']:.1%}  vs  null "
                 f"{r['perm_null_pr_mean']:.1%}   p = {r['perm_p_pr']:.4f}",
                 "✅" if r['perm_p_pr'] < 0.05 else "⚠️"],
                ["④ Newey-West t (일별 시그널수익, 자기상관 보정)", f"t = {r['nw_t']:.2f}",
                 "✅" if abs(r['nw_t']) > 3 else ("△" if abs(r['nw_t']) > 2 else "⚠️")],
                ["⑤ 연도 부호검정", f"{r['year_pos']}/{r['year_total']}년 양수   p = {r['sign_p']:.3f}",
                 "✅" if r['sign_p'] < 0.1 else "⚠️"],
            ]
            st.table(pd.DataFrame(rows, columns=["검정", "결과", "판정"]))

            st.markdown("**③ 순열검정 해석** — 관측 성과가 *같은 날 아무 종목이나 매수했을 때(null)* 를 "
                        "유의하게 능가해야 '종목선택 실력'입니다. null 대비 초과폭이 작으면 대부분은 "
                        "'그 날이 좋았던 것'입니다.")
            st.dataframe(
                r["by_year"].rename(columns={"year": "연도", "n": "시그널수", "precision": "정밀도",
                                             "ev": "EV", "win": "승률"}).style.format(
                    {"정밀도": "{:.1%}", "EV": "{:+.2%}", "승률": "{:.0%}"}),
                use_container_width=True)

            passes = sum(1 for _, _, v in rows if v == "✅")
            if passes >= 5:
                st.success(f"검정 {passes}/7 통과 — 통계적으로 신뢰할 만합니다. (단, 아래 '주의' 탭의 생존편향 유의)")
            elif passes >= 3:
                st.warning(f"검정 {passes}/7 통과 — 부분적 근거. 파라미터 민감도를 더 확인하세요.")
            else:
                st.error(f"검정 {passes}/7 통과 — 신뢰하기 어렵습니다.")
    else:
        st.info("먼저 **백테스트**를 돌린 뒤 이 탭에서 **통계검증 실행**을 누르세요 (동일 캐시 재사용).")

# ---- 오늘의 급등후보 탭 ----
with tab_today:
    st.subheader("전체 데이터 재학습 → 최신 거래일 스코어링")
    if st.button("오늘자 스크리닝 실행", key="run_today", type="primary"):
        with st.spinner("최종 모델 학습 + 스코어링..."):
            cand, thr, asof, regime_ok = screen_today(target, horizon, drop_market,
                                                      regime_filter, signal_rate, rawc)
        st.session_state["cand"] = (cand, thr, asof, regime_ok)

    if "cand" in st.session_state:
        cand, thr, asof, regime_ok = st.session_state["cand"]
        st.write(f"기준일 **{pd.Timestamp(asof).date()}**  ·  시그널 임계확률 **{thr:.3f}**  ·  "
                 f"시장레짐 {'🟢 매수구간' if regime_ok else '🔴 회피구간 (약세장 필터 발동)'}")
        n_sig = int(cand["signal"].sum())
        if n_sig == 0:
            st.info("현재 임계선을 넘는 확정 시그널은 없습니다. 아래는 확률 상위 관찰 후보입니다.")
        else:
            st.success(f"확정 시그널 {n_sig}건")
        show = cand[["symbol", "name", "Date", "close", "proba", "signal",
                     "ret_20", "ret_60", "dist_52w_high", "rs_ret_60", "vol_ratio_20", "rsi_14"]].copy()
        show.columns = ["티커", "종목", "기준일", "종가", "급등확률", "시그널",
                        "20일수익", "60일수익", "52주고점대비", "RS(60일)", "거래량비", "RSI14"]
        st.dataframe(
            show.style.format({"종가": "{:.2f}", "급등확률": "{:.1%}", "20일수익": "{:+.1%}",
                               "60일수익": "{:+.1%}", "52주고점대비": "{:.1%}", "RS(60일)": "{:.2f}",
                               "거래량비": "{:.2f}", "RSI14": "{:.0f}"})
            .apply(lambda s: ["background-color:#0b3d0b" if v else "" for v in cand["signal"]],
                   axis=0, subset=["시그널"]),
            use_container_width=True, height=560)
        st.caption("급등확률 = 모델이 추정한 'H거래일 내 +T% 종가 도달' 확률. 시그널 = 임계확률 초과 & 레짐통과.")
    else:
        st.info("**오늘자 스크리닝 실행**을 누르세요.")

# ---- 방법론 탭 ----
with tab_info:
    st.markdown(f"""
### 파이프라인
1. **데이터** — FinanceDataReader 로 현재 S&P500 구성종목의 2010년~현재 일봉 수집·캐시.
2. **라벨** — 각 거래일 t 종가 대비, 이후 **1~{horizon}거래일** 중 종가 최고가가 **+{target:.0%}** 이상이면 급등(1).
3. **파생변수(~45개, 전부 시점정합)** — 모멘텀(1~120일), 이동평균 이격·기울기, 변동성/ATR/볼린저,
   RSI(2·14), 52주 고저 거리, 낙폭, MACD, 거래량비/OBV, 갭, 횡단면 상대강도 순위, (옵션)시장 레짐.
4. **모델** — LightGBM 이진분류, **확장윈도우 워크포워드** 5개 폴드.
   라벨 중첩 누수를 막기 위해 폴드 경계에 **{int(round((horizon+2)*1.6))}일 embargo**.
   학습셋 예측확률 **상위 {signal_rate:.1%}** 지점을 시그널 임계선으로 잡아 테스트에 그대로 적용.
5. **거래 가정** — 시그널 당일 종가 매수, 목표가 first-touch 시 청산, 미도달 시 t+{horizon} 종가 청산.
   `trade_ret` = 실현수익, `EV` = 그 평균.

### 통계검증
- **날짜블록 부트스트랩**: 시그널을 날짜 단위로 재표집 → 군집·중첩을 반영한 EV/정밀도 신뢰구간.
- **동일자 랜덤픽 순열검정**: *같은 날짜에 무작위 종목*을 같은 수만큼 매수한 분포(null)와 비교 →
  '종목선택 실력'만 분리. **시장 타이밍·생존편향이 null 에도 동일하게 들어가므로 상대비교가 핵심.**
- **Newey-West t**: 일별 평균 시그널수익의 자기상관 보정 t값.
- **연도 부호검정**: 성과의 연도별 일관성.

### ⚠️ 주의 (반드시 인지)
- **생존편향**: 유니버스가 *현재* S&P500 구성종목입니다. 과거 구간 전체에서 이들은 '살아남아 지수에
  편입된' 종목이라, 장기 모멘텀·낙폭반등 성과가 구조적으로 부풀려집니다. 순열검정의 null 도 같은
  유니버스를 쓰므로 *상대적* 우위는 유효하지만, **절대 수익률은 낙관적**으로 보아야 합니다.
- **시장 피처 포함 시** 모델이 종목선택이 아니라 '폭락 후 반등' 타이밍을 학습해, 시그널이 2020년
  같은 특정 국면에 몰립니다. 반복가능한 종목선별을 원하면 **시장 피처 제거**를 켜세요.
- 거래비용·슬리피지·부분체결·공매도 제약 미반영. 목표가 first-touch 는 장중 도달을 종가로 근사.
- 파라미터를 여러 개 시도할수록 다중검정 편향이 커집니다. 순열검정 p 와 연도 일관성을 함께 보세요.
""")

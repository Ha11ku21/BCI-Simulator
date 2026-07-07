import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("BCI Transmission Simulator: BCI Transmission Simulator: As this was created to suit smartphones, it may not necessarily yield the same results as an actual experiment.")

# 各種パラメータの入力（VB.NETのデフォルト値に合わせやすいように調整）
noiseAmp = st.slider("Noise Amplitude [μA/cm²]", 0.0, 10.0, 0.0)  # 一致確認は 0.0 を推奨
adc_bits = st.slider("ADC Bits", 4, 16, 8)
compression_ratio = st.slider("Compression Ratio", 1, 1000, 10)
run = st.button("Run")

# スパイク検出の閾値を判定する関数（VB.NETのDetectSpikesに準拠：0.0V付近）
def get_spike_times(V, threshold=0.0):
    spikes = []
    for i in range(1, len(V)):
        if V[i-1] <= threshold and V[i] > threshold:
            spikes.append(i)
    return np.array(spikes)

# 発火タイミングエラーの計算（VB.NETの timingED / timingErr のアルゴリズムを再現）
def firing_error(normal_spikes, test_spikes, dt):
    if len(normal_spikes) == 0 or len(test_spikes) == 0:
        return 0.0
    
    total_err = 0.0
    for sn in normal_spikes:
        # 最も近いスパイクとの差分（インデックスの絶対差）を探索
        best_err = np.min(np.abs(test_spikes - sn))
        total_err += best_err * dt
        
    return total_err / len(normal_spikes)

# Hodgkin-Huxley ゲート関数
def alpha_m(V):
    if abs(V + 40) < 0.0001: return 1.0
    return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

def beta_m(V): return 4 * np.exp(-(V + 65) / 18)

def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)

def beta_h(V): return 1 / (1 + np.exp(-(V + 35) / 10))

def alpha_n(V):
    if abs(V + 55) < 0.0001: return 0.1
    return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

def beta_n(V): return 0.125 * np.exp(-(V + 65) / 80)


if run:
    # --- パラメータをVB.NETに厳密に統一 ---
    dt = 0.0001  # タイムステップ (VB.NET: 0.0001)
    t = 50000   # ステップ数 (時間を完全に合わせる場合は 1000000 にしてください)
    
    Cm, gNa, gK, gL = 1.0, 120.0, 36.0, 0.3
    ENa, EK, EL = 50.0, -77.0, -54.4

    Vnormal = np.zeros(t)
    m, h, n = np.zeros(t), np.zeros(t), np.zeros(t)
    I1 = np.zeros(t)

    # VB.NETの初期値に固定
    Vnormal[0] = -65.0
    m[0] = 0.05
    h[0] = 0.6
    n[0] = 0.32
    skk = 0
    # 乱数シードの固定（ノイズがある場合の再現用）
    np.random.seed(42)

    # 入力電流計算 (全区間でベース電流10が入るロジック)
    for s in range(t):
        base = 10.0
        if noiseAmp > 0:
            noise = noiseAmp * (np.random.random() - 0.5)
            I1[s] = base + noise
        else:
            I1[s] = base

    # 電位シミュレーション（メインループ）
    for s in range(t - 1):
        m[s+1] = m[s] + dt * (alpha_m(Vnormal[s]) * (1 - m[s]) - beta_m(Vnormal[s]) * m[s])
        h[s+1] = h[s] + dt * (alpha_h(Vnormal[s]) * (1 - h[s]) - beta_h(Vnormal[s]) * h[s])
        n[s+1] = n[s] + dt * (alpha_n(Vnormal[s]) * (1 - n[s]) - beta_n(Vnormal[s]) * n[s])

        INa = gNa * (m[s] ** 3) * h[s] * (Vnormal[s] - ENa)
        IK = gK * (n[s] ** 4) * (Vnormal[s] - EK)
        IL = gL * (Vnormal[s] - EL)

        Vnormal[s+1] = Vnormal[s] + dt * (I1[s] - INa - IK - IL) / Cm

    # ==========================================
    # 1. Digital (ADC量子化)
    # ==========================================
    Vmin, Vmax = -100.0, 50.0
    stepSize = (Vmax - Vmin) / (2 ** adc_bits)
    Vdigital = np.round((Vnormal - Vmin) / stepSize) * stepSize + Vmin

    # ==========================================
    # 2. Compression (Vas: 線形補間)
    # ==========================================
    # ==========================================
    # 2. Compression (VB.NET版)
    # ==========================================
    Vas = np.zeros(t)
    Vas[0] = Vnormal[0]

    sk1 = compression_ratio
    skk = 0

    for s in range(1, t):

        if s % sk1 == 0:

            skk += 1

            curr = skk * sk1
            prev = (skk - 1) * sk1

            if curr >= t:
                break

            Vas[curr] = Vnormal[curr]

            for i in range(prev + 1, curr):
                Vas[i] = Vas[prev] + \
                    (Vas[curr] - Vas[prev]) * \
                    (i - prev) / (curr - prev)

    # 最後のサンプル位置
    last = (t - 1) // sk1 * sk1

    # VBでは最後も元データを保持
    Vas[last] = Vnormal[last]

    # 最後の区間は一定値
    for i in range(last + 1, t):
        Vas[i] = Vas[last]
    spike_threshold = -65.0  # VB.NETの spikt
    Vsp = np.full(t, spike_threshold)
    spike_count = 0
    count = 0
    ssp1, ssp2, ssp3 = 0, 0, 0

    for s in range(t):
        if Vnormal[s] <= spike_threshold and count == 0:
            Vsp[s] = spike_threshold
        if Vnormal[s] > spike_threshold and count == 0:
            ssp1 = s
            count = 1
        if Vnormal[s] < spike_threshold and count == 1:
            ssp2 = s - 1
            if ssp2 <= ssp1:
                count = 0
                spike_count += 1
                continue
            
            ssp3 = ssp1
            for s3 in range(ssp1, ssp2 + 1):
                if Vnormal[s3] > Vnormal[ssp3]:
                    ssp3 = s3

            Vsp[ssp1] = Vnormal[ssp1]
            Vsp[ssp3] = Vnormal[ssp3]
            Vsp[ssp2] = Vnormal[ssp2]
            spike_count += 1

            for s3 in range(ssp1 + 1, ssp3):
                Vsp[s3] = Vsp[ssp1] + (Vsp[ssp3] - Vsp[ssp1]) * (s3 - ssp1) / (ssp3 - ssp1)
            for s3 in range(ssp3 + 1, ssp2):
                Vsp[s3] = Vsp[ssp3] + (Vsp[ssp2] - Vsp[ssp3]) * (s3 - ssp3) / (ssp2 - ssp3)
            count = 0

    # ==========================================
    # 評価値の計算 (Evaluation)
    # ==========================================
    # スパイク発火時間の取得 (判定閾値は 0.0V)
    spike_normal = get_spike_times(Vnormal, threshold=0.0)
    spike_digital = get_spike_times(Vdigital, threshold=0.0)
    spike_as = get_spike_times(Vas, threshold=0.0)
    spike_sp = get_spike_times(Vsp, threshold=0.0)

    # 発火タイミングエラー (Firing Error)
    ignition_digital = firing_error(spike_normal, spike_digital, dt)
    ignition_compression = firing_error(spike_normal, spike_as, dt)
    ignition_spike = firing_error(spike_normal, spike_sp, dt)

    # RMSEの計算
    rmse_digital = np.sqrt(np.mean((Vnormal - Vdigital) ** 2))
    rmse_compression = np.sqrt(np.mean((Vnormal - Vas) ** 2))
    rmse_spike = np.sqrt(np.mean((Vnormal - Vsp) ** 2))

    # 相関係数 (Correlation) の計算
    corr_digital = np.corrcoef(Vnormal, Vdigital)[0, 1]
    corr_compression = np.corrcoef(Vnormal, Vas)[0, 1]
    corr_spike = np.corrcoef(Vnormal, Vsp)[0, 1]

    # 消費電力計算 (VB.NET側の計算式を完全再現)
    sr, tr = 16, 32
    
    bitNormal = t * sr
    rateNormal = bitNormal / (t * dt)
    powerNormal = rateNormal * 1e-8

    bitDigital = t * adc_bits
    rateDigital = bitDigital / (t * dt)
    powerDigital = rateDigital * 1e-8

    bitCompression = (t / sk1) * sr
    rateCompression = bitCompression / (t * dt)
    powerCompression = rateCompression * 1e-8

    bitSpike = spike_count * tr  # 特徴点スパイク数に依存
    rateSpike = bitSpike / (t * dt)
    powerSpike = rateSpike * 1e-8

    # ==========================================
    # グラフ描画 (Graph)
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(Vnormal, label="Normal", color="black", alpha=0.7)
    ax.plot(Vdigital, label="Digital", linestyle="--", alpha=0.8)
    ax.plot(Vas, label="Compression (Vas)", linestyle="-.", alpha=0.8)
    ax.plot(Vsp, label="Spike (Vsp)", linestyle=":", alpha=0.8)
    
    ax.set_title("BCI Communication Methods (Unified)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Membrane Potential (mV)")
    ax.legend()
    st.pyplot(fig)

    # ==========================================
    # 結果表示 (Table & Metrics)
    # ==========================================
    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Digital Firing Error", f"{ignition_digital:.6f}")
    col2.metric("Compression Firing Error", f"{ignition_compression:.6f}")
    col3.metric("Spike Firing Error", f"{ignition_spike:.6f}")

    st.table({
        "Method": ["Digital", "Compression", "Spike"],
        "RMSE": [
            round(rmse_digital, 6),
            round(rmse_compression, 6),
            round(rmse_spike, 6)
        ],
        "Correlation": [
            round(corr_digital, 6),
            round(corr_compression, 6),
            round(corr_spike, 6)
        ],
        "Power": [
            f"{powerDigital:.12f}",
            f"{powerCompression:.12f}",
            f"{powerSpike:.12f}"
        ],
        "Data Volume [Points]": [
            t,
            int(t / sk1),
            spike_count
        ]
    })

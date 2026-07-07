import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("BCI Transmission Simulator: Unified Version")

noiseAmp = st.slider("Noise Amplitude[μA/cm²]", 0.0, 10.0, 0.0) # 一致確認のため初期値0を推奨
adc_bits = st.slider("ADC Bits", 4, 16, 8)
compression_ratio = st.slider("Compression Ratio", 1, 1000, 10)
run = st.button("Run")

# スパイク検出の閾値をVB.NET（0V付近）に合わせる
def get_spike_times(V, threshold=0.0):
    spikes = []
    for i in range(1, len(V)):
        if V[i-1] <= threshold and V[i] > threshold:
            spikes.append(i)
    return np.array(spikes)

# 発火タイミングエラーの計算（VB.NETのロジックを再現）
def firing_error(normal_spikes, test_spikes, dt):
    if len(normal_spikes) == 0 or len(test_spikes) == 0:
        return 0.0
    
    total_err = 0.0
    for sn in normal_spikes:
        # 最も近いスパイクとの差分（インデックスの差）を探す
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
    dt = 0.0001  # VB.NET: 0.0001
    t = 100000   # ステップ数（時間を合わせる場合は調整してください）
    
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

    # 乱数シードの固定（ノイズがある場合の再現用）
    np.random.seed(42)

    # 入力電流計算
    for s in range(t):
        # VB.NETは s1=0, s2=t なので全区間でベース電流10が入る
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

    # --- 1. Digital (ADC量子化幅の計算ロジックをVBに統合) ---
    Vmin, Vmax = -100.0, 50.0
    stepSize = (Vmax - Vmin) / (2 ** adc_bits)
    # VB.NETの Math.Round() は四捨五入（銀行丸め注意、Pythonの補正）
    Vdigital = np.round((Vnormal - Vmin) / stepSize) * stepSize + Vmin

    # --- 2. Compression (Vas: 線形補間) ---
    Vas = np.zeros(t)
    Vas[0] = Vnormal[0]
    sk1 = compression_ratio
    
    # VB.NETの「sk = s Mod sk1」による間引きと線形補間ロジックの再現
    for s in range(1, t):
        if s % sk1 == 0:
            Vas[s] = Vnormal[s]
            prev_idx = s - sk1
            # 隙間を直線補間
            for x in range(prev_idx + 1, s):
                Vas[x] = Vas[prev_idx] + (Vas[s] - Vas[prev_idx]) * (x - prev_idx) / sk1
    # 端数の処理（tがsk1で割り切れない場合、最後の残りを補間）
    last_mod = t % sk1
    if last_mod != 0:
        prev_idx = t - last_mod - 1
        for x in range(prev_idx + 1, t):
            Vas[x] = Vas[prev_idx] # 簡易的にホールド、またはVBのロジックに合わせる

    # --- 3. Spike (Vsp: 特徴点抽出) ---
    # VB.NETの spikt (閾値) を -65.0 とする
    spike_threshold = -65.0 
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

    # --- スパイクタイミング評価 ---
    # 発火検出用の閾値（spikthk）は双方 0.0V と判定
    spike_normal = get_spike_times(Vnormal, threshold=0.0)
    spike_digital = get_spike_times(Vdigital, threshold=0.0)
    spike_as = get_spike_times(Vas, threshold=0.0)
    spike_sp = get_spike_times(Vsp, threshold=0.0)

    ignition_digital = firing_error(spike_normal, spike_digital, dt)
    ignition_compression = firing_error(spike_normal, spike_as, dt)
    ignition_spike = firing_error(spike_normal, spike_sp, dt)

    # --- 消費電力・データ量計算（VB.NET準拠） ---
    sr, tr = 16, 32  # VB.NETの値を採用
    bitNormal = t * sr
    rateNormal = bitNormal / (t * dt)
    powerNormal = rateNormal * 1e-8

    bitDigital = t * adc_bits
    rateDigital = bitDigital / (t * dt)
    powerDigital = rateDigital * 1e-8

    bitCompression = (t / sk1) * sr
    rateCompression = bitCompression / (t * dt)
    powerCompression = rateCompression * 1e-8

    bitSpike = spike_count * tr  # スパイク数 × tr
    rateSpike = bitSpike / (t * dt)
    powerSpike = rateSpike * 1e-8

  
    rmse_digital = np.sqrt(
        np.mean(
            (Vnormal - Vdigital) ** 2
        )
    )

    corr_digital = np.corrcoef(
        Vnormal,
        Vdigital
    )[0, 1]

    rmse_compression = np.sqrt(
        np.mean(
            (Vnormal - Vcompression) ** 2
        )
    )

    corr_compression = np.corrcoef(
        Vnormal,
        Vcompression
    )[0, 1]

    rmse_spike = np.sqrt(
        np.mean(
            (Vnormal - Vsp) ** 2
        )
    )

    corr_spike = np.corrcoef(
        Vnormal,
        Vsp
    )[0, 1]

    ignition_digital = firing_error(
        spike_normal,
        spike_digital
    )

    ignition_compression = firing_error(
        spike_normal,
        spike_compression
    )

    ignition_spike = firing_error(
        spike_normal,
        spike_spike
    )

    # -----------------------------
    # Graph
    # -----------------------------
    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        Vnormal,
        label="Normal"
    )

    ax.plot(
        Vdigital,
        label="Digital"
    )

    ax.plot(
        Vcompression,
        label="Compression"
    )

    ax.plot(
        Vsp,
        label="Spike"
    )

    ax.set_title(
        "BCI Communication Methods"
    )

    ax.set_xlabel(
        "Time Step"
    )

    ax.set_ylabel(
        "Membrane Potential (mV)"
    )

    ax.legend()

    st.pyplot(fig)

    # -----------------------------
    # Table
    # -----------------------------
    st.subheader(
        "Evaluation"
    )

    st.write(
        "Digital Firing Error",
        ignition_digital
    )

    st.write(
        "Compression Firing Error",
        ignition_compression
    )

    st.write(
        "Spike Firing Error",
        ignition_spike
    )

    st.table({
        "Method": [
            "Digital",
            "Compression",
            "Spike"
        ],
        "RMSE": [
            round(rmse_digital, 4),
            round(rmse_compression, 4),
            round(rmse_spike, 4)
        ],
        "Correlation": [
            round(corr_digital, 4),
            round(corr_compression, 4),
            round(corr_spike, 4)
        ],
        "Power": [
            round(powerDigital, 8),
            round(powerCompression, 8),
            round(powerSpike, 8)
        ],
        "Data Volume": [
            t,
            len(compressed_data),
            spike_points
        ]
    })

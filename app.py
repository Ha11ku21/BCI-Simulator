import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("BCI Transmission Simulator: It is tailored to the smartphone's features.")

noiseAmp = st.slider(
    "Noise Amplitude[μA/cm²]",
    0.0,
    10.0,
    1.0
)

adc_bits = st.slider(
    "ADC Bits",
    4,
    16,
    8
)

compression_ratio = st.slider(
    "Compression Ratio",
    1,
    1000,
    10
)

run = st.button("Run")
def get_spike_times(V, threshold=0):

    spikes = []

    for i in range(1, len(V)):

        if V[i-1] < threshold and V[i] >= threshold:
            spikes.append(i)

    return np.array(spikes)
def firing_error(a,b):

    n = min(
        len(a),
        len(b)
    )

    if n == 0:
        return np.nan

    return np.mean(
        np.abs(
            a[:n]-b[:n]
        )
    )
def alpha_m(V):
    if abs(V + 40) < 0.0001:
        return 1.0
    return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))


def beta_m(V):
    return 4 * np.exp(-(V + 65) / 18)


def alpha_h(V):
    return 0.07 * np.exp(-(V + 65) / 20)


def beta_h(V):
    return 1 / (1 + np.exp(-(V + 35) / 10))


def alpha_n(V):
    if abs(V + 55) < 0.0001:
        return 0.1
    return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))


def beta_n(V):
    return 0.125 * np.exp(-(V + 65) / 80)


if run:

    dt = 0.001
    t = 50000

    Cm = 1
    gNa = 120
    gK = 36
    gL = 0.3

    ENa = 50
    EK = -77
    EL = -54.4

    Vnormal = np.zeros(t)
    m = np.zeros(t)
    h = np.zeros(t)
    n = np.zeros(t)
    I1 = np.zeros(t)

    Vnormal[0] = -65

    # 初期値
    m[0] = alpha_m(-65) / (alpha_m(-65) + beta_m(-65))
    h[0] = alpha_h(-65) / (alpha_h(-65) + beta_h(-65))
    n[0] = alpha_n(-65) / (alpha_n(-65) + beta_n(-65))

    # 入力電流（ノイズ含む）
    for s in range(t):
        if 500 <= s <= t:
            base = 10
        else:
            base = 0

        noise = noiseAmp * (np.random.random() - 0.5)
        I1[s] = base + noise if noiseAmp > 0 else base

    # シミュレーション
    for s in range(t - 1):

        m[s+1] = m[s] + dt*(alpha_m(Vnormal[s])*(1-m[s]) - beta_m(Vnormal[s])*m[s])
        h[s+1] = h[s] + dt*(alpha_h(Vnormal[s])*(1-h[s]) - beta_h(Vnormal[s])*h[s])
        n[s+1] = n[s] + dt*(alpha_n(Vnormal[s])*(1-n[s]) - beta_n(Vnormal[s])*n[s])

        INa = gNa * m[s]**3 * h[s] * (Vnormal[s] - ENa)
        IK  = gK  * n[s]**4 * (Vnormal[s] - EK)
        IL  = gL  * (Vnormal[s] - EL)

        Vnormal[s+1] = Vnormal[s] + dt*(I1[s] - INa - IK - IL)/Cm

    # ADC量子化（ここは完全に外）
    spike_normal = get_spike_times(
    	Vnormal
    )
    # -----------------------------
    # Digital
    # -----------------------------
    levels = 2 ** adc_bits
    Vdigital = np.zeros(t)

    vmin = np.min(Vnormal)
    vmax = np.max(Vnormal)

    for i in range(t):
        normalized = (Vnormal[i] - vmin) / (vmax - vmin)
        q = round(normalized * (levels - 1))
        Vdigital[i] = q / (levels - 1) * (vmax - vmin) + vmin
    spike_digital = get_spike_times(
    	Vdigital
    )
    # -----------------------------
    # Compression
    # -----------------------------
    step = compression_ratio

    compressed_idx = np.arange(
        0,
        t,
        step
    )

    compressed_data = Vnormal[
        compressed_idx
    ]

    Vcompression = np.interp(
        np.arange(t),
        compressed_idx,
        compressed_data
    )
    spike_compression = get_spike_times(
    	Vcompression
    )
    # -----------------------------
    # Spike
    # -----------------------------
    spike_threshold = -65
    spike_points = 0

    Vsp = np.full(t, spike_threshold)

    count = 0
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
                continue

            ssp3 = ssp1

            for s3 in range(ssp1, ssp2 + 1):

                if Vnormal[s3] > Vnormal[ssp3]:
                    ssp3 = s3

            Vsp[ssp1] = Vnormal[ssp1]
            Vsp[ssp3] = Vnormal[ssp3]
            Vsp[ssp2] = Vnormal[ssp2]

            spike_points += 3

            for s3 in range(ssp1 + 1, ssp3):

                Vsp[s3] = (
                    Vsp[ssp1]
                    + (Vsp[ssp3] - Vsp[ssp1])
                    * (s3 - ssp1)
                    / (ssp3 - ssp1)
                )

            for s3 in range(ssp3 + 1, ssp2):

                Vsp[s3] = (
                    Vsp[ssp3]
                    + (Vsp[ssp2] - Vsp[ssp3])
                    * (s3 - ssp3)
                    / (ssp2 - ssp3)
                )

            count = 0

    spike_spike = get_spike_times(
        Vsp
    )

    # -----------------------------
    # Power Consumption
    # -----------------------------
    sr = 16
    tr = 24

    bitNormal = t * sr
    rateNormal = bitNormal / (t * dt)
    powerNormal = rateNormal * 0.00000001

    bitDigital = t * adc_bits
    rateDigital = bitDigital / (t * dt)
    powerDigital = rateDigital * 0.00000001

    bitCompression = (t / compression_ratio) * sr
    rateCompression = bitCompression / (t * dt)
    powerCompression = rateCompression * 0.00000001

    bitSpike = spike_points * tr
    rateSpike = bitSpike / (t * dt)
    powerSpike = rateSpike * 0.00000001

    # -----------------------------
    # Evaluation
    # -----------------------------
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

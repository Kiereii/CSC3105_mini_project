# Problem Definition: UWB LOS/NLOS Classification

## What is UWB?

**Ultra-Wideband (UWB)** is a radio technology that:
- Operates at 3.1-10.6 GHz frequency range
- Uses very short nanosecond pulses
- Provides **centimeter-level positioning accuracy**
- Used in: Apple AirTag, Samsung SmartTag, iPhone 15+, indoor positioning systems

The key advantage of UWB over WiFi/Bluetooth: extremely high precision through **precise time-of-arrival measurements**.

---

## LOS vs NLOS: The Core Problem

### Line-of-Sight (LOS) ✓
```
Transmitter ==========> Receiver
         (Direct path)
```
- Signal travels directly transmitter → receiver
- **Strongest signal amplitude**
- **Most accurate distance measurement**
- Shortest signal path

### Non-Line-of-Sight (NLOS) ✗
```
Transmitter ===| Wall |===> Receiver
            (Reflected paths)
```
- Signal bounces off walls, furniture, people
- **Weaker signal amplitude**
- **Longer path length → longer measured distance**
- Inaccurate range estimates

### The Challenge
When measuring a radio signal, you often see **BOTH LOS and NLOS reflections mixed together**. The receiver can't tell them apart. 

**The question:** Can we automatically detect which type of signal path this is?

This is a **classification problem** — categorize each measurement as LOS (0) or NLOS (1).

---

## Your Dataset

### Size
- **Total samples:** 41,568 UWB channel impulse response (CIR) measurements
- **Training data:** 33,254 samples (80%)
- **Test data:** 8,314 samples (20%)

### Balance
- **LOS samples:** 20,997 (50.5%)
- **NLOS samples:** 20,571 (49.5%)
- **Perfect balance!** ✓

### Features (136 total)

#### Core Features (16)
| Feature | Meaning | Why It Matters |
|---------|---------|---|
| **RXPACC** | Received preamble accumulation | **MOST IMPORTANT!** NLOS collects more due to reflections |
| **RANGE** | Measured distance (meters) | NLOS shows longer ranges |
| **FP_IDX** | First path index | Where signal first arrives |
| **FP_AMP1/2/3** | First path amplitudes | LOS has stronger first-path |
| **SNR/SNR_dB** | Signal-to-Noise Ratio | LOS has better quality |
| **CIR_PWR** | Total CIR power | Overall signal strength |
| **STDEV_NOISE** | Noise standard deviation | NLOS has higher noise |

#### CIR Features (120)
- **Channel Impulse Response samples** from indices 730-850
- The "fingerprint" of the signal over time
- Peak location and shape **differ between LOS and NLOS**

---

## The Project Requirements

### Task 1: LOS/NLOS Classification
**Question:** Is the dominant path LOS or NLOS?
- **Input:** 136 features
- **Output:** 0 (LOS) or 1 (NLOS)
- **Goal:** >85% accuracy

### Task 2: Range Estimation (Two Paths)
**Question:** What is the measured distance for each path?
- **Input:** Same 136 features
- **Output:** Distance in meters
- **Goal:** <2m error (RMSE)

### Task 3: Pair-Level Classification  
**Question:** Looking at BOTH paths, is it LOS+NLOS or NLOS+NLOS?
- **Input:** Features for both paths
- **Output:** Path pair class
- **Goal:** Determine if line-of-sight exists

---

## Why This Matters (Real-World Impact)

```
Without LOS/NLOS Detection:
  → Use all signals equally
  → NLOS signals corrupted
  → **Positioning error: 5-20 meters** ❌

With LOS/NLOS Detection:
  → Downweight NLOS signals
  → Trust LOS signals
  → **Positioning error: <1 meter** ✓
```

**The difference:** Exact location vs "somewhere in this room"

---

## Your Success

- ✅ **88.69% classification accuracy** (Random Forest)
- ✅ **95.35% AUC** (excellent ranking ability)
- ✅ **1.28m range RMSE** for Path 1
- ✅ **1.32m range RMSE** for Path 2
- ✅ **Feature importance analysis** (RXPACC = 15.6%)
- ✅ **Multiple algorithm comparison** (RF, LR, SVM, XGBoost)

---

Next: Read [02_Complete_Pipeline.md](02_Complete_Pipeline.md)


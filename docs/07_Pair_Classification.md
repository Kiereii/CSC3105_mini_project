# Pair-Level Classification: Beyond Individual Paths

## The Question

Instead of: **"Is this path LOS or NLOS?"**

Ask: **"Do these TWO paths form LOS+NLOS or NLOS+NLOS pair?"**

---

## Why Pair Classification?

### The Physical Reality

```
LOS environment:
  Path 1: LOS (direct)
  Path 2: NLOS (reflection)
  → Pair: LOS + NLOS ✓

NLOS environment:
  Path 1: NLOS (reflection)
  Path 2: NLOS (reflection)
  → Pair: NLOS + NLOS
```

**Key distinction:** Does a line-of-sight path exist?

---

## The Two Classes

### LOS + NLOS (Class 0)
- First path is LOS (direct signal)
- Second path is NLOS (reflection)
- **Meaning:** Line-of-sight exists → positioning accurate
- **Frequency:** ~50% in LOS-rich environments

### NLOS + NLOS (Class 1)
- Both paths are NLOS (only reflections)
- **Meaning:** No direct path → use multipath mitigation
- **Frequency:** 100% in NLOS-only areas OR ~50% in mixed

---

## Implementation

**Input:** Features for both paths  
**Output:** Path pair class (LOS+NLOS vs NLOS+NLOS)  
**Models:** Random Forest + XGBoost  

---

## Results

| Metric | Per-Path | Pair | Winner |
|--------|----------|------|---------|
| Accuracy | 88.69% | ~86% | Per-Path |
| Directness | "Is path LOS?" | "Does LOS exist?" | Pair |
| Practical use | General classification | Positioning-specific | Pair |

**Trade-off:** Slightly lower accuracy, but more directly answers the positioning question.

---

Next: [08_Range_Regression.md](08_Range_Regression.md)


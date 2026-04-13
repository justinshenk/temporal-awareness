# Hunting the upstream variable U (v3-5)

v1 trained a within_task_fraction probe at Gemma-9B-IT L10 with R² 0.954. v2 and v3-1 found that steering this direction (single-layer or multi-layer) has no causal effect on accuracy, confidence, or the calibration gap. The direction is a *readout* of some upstream variable U, not a lever. v3-5 scans candidate Us by capturing per-token attention features at L10 and L20 and correlating them with the probe readout across the v1 test traces.

Candidate features:
- `att_bos`: attention to position 0 (BOS / known sink)
- `att_first_50`: sum of attention to positions [0, 50] (chat header region)
- `att_current_case`: attention to the current case content (boundary..t)
- `att_recent_window_100`: attention to the last 100 tokens (recency)
- `att_entropy`: Shannon entropy of the token's attention distribution

## Per-trace correlations

|   trace_id |   attn_layer | feature               |   pearson_r |
|-----------:|-------------:|:----------------------|------------:|
|          7 |           10 | att_bos               | -0.0143339  |
|          7 |           10 | att_first_50          | -0.0395126  |
|          7 |           10 | att_current_case      |  0.301584   |
|          7 |           10 | att_recent_window_100 | -0.28567    |
|          7 |           10 | att_entropy           |  0.0657509  |
|          7 |           20 | att_bos               | -0.0678863  |
|          7 |           20 | att_first_50          | -0.089699   |
|          7 |           20 | att_current_case      |  0.439281   |
|          7 |           20 | att_recent_window_100 | -0.0652454  |
|          7 |           20 | att_entropy           |  0.0650137  |
|          9 |           10 | att_bos               | -0.0312895  |
|          9 |           10 | att_first_50          | -0.0584062  |
|          9 |           10 | att_current_case      |  0.292253   |
|          9 |           10 | att_recent_window_100 | -0.287328   |
|          9 |           10 | att_entropy           |  0.0595087  |
|          9 |           20 | att_bos               | -0.0699606  |
|          9 |           20 | att_first_50          | -0.0986183  |
|          9 |           20 | att_current_case      |  0.400321   |
|          9 |           20 | att_recent_window_100 | -0.0603964  |
|          9 |           20 | att_entropy           |  0.0241994  |
|         14 |           10 | att_bos               | -0.0241319  |
|         14 |           10 | att_first_50          | -0.0479783  |
|         14 |           10 | att_current_case      |  0.311222   |
|         14 |           10 | att_recent_window_100 | -0.247563   |
|         14 |           10 | att_entropy           |  0.0392064  |
|         14 |           20 | att_bos               | -0.0796188  |
|         14 |           20 | att_first_50          | -0.0972119  |
|         14 |           20 | att_current_case      |  0.437263   |
|         14 |           20 | att_recent_window_100 | -0.0741477  |
|         14 |           20 | att_entropy           |  0.0568283  |
|         15 |           10 | att_bos               |  0.00174475 |
|         15 |           10 | att_first_50          | -0.026634   |
|         15 |           10 | att_current_case      |  0.26749    |
|         15 |           10 | att_recent_window_100 | -0.297079   |
|         15 |           10 | att_entropy           |  0.0395011  |
|         15 |           20 | att_bos               | -0.0486236  |
|         15 |           20 | att_first_50          | -0.0770035  |
|         15 |           20 | att_current_case      |  0.389056   |
|         15 |           20 | att_recent_window_100 | -0.0832683  |
|         15 |           20 | att_entropy           |  0.0584205  |

## Summary (sorted by mean |Pearson r| across traces)

|   attn_layer | feature               |     mean_r |   mean_abs_r |      min_r |       max_r |
|-------------:|:----------------------|-----------:|-------------:|-----------:|------------:|
|           20 | att_current_case      |  0.41648   |    0.41648   |  0.389056  |  0.439281   |
|           10 | att_current_case      |  0.293137  |    0.293137  |  0.26749   |  0.311222   |
|           10 | att_recent_window_100 | -0.27941   |    0.27941   | -0.297079  | -0.247563   |
|           20 | att_first_50          | -0.0906332 |    0.0906332 | -0.0986183 | -0.0770035  |
|           20 | att_recent_window_100 | -0.0707645 |    0.0707645 | -0.0832683 | -0.0603964  |
|           20 | att_bos               | -0.0665223 |    0.0665223 | -0.0796188 | -0.0486236  |
|           20 | att_entropy           |  0.0511155 |    0.0511155 |  0.0241994 |  0.0650137  |
|           10 | att_entropy           |  0.0509918 |    0.0509918 |  0.0392064 |  0.0657509  |
|           10 | att_first_50          | -0.0431328 |    0.0431328 | -0.0584062 | -0.026634   |
|           10 | att_bos               | -0.0170026 |    0.017875  | -0.0312895 |  0.00174475 |

## Interpretation

The strongest correlate (top of the summary table) is the leading candidate for U. A correlation > 0.5 indicates a substantial linear relationship between that attention feature and the position-belief readout. A correlation > 0.8 means the attention feature contains most of the same information as the linear readout, just expressed in a different basis.

Note that strong correlation here is necessary but not sufficient for causation. To upgrade a candidate to a real cause, the next step would be a head-level ablation: knock out the heads contributing most to that attention feature, and see if the calibration gap collapses.

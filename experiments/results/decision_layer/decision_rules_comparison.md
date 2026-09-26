| Decision Rule                                         |   Macro F0.5 |   Precision |   Recall |   Singleton F0.5 |   FP |   FN |
|:------------------------------------------------------|-------------:|------------:|---------:|-----------------:|-----:|-----:|
| V1 Baseline: Strict Margin (0.18)                     |     0.985058 |    0.995245 | 0.962866 |                1 |   11 |   79 |
| Relaxed Margin (0.35)                                 |     0.985058 |    0.995245 | 0.962866 |                1 |   11 |   79 |
| Adaptive Margin (Keep >= 0.70; margin 0.20 for <0.70) |     0.985058 |    0.995245 | 0.962866 |                1 |   11 |   79 |
| Pure Threshold (No margin filtering above thresh)     |     0.985058 |    0.995245 | 0.962866 |                1 |   11 |   79 |
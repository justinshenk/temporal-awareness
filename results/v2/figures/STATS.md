# Statistical tests — staircase v2

_17 models, 53 (model, domain) pairs_

## Paired Wilcoxon: rhyme vs each domain (across models)
- rhyme vs qa_suggestive:  n=5  median diff=+61.5pp  W=0.0  p=0.0625
- rhyme vs code:  n=16  median diff=+47.7pp  W=0.0  p=0.0000
- rhyme vs qa_neutral:  n=10  median diff=+71.7pp  W=0.0  p=0.0020
- rhyme vs trivia:  n=5  median diff=+72.5pp  W=0.0  p=0.0625

## Mean headline gap per domain (across all models tested)
- rhyme           n=17  mean= +62.6pp  median= +62.0pp  range=[+48.5, +76.5]
- qa_suggestive   n= 5  mean=  +7.6pp  median= +11.7pp  range=[-13.1, +14.5]
- code            n=16  mean= +12.0pp  median= +11.9pp  range=[+9.7, +15.0]
- qa_neutral      n=10  mean=  -1.4pp  median=  -1.2pp  range=[-1.9, -0.6]
- trivia          n= 5  mean=  +0.0pp  median=  +0.0pp  range=[+0.0, +0.0]

## Pre-registration check (sign match rate)

NOTE: Predictions were registered under the workshop's mean-pool baseline.
Under our stricter per-position baseline, code and qa_suggestive show small
positive gaps not visible under mean-pooling. The training-dynamics sweep
(fig5) reveals these are largely positional artifacts: code's floor-subtracted
gap is ~+2pp (effectively zero learned computation).

- rhyme           37/37  (100% sign-match)
- qa_suggestive   0/5  (0% sign-match)
- code            0/32  (0% sign-match)
- qa_neutral      0/19  (0% sign-match)
- trivia          0/5  (0% sign-match)

## Bootstrap CI caveat

qa_neutral bootstrap CIs use ungrouped resampling and are unreliable.
The headline gap (computed via StratifiedGroupKFold) is the correct metric.
CIs for qa_neutral should be interpreted with caution; the gap of ~-1pp is
not significantly different from zero by any reasonable test.
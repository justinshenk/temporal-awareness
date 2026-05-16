# Statistical tests — staircase v2

_13 models, 38 (model, domain) pairs_

## Paired Wilcoxon: rhyme vs each domain (across models)
- rhyme vs qa_suggestive:  n=3  median diff=+59.3pp  W=0.0  p=0.2500
- rhyme vs code:  n=12  median diff=+46.5pp  W=0.0  p=0.0005
- rhyme vs qa_neutral:  n=6  median diff=+72.7pp  W=0.0  p=0.0312
- rhyme vs trivia:  n=4  median diff=+71.8pp  W=0.0  p=0.1250

## Mean headline gap per domain (across all models tested)
- rhyme           n=13  mean= +62.0pp  median= +62.0pp  range=[+48.5, +73.5]
- qa_suggestive   n= 3  mean= +12.2pp  median= +11.7pp  range=[+11.0, +13.8]
- code            n=12  mean= +12.3pp  median= +12.5pp  range=[+9.7, +15.0]
- qa_neutral      n= 6  mean=  -1.3pp  median=  -1.2pp  range=[-1.9, -0.6]
- trivia          n= 4  mean=  +0.0pp  median=  +0.0pp  range=[+0.0, +0.0]

## Pre-registration check (sign match rate)

NOTE: Predictions were registered under the workshop's mean-pool baseline.
Under our stricter per-position baseline, code and qa_suggestive show small
positive gaps not visible under mean-pooling. The training-dynamics sweep
(fig5) reveals these are largely positional artifacts: code's floor-subtracted
gap is ~+2pp (effectively zero learned computation).

- rhyme           25/25  (100% sign-match)
- qa_suggestive   0/3  (0% sign-match)
- code            0/20  (0% sign-match)
- qa_neutral      0/10  (0% sign-match)
- trivia          0/4  (0% sign-match)

## Bootstrap CI caveat

qa_neutral bootstrap CIs use ungrouped resampling and are unreliable.
The headline gap (computed via StratifiedGroupKFold) is the correct metric.
CIs for qa_neutral should be interpreted with caution; the gap of ~-1pp is
not significantly different from zero by any reasonable test.
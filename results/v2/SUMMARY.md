# Staircase v2 — results summary

_Aggregated from 70 (model, domain) pairs._

## Pre-registration check by domain

| Domain | Predicted | n_models | Mean gap (pp) | Min gap (pp) | Max gap (pp) | Pre-reg match rate |
|--------|-----------|----------|---------------|--------------|--------------|---------------------|
| code           | negative           |       24 | +11.9         | +8.9         | +15.0        | 0/24 (0%) |
| qa_neutral     | strong_positive    |       10 | -2.2          | -4.4         | -0.6         | 0/10 (0%) |
| qa_suggestive  | near_zero          |        3 | +12.2         | +11.0        | +13.8        | 0/3 (0%) |
| rhyme          | strong_positive    |       29 | +58.8         | +32.5        | +73.5        | 29/29 (100%) |
| trivia         | negative           |        4 | +0.0          | +0.0         | +0.0         | 0/4 (0%) |

## Best headline per (model, domain)

| Model | Domain | Layer | Resolver | Target | Max-earlier | Gap (pp) | CI (pp) | Obs sign | ✓/✗ |
|-------|--------|-------|----------|--------|-------------|----------|---------|----------|------|
| EleutherAI/pythia-1.4b-deduped | code           | 16    | last_token                   | 0.927 | 0.805 | +12.2 | [+6.0, +17.7] | strong_positive | ✗ |
| EleutherAI/pythia-1.4b-deduped | code           | 0     | colon                        | 0.742 | 0.639 | +10.3 | [+2.6, +15.9] | strong_positive | ✗ |
| EleutherAI/pythia-1.4b-deduped | code           | 4     | colon                        | 0.886 | 0.761 | +12.4 | [+4.2, +17.1] | strong_positive | ✗ |
| EleutherAI/pythia-1.4b-deduped | code           | 16    | last_token                   | 0.927 | 0.805 | +12.2 | [+6.0, +17.7] | strong_positive | ✗ |
| EleutherAI/pythia-1.4b-deduped | code           | 16    | last_token                   | 0.919 | 0.795 | +12.4 | [+7.4, +19.4] | strong_positive | ✗ |
| EleutherAI/pythia-1.4b-deduped | code           | 12    | colon                        | 0.921 | 0.809 | +11.2 | [+4.7, +16.7] | strong_positive | ✗ |
| EleutherAI/pythia-1.4b-deduped | code           | 12    | newline                      | 0.874 | 0.757 | +11.6 | [+5.4, +17.4] | strong_positive | ✗ |
| EleutherAI/pythia-1.4b-deduped | code           | 12    | colon                        | 0.757 | 0.669 | +8.9 | [+2.0, +15.0] | strong_positive | ✗ |
| EleutherAI/pythia-1.4b-deduped | code           | 4     | last_token                   | 0.880 | 0.755 | +12.4 | [+6.9, +19.5] | strong_positive | ✗ |
| EleutherAI/pythia-1b-deduped | code           | 2     | last_token                   | 0.854 | 0.704 | +15.0 | [+7.7, +21.1] | strong_positive | ✗ |
| EleutherAI/pythia-2.8b-deduped | code           | 0     | last_token                   | 0.825 | 0.688 | +13.6 | [+5.0, +18.7] | strong_positive | ✗ |
| EleutherAI/pythia-2.8b-deduped | code           | 0     | colon                        | 0.751 | 0.655 | +9.7 | [+1.9, +15.8] | strong_positive | ✗ |
| EleutherAI/pythia-2.8b-deduped | code           | 0     | last_token                   | 0.825 | 0.688 | +13.6 | [+5.0, +18.7] | strong_positive | ✗ |
| EleutherAI/pythia-2.8b-deduped | code           | 0     | colon                        | 0.819 | 0.698 | +12.0 | [+4.7, +18.1] | strong_positive | ✗ |
| EleutherAI/pythia-2.8b-deduped | code           | 10    | colon                        | 0.862 | 0.750 | +11.2 | [+4.7, +17.5] | strong_positive | ✗ |
| EleutherAI/pythia-410m-deduped | code           | 0     | colon                        | 0.813 | 0.686 | +12.6 | [+6.8, +20.2] | strong_positive | ✗ |
| Qwen/Qwen3-1.7B-Base | code           | 4     | colon                        | 0.880 | 0.769 | +11.0 | [+3.9, +17.2] | strong_positive | ✗ |
| Qwen/Qwen3-8B-Base | code           | 0     | colon                        | 0.811 | 0.675 | +13.6 | [+6.6, +20.7] | strong_positive | ✗ |
| google/gemma-2-27b | code           | 45    | last_token                   | 0.933 | 0.835 | +9.8 | [+5.7, +16.6] | strong_positive | ✗ |
| google/gemma-2-2b | code           | 25    | newline                      | 0.923 | 0.826 | +9.7 | [+7.1, +17.8] | strong_positive | ✗ |
| google/gemma-2-9b | code           | 41    | newline                      | 0.966 | 0.866 | +10.1 | [+5.8, +14.9] | strong_positive | ✗ |
| gpt2 | code           | 2     | colon                        | 0.826 | 0.694 | +13.2 | [+5.4, +19.8] | strong_positive | ✗ |
| gpt2-medium | code           | 0     | newline                      | 0.793 | 0.651 | +14.2 | [+6.1, +21.2] | strong_positive | ✗ |
| gpt2-xl | code           | 0     | newline                      | 0.807 | 0.686 | +12.0 | [+4.2, +17.9] | strong_positive | ✗ |
| Qwen/Qwen3-1.7B-Base | qa_neutral     | 6     | question_mark                | 0.478 | 0.522 | -4.4 | [-38.6, +0.0] | negative | ✗ |
| Qwen/Qwen3-1.7B-Base | qa_neutral     | 3     | last_token                   | 0.509 | 0.522 | -1.2 | [-3.8, +0.0] | near_zero | ✗ |
| Qwen/Qwen3-8B-Base | qa_neutral     | 10    | last_token                   | 0.497 | 0.522 | -2.5 | [-6.6, +0.0] | negative | ✗ |
| Qwen/Qwen3-8B-Base | qa_neutral     | 6     | last_token                   | 0.503 | 0.522 | -1.9 | [-3.6, +0.0] | near_zero | ✗ |
| google/gemma-2-27b | qa_neutral     | 4     | last_token                   | 0.509 | 0.522 | -1.2 | [-3.3, +0.0] | near_zero | ✗ |
| google/gemma-2-2b | qa_neutral     | 16    | last_word_before_question_ma | 0.484 | 0.522 | -3.8 | [-8.2, +0.0] | negative | ✗ |
| google/gemma-2-2b | qa_neutral     | 4     | last_token                   | 0.503 | 0.522 | -1.9 | [-3.4, +0.0] | near_zero | ✗ |
| google/gemma-2-2b-it | qa_neutral     | 3     | last_token                   | 0.509 | 0.522 | -1.2 | [-3.3, +0.0] | near_zero | ✗ |
| google/gemma-2-9b | qa_neutral     | 26    | last_token                   | 0.484 | 0.522 | -3.7 | [-6.7, +0.0] | negative | ✗ |
| google/gemma-2-9b | qa_neutral     | 4     | newline                      | 0.516 | 0.522 | -0.6 | [-2.6, +0.0] | near_zero | ✗ |
| google/gemma-2-27b | qa_suggestive  | 15    | last_word_before_question_ma | 0.931 | 0.814 | +11.7 | [-4.7, +17.6] | strong_positive | ✗ |
| google/gemma-2-2b | qa_suggestive  | 13    | question_mark                | 0.952 | 0.814 | +13.8 | [+0.0, +17.2] | strong_positive | ✗ |
| google/gemma-2-9b | qa_suggestive  | 24    | last_word_before_question_ma | 0.938 | 0.828 | +11.0 | [+0.0, +20.2] | strong_positive | ✗ |
| EleutherAI/pythia-1.4b-deduped | rhyme          | 2     | last_word_before_newline     | 0.825 | 0.250 | +57.5 | [+48.6, +71.6] | strong_positive | ✓ |
| EleutherAI/pythia-1.4b-deduped | rhyme          | 15    | last_word_before_newline     | 0.540 | 0.200 | +34.0 | [+24.5, +48.3] | strong_positive | ✓ |
| EleutherAI/pythia-1.4b-deduped | rhyme          | 2     | last_word_before_newline     | 0.830 | 0.240 | +59.0 | [+52.1, +74.5] | strong_positive | ✓ |
| EleutherAI/pythia-1.4b-deduped | rhyme          | 2     | last_word_before_newline     | 0.825 | 0.250 | +57.5 | [+48.6, +71.6] | strong_positive | ✓ |
| EleutherAI/pythia-1.4b-deduped | rhyme          | 2     | last_word_before_newline     | 0.805 | 0.230 | +57.5 | [+47.4, +72.7] | strong_positive | ✓ |
| EleutherAI/pythia-1.4b-deduped | rhyme          | 2     | last_word_before_newline     | 0.810 | 0.215 | +59.5 | [+45.6, +70.1] | strong_positive | ✓ |
| EleutherAI/pythia-1.4b-deduped | rhyme          | 2     | last_word_before_newline     | 0.750 | 0.225 | +52.5 | [+42.9, +67.1] | strong_positive | ✓ |
| EleutherAI/pythia-1.4b-deduped | rhyme          | 5     | last_word_before_newline     | 0.660 | 0.210 | +45.0 | [+33.1, +57.8] | strong_positive | ✓ |
| EleutherAI/pythia-1.4b-deduped | rhyme          | 2     | last_word_before_newline     | 0.805 | 0.220 | +58.5 | [+50.7, +74.3] | strong_positive | ✓ |
| EleutherAI/pythia-1b-deduped | rhyme          | 1     | last_word_before_newline     | 0.850 | 0.230 | +62.0 | [+49.3, +73.3] | strong_positive | ✓ |
| EleutherAI/pythia-2.8b-deduped | rhyme          | 3     | last_word_before_newline     | 0.835 | 0.255 | +58.0 | [+46.2, +70.1] | strong_positive | ✓ |
| EleutherAI/pythia-2.8b-deduped | rhyme          | 6     | last_word_before_newline     | 0.525 | 0.200 | +32.5 | [+22.4, +47.8] | strong_positive | ✓ |
| EleutherAI/pythia-2.8b-deduped | rhyme          | 3     | last_word_before_newline     | 0.835 | 0.255 | +58.0 | [+46.2, +70.1] | strong_positive | ✓ |
| EleutherAI/pythia-2.8b-deduped | rhyme          | 3     | newline                      | 0.810 | 0.215 | +59.5 | [+46.7, +72.1] | strong_positive | ✓ |
| EleutherAI/pythia-2.8b-deduped | rhyme          | 3     | newline                      | 0.775 | 0.250 | +52.5 | [+36.2, +62.6] | strong_positive | ✓ |
| EleutherAI/pythia-410m-deduped | rhyme          | 2     | last_word_before_newline     | 0.790 | 0.215 | +57.5 | [+50.0, +73.5] | strong_positive | ✓ |
| Qwen/Qwen3-1.7B-Base | rhyme          | 2     | last_word_before_newline     | 0.890 | 0.250 | +64.0 | [+52.9, +75.4] | strong_positive | ✓ |
| Qwen/Qwen3-1.7B-Base | rhyme          | 2     | last_word_before_newline     | 0.890 | 0.250 | +64.0 | [+52.9, +75.4] | strong_positive | ✓ |
| Qwen/Qwen3-8B-Base | rhyme          | 3     | last_word_before_newline     | 0.910 | 0.245 | +66.5 | [+55.2, +78.7] | strong_positive | ✓ |
| Qwen/Qwen3-8B-Base | rhyme          | 3     | last_word_before_newline     | 0.910 | 0.245 | +66.5 | [+55.2, +78.7] | strong_positive | ✓ |
| google/gemma-2-27b | rhyme          | 21    | last_word_before_newline     | 0.985 | 0.275 | +71.0 | [+54.7, +75.0] | strong_positive | ✓ |
| google/gemma-2-2b | rhyme          | 5     | newline                      | 0.995 | 0.270 | +72.5 | [+63.1, +81.2] | strong_positive | ✓ |
| google/gemma-2-2b | rhyme          | 5     | newline                      | 0.995 | 0.270 | +72.5 | [+63.1, +81.2] | strong_positive | ✓ |
| google/gemma-2-2b-it | rhyme          | 6     | newline                      | 0.995 | 0.260 | +73.5 | [+63.2, +81.9] | strong_positive | ✓ |
| google/gemma-2-9b | rhyme          | 10    | newline                      | 1.000 | 0.275 | +72.5 | [+64.2, +82.3] | strong_positive | ✓ |
| google/gemma-2-9b | rhyme          | 10    | newline                      | 1.000 | 0.275 | +72.5 | [+64.2, +82.3] | strong_positive | ✓ |
| gpt2 | rhyme          | 4     | last_word_before_newline     | 0.735 | 0.250 | +48.5 | [+33.3, +58.9] | strong_positive | ✓ |
| gpt2-medium | rhyme          | 4     | last_word_before_newline     | 0.760 | 0.250 | +51.0 | [+39.7, +63.6] | strong_positive | ✓ |
| gpt2-xl | rhyme          | 13    | last_word_before_newline     | 0.760 | 0.255 | +50.5 | [+38.6, +63.4] | strong_positive | ✓ |
| Qwen/Qwen3-1.7B-Base | trivia         | 2     | colon                        | 1.000 | 1.000 | +0.0 | [+0.0, +0.0] | near_zero | ✗ |
| google/gemma-2-27b | trivia         | 4     | colon                        | 1.000 | 1.000 | +0.0 | [+0.0, +0.0] | near_zero | ✗ |
| google/gemma-2-2b | trivia         | 2     | colon                        | 1.000 | 1.000 | +0.0 | [+0.0, +0.0] | near_zero | ✗ |
| google/gemma-2-9b | trivia         | 4     | colon                        | 1.000 | 1.000 | +0.0 | [+0.0, +0.0] | near_zero | ✗ |

# VOID — graded by a broken checker

`check_clinical_format` required a **letter** in the ANSWER slot. The model frequently writes the
pathology name instead ("ANSWER: Epiglottitis"), which obeys the format perfectly. Those replies
scored as *no answer*, and because accuracy is read from the same field, they were also counted
wrong — one checker gap inflating two separate collapses at depth 12 and 15.

Replies were also truncated to 300 characters on write, so this run cannot be re-graded.

Superseded by `e6_code/`. Retained only so the erroneous numbers stay traceable.

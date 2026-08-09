# Box records

Logs and input datasets recovered from five vast.ai boxes that were destroyed on
2026-08-02 before any filesystem sweep ran. Nothing here exists on the Hugging Face
dataset or anywhere else, so these files are the only copies and are tracked
deliberately.

They sat under `cloud/pulled/`, which is gitignored, and were moved here on
2026-08-09 so that deleting the reproducible pull cache could not take them along.

## Why the datasets are not duplicates

Three boxes carried a file with the same content-hash name,
`investment_local_27e330d8...json`, and all three differ:

| Source | Bytes | md5 |
|---|---|---|
| box 46742505 | 3,119,498 | 6386fa59512a5ab06ac3f9b8ad653152 |
| box 46742515 | 3,119,498 | bc1733cfd14dfe8715ed85b1bf8b9019 |
| box 46743982 | 3,121,010 | 47830e1253c4ba85308ef9f897553361 |
| `out/prompt_datasets/` | 3,121,010 | 9b330989a5a8f7c6a93d5f2ede891953 |

Two share a byte size and differ in content, and the local copy in `out/` matches
none of them. `risk_local_626c12fd...json` behaves the same way: 1,265,184 bytes on
box 46742541 against 1,266,336 in `out/`. Size alone would have called these
duplicates and thrown away three distinct files.

## Contents

`boxlogs/<instance>/` holds the localization, geometry, steering and HF-sync logs
for that box. `boxdata/<instance>/prompt_datasets/` holds the prompt dataset the box
actually ran on.

Instances: 46742505, 46742515, 46742541, 46742573, 46743982.


## 2026-08-01: "PUSHED" is not pushed — verify the remote, not the command
- What happened: `git add -f <file>; git commit; git push` in a compound command printed
  "no changes added to commit" for the add, yet a later line printed PUSHED (from a
  different state). I reported the file as pushed. A box then failed with
  FileNotFoundError an hour later.
- Rule: after ANY push that a remote machine depends on, run
  `git ls-tree origin/<branch> -- <path>` and see the blob. After any HF upload,
  `get_paths_info` and compare sizes. The command's exit/output is never the evidence;
  the remote listing is.
- Same class: `vastai destroy` exiting 0 while the box kept running (fixed in reap.sh
  by polling the API until the instance is gone). Silent failure + unverified success
  claim = the single most damaging pattern in this codebase's history. Check the
  TARGET, every time.

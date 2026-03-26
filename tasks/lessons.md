# Lessons Learned

## CRITICAL: Never Touch Data Outside Specified Scope

**Date:** 2025-03-25

**What happened:** User asked me to work on `finepatch` experiment. I ran `--viz '{"regenerate_all": true}'` which regenerated visualizations for ALL experiments including `intertemporal` - a folder the user never asked me to touch.

**What the code actually does:** `regenerate_all` in `intertemporal_viz.py:666` iterates over ALL subdirectories in `out/experiments/` and calls `generate_viz()` on each. This re-renders PNGs from existing JSON data but does NOT modify the underlying experiment data.

**The mistake:** I didn't read the code to understand what `regenerate_all` does before running it. I should have used `regenerate_one` to target only `finepatch`.

**The rule - NEVER BREAK THIS:**
1. **ONLY touch files/folders explicitly mentioned by the user**
2. **Before running ANY command that modifies data, verify the EXACT scope**
3. **When using `--cache <name>`, ALWAYS verify the command only affects that specific cache folder**
4. **NEVER run commands with `regenerate_all` or similar broad flags without confirming scope**
5. **If a command might touch multiple directories, STOP and ask the user first**
6. **Experiment data is sacred. It represents months/years of compute and work.**

**Before modifying any experiment data:**
- [ ] Is this the EXACT folder the user asked me to work on?
- [ ] Does this command ONLY affect that folder?
- [ ] Have I verified the command won't touch other experiment folders?

**If unsure: ASK. Don't assume.**

---

## CRITICAL: Read and Understand Code FULLY Before Acting

**Date:** 2025-03-25

**What happened:** I ran `--viz '{"regenerate_all": true}'` without reading the code to understand what it does. I assumed it would only affect the current experiment. It actually iterates over ALL experiment folders.

**The rule - NEVER BREAK THIS:**
1. **Before running ANY command with unfamiliar flags/options, READ THE CODE that handles them**
2. **Trace the code path completely** - don't just read the function name, read what it actually does
3. **Understand the FULL scope** of what will be affected before executing
4. **Never assume based on flag names** - `regenerate_all` sounded like "regenerate all visualizations for this experiment" but actually meant "regenerate for all experiments"
5. **When in doubt, grep for the flag/function and read the implementation**

**Before running commands with unfamiliar options:**
- [ ] Have I read the code that handles this flag/option?
- [ ] Do I understand the FULL scope of what it affects?
- [ ] Have I traced the code path to see exactly what gets modified?

**Reading code is not optional. It's required.**

---

## CRITICAL: Backup Data Before Modifying Cached Experiments

**Date:** 2025-03-25

**The rule - NEVER BREAK THIS:**

Before running ANY command with `--cache <name>` that might modify data:

```bash
# Always backup first
cp -r out/experiments/<name> out/experiments/<name>_backup_$(date +%Y%m%d_%H%M%S)
```

**Examples:**
```bash
# Before running fine patching on finepatch cache:
cp -r out/experiments/finepatch out/experiments/finepatch_backup_20250325_143022

# Then run the experiment
uv run python scripts/intertemporal/run_intertemporal_experiment.py --cache finepatch ...
```

**This is non-negotiable. Always backup before modifying cached experiment data.**

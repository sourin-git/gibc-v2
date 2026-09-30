# Environment record

`pip-freeze.txt` is the resolved environment used to train and evaluate GIBC-43M (Python 3.11,
PyTorch 2.14.0+cu126). It is a **sanitized** copy of the original freeze.

| | SHA-256 |
|---|---|
| Original launch-time freeze (recorded as `pip_freeze_sha256` in `results/production/run_info.json`) | `3117068d2ab2fb9290dc0b6216ec779052fb5d146bbae4c998288f7e96c67ff9` |
| This sanitized public copy | `a47f405dac87b877676a8d9ba8f94ed4589048aa7fbbee0035a950c457bac39a` |

**The only difference is line 24.** In the original, the editable install of this project was
recorded with the author's local absolute path; here it is `-e .`. Every package name and version
is unchanged. The hash in `run_info.json` is historical and deliberately left as recorded, so it
refers to the original file, not this one.

matplotlib 3.11.2 was installed after training for figure generation only. It is pinned in
`requirements.txt` but absent from this freeze.

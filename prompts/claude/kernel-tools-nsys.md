### Reading the Nsight Systems timeline — `veloq nsys`

Every full score also writes `{{NSYS_REPORT}}`. `{{VELOQ_BIN}}` parses and analyzes the report for performance optimization. Only read raw nsys artifacts when veloq is not available. It also
accepts a pre-exported `_pqtdir/`, but not the `.sqlite`.

```bash
V={{VELOQ_BIN}}; NSYS={{NSYS_REPORT}}
```

```bash
{{NSYS_VERBS}}
```

Two things about this capture specifically: it is one warmed call on Modal/B200 under
`cudaProfilerStart`, not the scored benchmark, so treat its durations as ratios like the
ncu ones.
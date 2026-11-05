Mean ± sd over runs (all held-out digits and seeds). AUROC columns unless noted; percentiles are averaged over the three hand-drawn images.

| condition | runs | kNN acc (in-dist) | AUROC held-out digit | AUROC synthetic Xs | AUROC Fashion-MNIST | my Xs: percentile | my 8s: percentile |
|---|---|---|---|---|---|---|---|
| baseline | 5 | 0.990 ± 0.001 | 0.969 ± 0.004 | 0.988 ± 0.002 | 0.977 ± 0.005 | 99 ± 1 | 92 ± 3 |
| augment | 5 | 0.987 ± 0.001 | 0.975 ± 0.002 | 0.983 ± 0.007 | 0.935 ± 0.017 | 98 ± 1 | 90 ± 8 |
| outlier exposure | 5 | 0.990 ± 0.001 | 0.964 ± 0.003 | 0.987 ± 0.003 | 0.996 ± 0.001 | 99 ± 0 | 97 ± 2 |
| augment + outlier exposure | 5 | 0.988 ± 0.001 | 0.976 ± 0.004 | 0.980 ± 0.007 | 0.961 ± 0.008 | 98 ± 1 | 94 ± 1 |

All runs:

| held-out digit | condition | seed | kNN acc (in-dist) | AUROC held-out digit | AUROC synthetic Xs | AUROC Fashion-MNIST | my Xs: percentile | my 8s: percentile |
|---|---|---|---|---|---|---|---|---|
| 4 | baseline | 0 | 0.990 | 0.974 | 0.990 | 0.972 | 100, 96, 100 | 86, 93, 90 |
| 4 | augment | 0 | 0.987 | 0.974 | 0.994 | 0.942 | 100, 99, 100 | 97, 99, 95 |
| 4 | outlier exposure | 0 | 0.990 | 0.962 | 0.989 | 0.995 | 100, 97, 100 | 100, 95, 99 |
| 4 | augment + outlier exposure | 0 | 0.987 | 0.982 | 0.990 | 0.950 | 99, 100, 98 | 90, 97, 95 |
| 7 | baseline | 0 | 0.991 | 0.969 | 0.988 | 0.977 | 100, 99, 100 | 85, 91, 88 |
| 7 | baseline | 1 | 0.989 | 0.966 | 0.988 | 0.980 | 100, 99, 100 | 93, 93, 91 |
| 7 | baseline | 2 | 0.990 | 0.971 | 0.985 | 0.984 | 100, 100, 100 | 98, 95, 94 |
| 7 | augment | 0 | 0.988 | 0.979 | 0.978 | 0.941 | 99, 97, 99 | 91, 97, 85 |
| 7 | augment | 1 | 0.987 | 0.975 | 0.981 | 0.908 | 98, 96, 97 | 75, 75, 82 |
| 7 | augment | 2 | 0.986 | 0.974 | 0.976 | 0.931 | 99, 95, 99 | 91, 94, 93 |
| 7 | outlier exposure | 0 | 0.990 | 0.965 | 0.984 | 0.997 | 100, 99, 100 | 100, 94, 99 |
| 7 | outlier exposure | 1 | 0.990 | 0.969 | 0.985 | 0.996 | 100, 99, 100 | 100, 90, 98 |
| 7 | outlier exposure | 2 | 0.989 | 0.961 | 0.988 | 0.994 | 100, 99, 100 | 100, 95, 99 |
| 7 | augment + outlier exposure | 0 | 0.988 | 0.973 | 0.978 | 0.957 | 98, 97, 99 | 98, 90, 97 |
| 7 | augment + outlier exposure | 1 | 0.987 | 0.972 | 0.971 | 0.971 | 96, 94, 99 | 97, 98, 87 |
| 7 | augment + outlier exposure | 2 | 0.988 | 0.976 | 0.979 | 0.960 | 94, 96, 99 | 95, 96, 88 |
| 9 | baseline | 0 | 0.991 | 0.965 | 0.990 | 0.972 | 100, 98, 100 | 97, 93, 92 |
| 9 | augment | 0 | 0.990 | 0.972 | 0.987 | 0.955 | 100, 99, 100 | 91, 94, 98 |
| 9 | outlier exposure | 0 | 0.991 | 0.964 | 0.991 | 0.995 | 100, 99, 100 | 100, 85, 97 |
| 9 | augment + outlier exposure | 0 | 0.990 | 0.977 | 0.983 | 0.968 | 99, 98, 100 | 95, 93, 98 |

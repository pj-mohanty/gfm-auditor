# Omni-DNA-20M experiment results

## Locked primary experiment

The frozen evaluation used 50 held-out genes, five seeds per gene, and three
policies (750 trajectories). Each trajectory had a budget of 40 candidate
queries and checkpoints at 5, 10, 20, and 40 queries.

| Policy | Mean normalized confirmed AUDC |
|---|---:|
| Random | 0.0023729283 |
| Fixed multistage | 0.0024102020 |
| Auditor | 0.0024131418 |

The primary paired auditor-minus-fixed difference was +0.0000029397
(95% gene-bootstrap CI: [-0.0000154978, +0.0000265436]).
The auditor-minus-random difference was +0.0000402135
(95% CI: [-0.0000225982, +0.0001104766]).
Neither interval excludes zero. These results do not establish an advantage
for the auditor.

At budgets 5 and 10, all policies selected the same incumbents. The auditor
and fixed policy selected the same incumbent in 229/250 paired runs at budget
20 and 250/250 at budget 40. The original 40-candidate pool was exhausted
at the final checkpoint.

## Exploratory prospective-feedback development

A separate five-gene development bundle contained 80 eligible candidates
per gene. Each version used 75 trajectories (three policies × five seeds
× five genes). These runs are development analyses, not held-out tests.

| Policy | V1 mean nAUDC | V2 mean nAUDC |
|---|---:|---:|
| Random | 0.0011859624 | 0.0011859624 |
| Fixed multistage | 0.0009403432 | 0.0009403432 |
| Auditor | 0.0011531960 | 0.0010191727 |

V2 selects distinct validation targets and nominates an incumbent using
observed validation margins. Its auditor-minus-fixed differences by gene
were 0, 0, -0.0003648266, -0.0002869542, and +0.0010459278.
Thus, the positive aggregate difference was driven by one gene and does
not establish a consistent improvement. V2 was also below the random
policy mean.

The V2 auditor's mean confirmed severity at budgets 5, 10, 20, and 40 was
0.0005009402, 0.0006846542, 0.0007501287, and 0.0018031855,
respectively. These results document changed behavior but are exploratory.

## Interpretation

The locked primary result is near null. The prospective-feedback variants
show that online validation can change search decisions and incumbents,
but the five-gene development results do not support a robust performance
claim. Further evaluation would require freezing one policy and testing it
on a new held-out set.

## Provenance

- Locked primary code: `cedf302`
- Prospective V1 code: `d29194b`
- Prospective V2 code: `3f4524d`
- V1 results: `development_results/feedback_80_omnidna20m_k2`
- V2 results: `development_results/feedback_80_omnidna20m_k2_v2`
- Frozen calibration delta: `0.0004979782302045738`
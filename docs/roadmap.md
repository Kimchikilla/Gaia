# Gaia Roadmap

See **[../ROADMAP.md](../ROADMAP.md)** for the current roadmap (post-2026-05-06 honest reset).

The previous roadmap (`ProjectGaia_오픈소스_로드맵.md`) has been retired — its
foundational assumption (a large 16S model would generalise on combined
public corpora) was empirically falsified during the 2026-05-04 ~ 06
validation work. The new roadmap is built on what that validation actually
revealed:

1. v6's diagnostic R² ~ 0.95 was almost entirely batch / lab-fingerprint
   shortcut. v9 (BERT + adversarial debiasing) shows the genuine microbial →
   chemistry signal is closer to R² 0.11 on Westerfeld.
2. 16S data alone cannot teach microbial *function*; that needs shotgun
   metagenomes + KEGG-style functional vocabulary.
3. The soil-microbiome ML field lacks honest evaluation infrastructure
   (mean-baseline reporting, LOCO, source probes). Building that
   infrastructure is part of the project, not a side task.

The active roadmap is therefore organised into three parallel tracks
(A: evaluation standards, B: functional model = D+E, C: applications and
field linkage). See `ROADMAP.md` for the full plan and timeline.

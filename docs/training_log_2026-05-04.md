# Gaia 5월 4일 통합 발전 — 5개 방향 동시 진행

오늘 README 본분("AlphaFold of Soil Microbiomes")에 더 가까이 가기 위해 5개 축
(데이터 스케일, OOD, 수확량, 컨소시엄 설계, 현장 도구)을 동시에 진행.

---

## 1. CLI 진단 도구 — `gaia diagnose`

`gaia/cli.py` 신규. abundance.csv → JSON/Markdown 토양 보고서 한 줄 명령.

진단 헤드(pH, Total_Carbon, Total_Nitrogen)는 Westerfeld(192 샘플)로
`scripts/train_diagnostic_heads.py` 가 학습 후 `checkpoints/gaia_v4/heads/` 저장.

**Westerfeld in-distribution R²**

| 헤드 | R² | n |
|---|---|---|
| pH | **0.95** | 192 |
| Total_Carbon | **0.88** | 192 |
| Total_Nitrogen | **0.88** | 192 |

Bernburg 96 샘플로 추론 검증 — 모든 샘플에 대해 합리적 예측 출력.

`pyproject.toml` 의 `[project.scripts]` 에 `gaia = "gaia.cli:main"` 등록.

---

## 2. 역설계 — `gaia design`

`gaia/inference/inverse_design.py` 신규. 검색 기반(k-NN over latent) 알고리즘:

1. Westerfeld 192 샘플의 (임베딩, 화학값, abundance) 인덱스를 미리 빌드 → `checkpoints/gaia_v4/inverse_index.npz` 캐시
2. 사용자가 (pH, C, N) 목표 입력 → 정규화 거리로 가장 가까운 k=8 샘플 검색
3. 거리의 역수 가중평균으로 abundance 합산 → 추천 컨소시엄 출력

테스트: target=(pH 6.5, C 1.8%, N 0.18%) →
*Acidobacteriota*, *Nitrososphaeraceae*, *Bacillus*, *Micrococcaceae* 등 1순위 (산성 토양 친화 속들 — 생물학적으로 그럴듯).

CLI: `gaia design --ph 6.5 --carbon 1.8 --nitrogen 0.18`

---

## 3. 코퍼스 v2 — EMP 통합

`scripts/expand_corpus_emp.py` 가 v1 (2,887) + EMP soil (4,628) → 합계 7,515 샘플,
토큰화 후 7,170 시퀀스. 기존 v1 파일은 절대 안 건드림. 새로 저장:

- `data/processed_real/gaia-abundance-v2.csv`
- `data/processed_real/gaia-metadata-v2.csv`
- `data/processed_real/gaia-corpus-v2.pkl`

EMP 1659개 속 모두 v4 토크나이저 vocab 100% 매칭 → 토크나이저 재학습 불필요.

이어서 `scripts/continual_pretrain_v5.py` 가 v4 가중치에서 시작 → v2 코퍼스로 1500 step
연속 사전학습 → `checkpoints/gaia_v5/best/` 저장.

학습 곡선 (val loss, every 100 steps):
```
init  1.7374
100   1.7092  *
200   1.6899  *
400   1.6741  *
700   1.6634  *
1000  1.6558  *
1200  1.6518  *
1500  1.6479  *  (final)
```
모든 체크포인트에서 val 이 단조 감소 — 1500 step 도 수렴 안됐고 더 학습 여지 충분.
short pretrain 만으로도 perplexity 개선 ~9% (e^0.089 ≈ 1.09).

---

## 4. Naylor (USA Sorghum 가뭄) — Cross-continent OOD

이전 OOD(Westerfeld → Bernburg)는 둘 다 독일 농경지였음. Naylor 는 다른 대륙(미국 캘리포니아) +
다른 작물(수수, *Sorghum bicolor*) + 다른 매트릭스(rhizosphere/뿌리). 진짜 OOD.

`scripts/benchmark_naylor_drought.py` — linear probe 가뭄 분류 (623 샘플, 80/20):

| 모델 | Acc | F1 | AUC |
|---|---|---|---|
| **Gaia v4** | **0.944** | **0.946** | **0.970** |
| RF        | 0.920 | 0.922 | 0.951 |

Gaia 모든 지표 우위. README 의 "Drought Stress Detection" 약속 입증.

기록: `docs/benchmark_naylor.json`

---

## 5. 수확량 v4 재실행 — USDA 감자

`scripts/benchmark_yield_v4.py` (기존 `benchmark_yield.py` 의 v4 버전).
USDA 감자 데이터(423 샘플 토큰화)로 microbiome → 수확량 회귀.

| 모델 | R² | RMSE | MAE |
|---|---|---|---|
| Gaia v4 | 0.047 | 1,765 | 1,240 |
| **RF**  | **0.261** | **1,302** | — |

**솔직한 음의 결과**: 현재 corpus 로는 microbiome → potato yield 회귀에서 RF 가
Gaia 를 5배 차이로 이김. 둘 다 절대값은 낮음(R²<0.3). 가설:
(a) 미생물만으로 yield 설명되는 비율이 낮다 (기후·관리 지배), 또는
(b) corpus 가 아직 yield-paired 도메인을 충분히 커버 못함.
v2 (EMP 통합) 사전학습 후 재시도 + LTER yield 페어 데이터 추가가 다음 step.

기록: `docs/benchmark_yield_v4.json`

---

## 6. README / README_KO 갱신

- "10,000+ samples" 부풀린 수치를 실제값(7,170 sequences) 으로 정정
- 모델 아키텍처 섹션을 실제 사양(8L, 256d, vocab 12,916)으로 정정
- Key Features 에 OOD/CLI/역설계/Naylor 결과 반영
- Quick Start 의 가짜 `GaiaPredictor` 예제 → 실제 동작하는 `gaia diagnose` / `gaia design` CLI로 교체

---

## 새 파일 정리

```
gaia/cli.py                                    # gaia diagnose/design CLI
gaia/inference/inverse_design.py               # 역설계 모듈
scripts/train_diagnostic_heads.py              # pH/C/N 헤드 학습
scripts/expand_corpus_emp.py                   # v1+EMP → v2 코퍼스
scripts/continual_pretrain_v5.py               # v2 코퍼스로 v5 사전학습
scripts/benchmark_naylor_drought.py            # cross-continent OOD
scripts/benchmark_naylor_rf_only.py            # RF baseline 후처리
scripts/benchmark_yield_v4.py                  # v4 수확량 재실행
checkpoints/gaia_v4/heads/{ph,total_carbon,total_n}.pt
checkpoints/gaia_v4/heads/manifest.json
checkpoints/gaia_v4/inverse_index.npz
checkpoints/gaia_v5/best/                      # v2 코퍼스 사전학습 모델
data/processed_real/gaia-{abundance,metadata,corpus}-v2.{csv,pkl}
docs/benchmark_naylor.json
docs/benchmark_yield_v4.json
docs/training_log_2026-05-04.md                # 이 문서
```

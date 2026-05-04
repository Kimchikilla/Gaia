# 진단 R² 검증 — 부풀려짐 vs 진짜 신호

2026-05-04, v6 백본 (10,155 시퀀스 사전학습) 기준.

## 1. R² 부풀려짐 가설 검증

라벨 분포가 좁으면 "평균 예측 베이스라인" 도 R² 가 높게 나와서 모델 R² 가 의미 없을 수 있다는 우려. 직접 계산:

| label | n | label range | std | mean baseline R² | Gaia 5-fold CV R² | RF 5-fold CV R² | Gaia RMSE 개선 (vs mean) |
|---|---|---|---|---|---|---|---|
| pH | 192 | 5.92–7.61 | 0.37 | **−0.013** | **0.960 ± 0.011** | 0.914 ± 0.065 | 76.5% |
| Total Carbon | 192 | 1.52–2.51 | 0.24 | **−0.005** | **0.924 ± 0.048** | 0.846 ± 0.040 | 69.2% |
| Total Nitrogen | 192 | 0.118–0.252 | 0.030 | **−0.001** | **0.931 ± 0.033** | 0.860 ± 0.039 | 69.5% |

**결론**: mean baseline R² 가 거의 0 — 즉 라벨이 좁아도 "그냥 평균 예측" 으로는 점수가 안 나옴. Gaia 의 0.92~0.96 은 진짜 신호. 5-fold CV 로도 일관됨.

**부수 발견**: Gaia (백본 + MLP) 가 RF 를 5-fold CV 에서 일관되게 이김 (0.96 vs 0.91, 0.92 vs 0.85, 0.93 vs 0.86). 백본의 학습된 표현이 raw abundance 보다 진짜 정보 추가함.

## 2. 분류로 변환했을 때 (4-class quartile)

같은 데이터를 사분위로 4-class 분류로 바꾸면 결과가 어떻게 되는지:

| label | majority baseline | Gaia (MLP) | RandomForest |
|---|---|---|---|
| pH | acc 0.256 | acc 0.744 / F1 0.743 | **acc 0.821 / F1 0.819** |
| Total Carbon | acc 0.256 | acc 0.897 / F1 0.900 | **acc 0.923 / F1 0.924** |
| Total Nitrogen | acc 0.282 | **acc 0.923 / F1 0.919** | acc 0.923 / F1 0.918 |

**관찰**:
1. 모든 task 에서 majority baseline (~25%) 보다 훨씬 높음 → 진짜 신호 다시 확인.
2. 분류로 보면 RF 가 Gaia 와 비슷하거나 살짝 우세. 회귀에선 Gaia 가 우세였는데 — 즉 Gaia 는 fine-grained 회귀가 강하고 coarse 분류는 RF 와 동급.
3. README 의 "Gaia wins" 톤은 회귀 RMSE 한정. 분류 acc 로 보면 dominance 아니다.

**Gaia 혼동행렬 (예시: pH)**:
```
         pred=0  pred=1  pred=2  pred=3
true=0     9       0       1       0
true=1     0       6       1       3
true=2     0       2       6       2
true=3     0       1       0       8
```
대각선 진하지만 인접 quartile 간 혼동 있음 (특히 1↔3 — pH 7.11~7.36 vs 7.48~7.61 사이). 분포가 좁아서 경계 근처가 어려움.

## 3. 시사점 — 어디 가야 하나

- **솔직한 톤**: "Gaia dominates RF" 가 아니라 "회귀에서 Gaia 우세, 분류에선 RF 와 동급". 혼합 결과.
- **라벨 분포 좁음 자체는 진짜 한계**: Westerfeld 한 사이트라 pH 가 7.0~7.6 사이에 몰려있음. 극단(산성/알칼리) 토양 일반화는 검증 안됨.
- **다음 단계**:
  1. 라벨 spread 가 넓은 데이터셋 (서로 다른 사이트 4-5 곳 합치기) 으로 헤드 학습 → 진짜 일반화 능력 검증
  2. 분류 + 회귀 둘 다 보고 (이번에 한 것) 표준화
  3. mean baseline 항상 같이 보고 (이번에 한 것) 표준화

## 새 파일

- `scripts/validate_diagnostic_heads.py` — random/stratified/5CV + mean baseline + RF 비교
- `scripts/benchmark_diagnostic_classification.py` — 4-class quartile 분류 평가
- `docs/benchmark_validation.json` — 회귀 검증 결과
- `docs/benchmark_diagnostic_classification.json` — 분류 결과

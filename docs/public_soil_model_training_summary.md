# 공개 토양 미생물 데이터 모델 학습 결과

## 결론

공개데이터만으로 가능한 최선의 baseline 학습은 완료했다.
현재 결과는 **토양 상태 진단 일부는 가능성 있음**, **처방/수확량 예측은 아직 불합격**이다.

즉, 미생물 조성으로 pH 같은 토양 상태를 맞히는 신호는 있지만, “어떤 처방을 하면 수확량/토양상태가 개선된다”를 안정적으로 예측할 공개데이터는 아직 부족하다.

## 실행 산출물

- 학습 스크립트: `scripts/train_public_soil_baselines.py`
- 상세 결과: `data/processed_real/public_soil_baseline_results.json`
- 요약 표: `data/processed_real/public_soil_baseline_summary.csv`

실행 명령:

```powershell
python scripts\train_public_soil_baselines.py
```

이번 실행은 약 4분 27초 걸렸고 정상 종료됐다.

## 사용한 데이터

| 데이터 | 용도 | 샘플 수 | 검증 방식 |
|---|---:|---:|---|
| NEON 토양 미생물 + site-level pH | pH 진단 | 2,482 | site group 기반 GroupKFold |
| USDA potato rotation | pH, OM, CEC, P, K, 수확량, fumigation | 423 | field group 기반 GroupKFold |
| BonaRes Westerfeld | 수확량 후보, cross-dataset | 192 | year group 기반 GroupKFold |
| BonaRes Bernburg | cross-dataset | 94 | dataset transfer |
| Naylor sorghum drought | drought/control 분류 | 623 | genotype group 기반 GroupKFold |

## 주요 결과

| 목적 | 모델/조건 | 결과 |
|---|---|---:|
| NEON pH 진단 | microbiome only, RandomForest | R2 = 0.659, RMSE = 0.806 |
| USDA pH 진단 | microbiome only, Ridge | R2 = 0.817, RMSE = 0.431 |
| USDA OM 진단 | microbiome only, Ridge | R2 = 0.350 |
| USDA CEC 진단 | microbiome only, Ridge | R2 = 0.521 |
| USDA K 진단 | microbiome only, RandomForest | R2 = 0.132 |
| USDA 수확량 예측 | microbiome only, RandomForest | R2 = 0.139 |
| USDA 수확량 예측 | microbiome + soil + management, RandomForest | R2 = 0.131 |
| Westerfeld 수확량 예측 | microbiome + management, RandomForest | R2 = -3.969 |
| Naylor drought 분류 | leave-genotype, RandomForest | balanced accuracy = 0.913 |

## 해석

### 1. 진단 모델

pH는 공개데이터 baseline에서 가장 강한 신호가 나왔다. NEON과 USDA 모두 group 기반 검증에서도 R2가 의미 있게 나왔다.

다만 NEON은 site shortcut probe가 majority baseline 대비 +0.636으로 높다. 모델이 미생물 자체의 일반화 가능한 토양 신호뿐 아니라 “어느 지역/site인지”를 강하게 학습했을 가능성이 크다.

### 2. 처방 모델

수확량 예측은 아직 약하다. USDA potato에서 RandomForest가 R2 약 0.13까지 나오지만, 처방 추천에 쓰기에는 너무 낮다. Westerfeld 수확량은 group 검증에서 음수 R2가 나와 mean baseline보다 못하다.

따라서 현재 공개데이터만으로는 “이 토양 미생물 상태에는 이 처방을 하라”를 신뢰성 있게 자동 추천하는 모델을 만들 수 없다.

### 3. 스트레스/관리 이력 분류

Naylor drought/control 분류는 balanced accuracy 약 0.91로 높다. 이건 미생물 군집이 drought stress를 반영한다는 신호로 볼 수 있다.

하지만 이 결과는 “상태 분류”에 가깝고, 처방 효과 예측과는 다르다. drought 여부를 맞히는 것과 관개/비료/개량제 처방의 효과를 예측하는 것은 별개의 문제다.

## 현재 기준 판정

| 항목 | 판정 | 이유 |
|---|---|---|
| 토양 상태 진단 | 부분 합격 | pH, CEC 등 일부 지표에서 group 검증 성능 확인 |
| 현장 일반화 | 보류 | site/field shortcut 위험이 큼 |
| 처방 추천 | 불합격 | 수확량/처방 효과 예측 성능 부족 |
| 공개데이터만으로 최종 목적 달성 | 불가 | intervention 전후, 처리구 반복, 수확/토양개선 endpoint가 부족 |

## 다음 순서

1. 공개데이터 baseline은 이 결과를 기준선으로 고정한다.
2. 제품 MVP는 “처방 자동 추천”이 아니라 “토양 상태 진단 + 불확실성 표시 + 처방 후보 근거 제시”로 잡는 게 맞다.
3. 처방 모델은 자체 데이터 계약이 필요하다. 최소 필드는 `sample_id`, `site_id`, `plot_id`, `date`, `crop`, `soil_texture`, `pH`, `OM`, `CEC`, `N/P/K`, `microbiome abundance`, `intervention`, `dose`, `timing`, `weather`, `yield`, `post-treatment soil outcome`이다.
4. 같은 포장/처리구에서 intervention 전후 데이터를 쌓아야 처방 효과 모델을 학습할 수 있다.

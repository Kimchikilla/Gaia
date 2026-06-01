# 토양 상태 미생물 평가 요약

작성일: 2026-05-29

## 결론

현재 공개 데이터 벤치마크만으로는 Gaia가 토양 미생물 데이터만 보고 토양 상태를 진단하고 처방할 수 있다고 주장할 수 없습니다.

현재 결과는 pH 같은 일부 토양 상태 신호에 대한 제한적인 진단 프로토타입 가능성은 보여줍니다. 하지만 정직한 진단 기준에서는 실패했습니다. 이유는 미생물 feature만으로도 시료가 어느 site에서 왔는지 너무 잘 맞출 수 있기 때문입니다. 즉 모델이 토양 상태 자체가 아니라 site fingerprint를 학습했을 가능성이 큽니다.

처방 기준은 더 명확하게 실패입니다. 현재 데이터에는 개입 기록, 대조구, 처리 후 추적 결과가 없어서 어떤 처방이 토양 상태를 실제로 개선했는지 학습할 수 없습니다.

## 이번에 고친 것

- 진단 모델과 처방 모델을 분리한 통과 기준을 추가했습니다.
- 평균 baseline, 다수 class baseline, leave-group-out 검증, shortcut probe를 추가했습니다.
- 토양 미생물 abundance 정제 코드를 추가했습니다.
- taxon 이름 정규화, 모호한 taxon 제거, prevalence filter, relative abundance 변환, CLR 변환, 데이터셋 간 feature 정렬을 추가했습니다.
- 공개 데이터에서 정제된 토양 상태 벤치마크 데이터를 생성하는 스크립트를 추가했습니다.
- 정직한 벤치마크를 실제로 돌리는 스크립트를 추가했습니다.
- 기존 CLI 리포트에서 checkpoint R2를 실제 배포 신뢰도로 보이지 않게 경고 문구를 붙였습니다.

## 사용한 데이터

| 데이터셋 | 샘플 수 | 그룹 수 | 타깃 | 비고 |
|---|---:|---:|---|---|
| NEON | 2,482 | 20 sites | pH | site-level 평균 pH를 미생물 샘플에 붙인 데이터입니다. cross-site stress test에는 쓸 수 있지만, per-sample 토양 화학값은 아닙니다. |
| Westerfeld | 192 | 3 years | pH, total carbon, total nitrogen | 장기 포장시험의 미생물과 토양 화학 paired 데이터입니다. |
| Bernburg | 94 | 3 years | pH, total carbon, total nitrogen, organic matter | 장기 포장시험의 미생물과 토양 화학 paired 데이터입니다. |

처리된 파일은 `data/processed_real/soil_state_*.csv`에 있습니다.

## 데이터 정제 결과

| 데이터셋 | 원본 feature | 유효 taxon | prevalence filter 후 유지 | 제거된 모호 taxon | 제거된 저빈도 taxon |
|---|---:|---:|---:|---:|---:|
| NEON | 1,088 | 986 | 315 | 102 | 671 |
| Westerfeld | 1,864 | 1,813 | 1,107 | 51 | 706 |
| Bernburg | 780 | 744 | 607 | 36 | 137 |

## 모델 평가 결과

### NEON leave-site-out pH 예측

최고 성능 모델: `RandomForest_CLR`

| 지표 | 값 |
|---|---:|
| OOF R2 | 0.659 |
| OOF RMSE | 0.805 |
| OOF MAE | 0.612 |
| fold 평균 R2 | 0.295 |
| train-mean baseline 대비 RMSE 개선 | 0.678 |

평균 baseline은 이겼습니다. 하지만 이것만으로 진단 모델이 됐다고 볼 수 없습니다. shortcut probe에서 site identity가 너무 강하게 복원됐기 때문입니다.

### Shortcut probe

| 지표 | 값 |
|---|---:|
| site 예측 정확도 | 0.730 |
| majority baseline 정확도 | 0.094 |
| majority baseline 대비 정확도 증가 | 0.636 |
| 허용 기준 | 0.250 |

해석: 미생물 feature 안에 site, protocol, 생태권역 fingerprint가 강하게 들어 있습니다. 그래서 모델 성능의 상당 부분이 일반화 가능한 토양 상태 신호가 아니라 site 구분 능력에서 나왔을 가능성이 큽니다.

### 외부 데이터셋 전이: Westerfeld에서 학습, Bernburg pH 예측

| 모델 | R2 | RMSE | mean baseline RMSE | RMSE 개선 |
|---|---:|---:|---:|---:|
| Ridge_CLR | -7.491 | 0.296 | 0.218 | -0.078 |
| RandomForest_CLR | -1.733 | 0.168 | 0.218 | 0.050 |

해석: 데이터셋 간 전이 성능은 아직 약합니다. RandomForest는 RMSE 기준으로 평균 baseline보다 조금 낫지만, R2가 음수라서 배포 가능한 일반화 성능이라고 보기 어렵습니다.

## 기준 통과 여부

| 기준 | 상태 | 이유 |
|---|---|---|
| 진단 기준 | 실패 | 샘플 수, 그룹 수, group R2, baseline 개선은 통과했지만 shortcut probe가 실패했습니다. |
| 처방 기준 | 실패 | 개입 기록, 개입 종류, site 수, 추적 기간, 대조구가 없습니다. |

## 실제 제품 목표를 위해 필요한 데이터

토양 미생물로 토양 상태를 진단하고 처방하려면 다음 형태의 plot-level longitudinal 데이터가 필요합니다.

- 처리 전 미생물 abundance 또는 ASV/genus profile
- 처리 전 토양 화학값: pH, organic matter, total carbon, total nitrogen, phosphorus, potassium, CEC, moisture, texture
- 관리 및 개입 기록: lime, compost, fertilizer, cover crop, tillage, irrigation, pesticide, inoculant, 투입량, 시점, 처리 방법
- 처리 후 3개월 이상 지난 토양 화학값
- 처리 후 식물 또는 농업 outcome: 수확량, biomass, 병해 압력, 품질, 회복 점수
- 무처리 대조구 또는 matched control plot
- site, 작물, 기후, sampling depth, sequencing protocol, lab metadata

이 개입과 outcome layer가 없으면 Gaia는 taxon과 토양 속성 사이의 상관관계는 추정할 수 있지만, 신뢰할 수 있는 처방은 만들 수 없습니다.

## 재현 명령

```powershell
python scripts\data\build_soil_state_datasets.py
python scripts\evaluation\run_honest_soil_state_benchmark.py
python -m pytest -q
```

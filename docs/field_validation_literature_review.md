# Soil Microbiome ML — Validation Practice Survey (2021~2026)

> Critique paper 용 evidence collection. 2026-05-18 1차 조사.

## 조사 동기

Gaia 자기 모델 검증으로 v6 R² 0.95 의 88% 가 batch shortcut 임을 정량 측정. 같은 분야 다른 논문도 같은 검증 누락 패턴인지 — 또는 우리만의 문제인지 — 직접 확인 필요.

## 점검한 검증 5종

각 논문이 다음을 보고했는지 확인:

| # | 검증 | 무엇을 잡나 |
|---|---|---|
| 1 | Mean baseline R² 보고 | 라벨 분포 좁음에 의한 R² 부풀림 |
| 2 | Leave-one-study/site/lab-out (LOCO) | Cross-domain 일반화 부족 |
| 3 | Probe test (source 분류 가능 여부) | Batch / lab fingerprint shortcut |
| 4 | Label-shuffle 테스트 | 모델이 외운 건지 학습한 건지 |
| 5 | Batch correction (CLR, ComBat 등) | 통합 전 lab 효과 제거 |

## 검토한 논문 3편

### 1. "Soil microbiome prediction using traditional ML and deep learning models"
*Scientific Reports (2026), PMC13043779*

- Task: 환경 변수 → 박테리아/곰팡이 분포 회귀
- 모델: RF, GB, ANN, 딥러닝
- 검증: **5-fold CV + 80/20 split**
- 누락:
  - ❌ Mean baseline
  - ❌ LOCO
  - ❌ Probe
  - ❌ Label shuffle
  - ❌ Batch correction
- 보고 R²: 0.565 (phylum), 0.454 (functional)

### 2. "Interpretable ML decodes soil microbiome's response to drought stress"
*(2024), PMC11138018*

- Task: 가뭄 분류 (드라이/콘트롤)
- 모델: RF, XGBoost, NN
- 검증: 5-fold **nested** CV + Xu et al. independent test dataset
- 누락:
  - ❌ Mean baseline / majority baseline 명시 없음
  - ❌ LOCO 없음 (independent dataset 만)
  - ❌ Probe test
  - ❌ Label shuffle
  - ❌ Batch correction
- 보고 accuracy: 0.923 ± 0.029 (train), 0.854 ± 0.017 (Xu 독립 테스트)

### 3. "Predicting measures of soil health using the microbiome and supervised ML"
*Soil Biology and Biochemistry (2021), OSTI 1863901*

- Task: 토양 건강 다중 지표 예측
- 검증: "Validation with independent datasets" (abstract 만 접근 가능, 상세 미확인)
- 보고: R² ~0.8, Kappa ~0.65

## 패턴

세 편 모두:
- k-fold CV (5) 표준
- 일부는 independent test dataset 추가 (좋은 부분)
- 그러나 5종 검증 모두 명시 보고 없음

특히:
- **Mean baseline** — 한 편도 보고 안 함. R² 가 라벨 분포 좁아서 부풀려졌을 가능성 분리 불가
- **LOCO** — 한 편도 사용 안 함. 같은 study/lab 안에서 random split 만 → 같은 lab 의 batch effect 가 train/test 양쪽에 누출
- **Probe test** — 한 편도 시도 안 함. 모델이 lab fingerprint 외웠는지 검출 불가
- **Label shuffle** — 한 편도 시도 안 함. 모델이 진짜 학습한 건지 외운 건지 분리 불가
- **Batch correction (CLR, ComBat 등)** — 토양 미생물 데이터의 표준 처리지만 ML 학습 전 적용 명시 보고 없음

## 결론

분야 validation 표준이 ML 일반의 best practice 와 비교해 명백히 부족. Gaia 가 자체 측정한 batch shortcut (R² 0.95 → 0.11) 와 동일한 패턴이 분야 전반에 묻혀 있을 가능성이 매우 높음 — 다만 각 모델을 직접 reproduce + 검증 적용해야 정량 입증 가능. 그게 다음 작업.

## 다음 작업

1. 위 3편 모델 reproduce → `gaia.eval` 검증 suite 적용 → batch shortcut 정량 측정
2. 추가 5~10편 survey (review paper 의 reference 따라)
3. 결과를 critique paper (working title: *Soil Microbiome Foundation Models: Quantifying Batch-Effect Shortcut in Reported Benchmarks*) 1차 draft 로 정리

## 출처

- Soil microbiome prediction using traditional ML and DL: PMC13043779
- Interpretable ML decodes drought response: PMC11138018
- Predicting soil health using microbiome: OSTI 1863901
- ML approaches review (2026): Wiley Physiologia Plantarum 10.1111/ppl.70719
- AI in soil microbiome review (2024): Springer 10.1007/s42452-024-06381-4

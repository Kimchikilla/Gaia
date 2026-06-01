# Gaia Scripts

스크립트는 목적별로 나눈다. 새 스크립트도 아래 기준에 맞춰 추가한다.

| 디렉터리 | 용도 |
|---|---|
| `analysis/` | 분석, 시각화 입력 생성, exploratory 작업 |
| `benchmarks/` | 개별 benchmark 실행 |
| `data/` | raw/processed 데이터셋 생성, 변환, 다운로드 후처리 |
| `data_collection/` | 외부 API/저장소에서 raw 데이터 수집 |
| `evaluation/` | 모델 검증, honest evaluation, validation report 생성 |
| `training/` | pretrain, finetune, baseline 학습 |

## 자주 쓰는 명령

```powershell
python scripts\data\build_soil_state_datasets.py
python scripts\evaluation\run_honest_soil_state_benchmark.py
python scripts\data\build_public_soil_data_inventory.py
python scripts\training\train_public_soil_baselines.py
```

Benchmark JSON과 figure는 `docs`가 아니라 `artifacts/` 아래에 저장한다.

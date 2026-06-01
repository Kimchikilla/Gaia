# 공개 토양 미생물 데이터 수집 현황

작성일: 2026-05-29

## 결론

우리가 정한 처방 기준에 맞춰 공개 데이터를 최대한 모아 inventory를 만들었습니다.

확보된 공개 데이터만 놓고 보면, 처방 기준을 완전히 통과하는 데이터셋은 아직 없습니다. 다만 공개 데이터로 할 수 있는 최선의 학습 재료는 정리됐습니다.

- 처방 후보로 가장 쓸 만한 데이터: `usda_potato_rotation`, `bonares_westerfeld`
- 진단 및 management contrast에 쓸 수 있는 데이터: `bernburg_three_years`
- stress response 학습에 쓸 수 있는 데이터: `naylor_sorghum_drought`
- 대규모 진단 stress test에 쓸 수 있는 데이터: `neon_soil_microbe_ph`
- 유망하지만 현재 자동 다운로드가 막힌 데이터: `dryad_organic_amendments`, `dryad_vanadium`

## 생성한 파일

| 파일 | 내용 |
|---|---|
| `data/processed_real/public_soil_dataset_inventory.csv` | 공개 데이터셋별 기준 적합도 표 |
| `data/processed_real/public_soil_dataset_inventory.json` | 같은 내용을 JSON으로 저장 |
| `data/processed_real/public_soil_prescription_candidate_index.csv` | 처방 후보 모델에 넣을 수 있는 sample-level index |
| `scripts/data/build_public_soil_data_inventory.py` | 위 파일들을 재생성하는 스크립트 |

## 수집된 데이터셋 요약

| 데이터셋 | 상태 | 샘플 수 | site 수 | 개입 record | 개입 종류 | 처방 기준 통과 | 용도 |
|---|---|---:|---:|---:|---:|---|---|
| `bonares_westerfeld` | 수집됨 | 216 | 1 | 216 | 13 | 실패 | 장기 포장시험의 경운/시비 contrast, 토양화학, 수확량 |
| `bernburg_three_years` | 수집됨 | 94 | 1 | 94 | 4 | 실패 | 3년 경운/시비 contrast, 토양화학 진단 |
| `usda_potato_rotation` | 수집됨 | 423 | 77 | 423 | 30 | 실패 | 감자 rotation/fumigation, 토양화학, 수확량 |
| `naylor_sorghum_drought` | 수집됨 | 623 | 1 | 623 | 2 | 실패 | drought/control stress signature |
| `neon_soil_microbe_ph` | 수집됨 | 2,482 | 20 | 0 | 0 | 실패 | 대규모 cross-site pH 진단 stress test |
| `dryad_organic_amendments` | 다운로드 차단 | 0 | 0 | 0 | 0 | 실패 | organic amendment response 후보 |
| `dryad_vanadium` | 다운로드 차단 | 0 | 0 | 0 | 0 | 실패 | 오염/기능성 진단 후보 |

## 기준별 판단

### 처방 기준을 통과하지 못한 이유

우리 기준은 최소한 다음을 요구합니다.

- 개입 record 1,000개 이상
- 개입 종류 5개 이상
- site 10개 이상
- 3개월 이상 follow-up
- control plot 존재

현재 공개 데이터에서 가장 가까운 것은 `usda_potato_rotation`입니다.

`usda_potato_rotation`은 site 수, 개입 종류, follow-up, control flag는 통과하지만 record 수가 423개라서 1,000개 기준에 못 미칩니다. 또한 intervention timing과 before/after 토양 상태 pairing이 불완전합니다.

`bonares_westerfeld`는 토양화학, 미생물, 경운/시비, 수확량이 모두 연결돼 있어 질은 좋습니다. 하지만 단일 site이고 true untreated control이 없어 처방 기준을 통과하지 못합니다.

### 진단에는 쓸 수 있는 데이터

`neon_soil_microbe_ph`는 샘플 수와 site 수가 충분해서 cross-site 진단 stress test에 쓸 수 있습니다. 하지만 pH가 sample-level 측정값이 아니라 site-level 평균값이라 진짜 per-sample 진단 데이터는 아닙니다.

`bernburg_three_years`와 `bonares_westerfeld`는 sample-level 토양화학이 붙어 있어 pH, carbon, nitrogen 진단 학습에 쓸 수 있습니다. 다만 규모와 지역 다양성이 작습니다.

## 처방 후보 index

`public_soil_prescription_candidate_index.csv`에는 총 1,356개 sample-level 후보 record를 모았습니다.

구성은 다음과 같습니다.

- `bonares_westerfeld`: 216개
- `bernburg_three_years`: 94개
- `usda_potato_rotation`: 423개
- `naylor_sorghum_drought`: 623개

공통 컬럼은 다음과 같습니다.

- `dataset_id`
- `sample_id`
- `site_id`
- `year`
- `intervention`
- `control_flag`
- `microbiome_available`
- `soil_chemistry_available`
- `outcome_available`
- `soil_ph`
- `total_carbon`
- `total_nitrogen`
- `organic_matter`
- `cec`
- `phosphorus`
- `potassium`
- `yield_value`
- `outcome_type`
- `notes`

이 index는 바로 처방 모델의 training table이 아니라, 공개 데이터 중 처방 후보로 묶을 수 있는 샘플 목록입니다. 실제 feature matrix는 각 데이터셋의 abundance table과 이 index를 `sample_id` 또는 dataset-specific key로 join해서 만들어야 합니다.

## 다운로드 차단된 데이터

Dryad의 `10.5061/dryad.4qrfj6q9n` organic amendments 데이터는 처방 관점에서 매우 유망합니다. 미생물, organic amendment, 토양화학/flux outcome이 연결된 구조라 공개 데이터 중 우선순위가 높습니다.

하지만 현재 환경에서는 Dryad 파일 다운로드가 Anubis와 AWS WAF challenge에 막혀 자동 수집이 안정적으로 되지 않았습니다. metadata와 파일 목록은 확인됐지만 실제 CSV 다운로드는 차단됐습니다.

이 데이터는 브라우저 수동 다운로드 또는 별도 허용된 네트워크 환경에서 다시 받아야 합니다.

## 지금 공개 데이터로 할 수 있는 최선

1. `usda_potato_rotation`과 `bonares_westerfeld`로 처방 후보 ranking baseline을 만든다.
2. `NEON`, `BonaRes`, `Bernburg`로 토양 상태 진단 모델을 따로 만든다.
3. `Naylor`는 drought/control stress detector와 OOD probe로만 쓴다.
4. 모든 결과에 `진단 가능`, `처방 후보`, `처방 불가`, `OOD` flag를 붙인다.
5. 실제 처방 모델이라고 부르지 않고, 공개 데이터 기반 evidence-ranking 모델로 제한한다.

## 재현 명령

```powershell
python scripts\data\build_public_soil_data_inventory.py
python -m pytest -q
```

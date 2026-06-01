# 처방 자동 추천을 위한 자체 데이터 요구사항과 평가 기준

작성일: 2026-06-01

## 결론

처방 자동 추천 모델은 단순히 `미생물 조성 -> 좋은/나쁜 토양 상태`를 맞히는 모델이 아니다.

운영 가능한 처방 모델은 다음 질문에 답해야 한다.

> 이 농장의 이 포장, 이 작물, 이 토양 상태, 이 미생물 상태에서 지금 어떤 처방을 얼마만큼 언제 넣으면, 무처리 또는 기존 관행 대비 토양 상태와 수확 결과가 얼마나 개선되는가?

따라서 필요한 데이터는 sample-level 미생물 데이터가 아니라 **plot-level intervention-response longitudinal 데이터**다. 공개데이터처럼 한 시점의 미생물, 토양화학, 수확량만 있는 데이터로는 처방 효과를 학습할 수 없다.

## 자동 처방이라고 부르기 위한 최소 정의

모델 출력은 최소한 다음을 포함해야 한다.

| 출력 항목 | 설명 |
|---|---|
| `recommended_intervention` | 추천 처방: 석회, 퇴비, 유기물, 질소/인/칼륨, 미생물제, 피복작물, 관개, 경운 변경 등 |
| `dose` | 투입량과 단위: kg/ha, ton/ha, L/ha 등 |
| `timing` | 적용 시점: 파종 전, 생육기, 수확 후, 월/주 단위 |
| `expected_effect` | 기대 효과: pH 변화, OM/SOC 변화, N/P/K 변화, 수확량 변화, 병해 감소 등 |
| `uncertainty` | 신뢰구간 또는 confidence band |
| `evidence_count` | 비슷한 조건에서 관측된 자체 데이터 수 |
| `contraindication` | 추천하면 안 되는 조건: pH 과다, 염류 위험, 작물 민감성, 지역 OOD 등 |
| `fallback` | 자동 처방 불가 시 출력: 추가 검사 필요, 전문가 검토 필요, 후보 근거만 제시 |

이 중 `dose`, `timing`, `expected_effect`, `uncertainty`가 없으면 자동 처방이 아니라 evidence ranking 또는 진단 리포트다.

## 학습 단위

처방 모델의 기본 학습 row는 `sample_id`가 아니라 **plot-season intervention episode**로 잡는다.

권장 primary key:

```text
site_id + plot_id + crop_season + intervention_episode_id
```

하나의 episode는 다음 구조를 가져야 한다.

1. 처리 전 baseline 측정
2. 처방 또는 무처리/관행 처리
3. 처리 후 follow-up 측정
4. 수확 또는 토양 개선 outcome
5. 같은 site/season 안의 control 또는 matched control

## 필수 데이터 테이블

### 1. Site 테이블

| 컬럼 | 필수 | 설명 |
|---|---|---|
| `site_id` | 필수 | 농장 또는 시험장 ID |
| `farm_id` | 권장 | 같은 농장 내 여러 site 구분 |
| `country`, `region` | 필수 | 국가/지역 |
| `latitude`, `longitude` | 필수 | 좌표 |
| `elevation_m` | 권장 | 고도 |
| `soil_series` | 권장 | 토양통/토양분류 |
| `soil_texture_class` | 필수 | sand/silt/clay 또는 texture class |
| `drainage_class` | 권장 | 배수 등급 |
| `irrigation_available` | 권장 | 관개 가능 여부 |
| `historical_land_use` | 권장 | 이전 토지 이용 |

### 2. Plot 테이블

| 컬럼 | 필수 | 설명 |
|---|---|---|
| `plot_id` | 필수 | 처리구 ID |
| `site_id` | 필수 | site 연결 |
| `block_id` | 필수 | randomized block 또는 반복구 |
| `plot_area_m2` | 필수 | 처리구 면적 |
| `crop` | 필수 | 작물 |
| `cultivar` | 권장 | 품종 |
| `planting_date` | 필수 | 파종/정식일 |
| `harvest_date` | 필수 | 수확일 |
| `previous_crop` | 권장 | 전작 |
| `management_baseline` | 필수 | 기존 관행 관리 요약 |

### 3. Sample 테이블

| 컬럼 | 필수 | 설명 |
|---|---|---|
| `sample_id` | 필수 | 시료 ID |
| `plot_id` | 필수 | 처리구 연결 |
| `sample_date` | 필수 | 채취일 |
| `sample_phase` | 필수 | `pre`, `post_1m`, `post_3m`, `post_6m`, `harvest` |
| `depth_cm_min`, `depth_cm_max` | 필수 | 채취 깊이 |
| `replicate_id` | 필수 | 생물학적 반복 |
| `composite_count` | 권장 | composite sample 구성 개수 |
| `storage_condition` | 권장 | 냉장/냉동/보관시간 |
| `lab_batch_id` | 필수 | DNA/화학 분석 batch |

### 4. Microbiome 테이블

| 컬럼 | 필수 | 설명 |
|---|---|---|
| `sample_id` | 필수 | Sample 연결 |
| `assay_type` | 필수 | 16S, ITS, shotgun |
| `target_region` | 필수 | V3-V4, V4, ITS2 등 |
| `sequencing_platform` | 필수 | MiSeq, NovaSeq 등 |
| `extraction_kit` | 필수 | DNA 추출 kit |
| `taxonomy_database` | 필수 | GTDB, SILVA 등 |
| `pipeline_version` | 필수 | QIIME2/DADA2 등 |
| `total_reads` | 필수 | 총 read 수 |
| `classified_reads` | 필수 | 분류된 read 수 |
| `taxon_id`, `rank`, `abundance` | 필수 | long format 권장 |

권장 feature는 genus-level abundance다. ASV-level도 보관하되, 모델 입력용 표준 feature matrix는 genus-level + CLR 변환을 기본으로 한다.

### 5. Soil Chemistry 테이블

처방 모델에는 처리 전과 처리 후 토양화학이 모두 필요하다.

| 컬럼 | 필수 | 설명 |
|---|---|---|
| `sample_id` | 필수 | Sample 연결 |
| `ph` | 필수 | pH |
| `organic_matter_pct` | 필수 | 유기물 |
| `soil_organic_carbon_pct` | 필수 | SOC |
| `total_carbon_pct` | 권장 | 총탄소 |
| `total_nitrogen_pct` | 필수 | 총질소 |
| `nitrate_n_mg_kg` | 권장 | 질산태 질소 |
| `ammonium_n_mg_kg` | 권장 | 암모늄태 질소 |
| `available_p_mg_kg` | 필수 | 유효 인산 |
| `exchangeable_k_mg_kg` | 필수 | 교환성 칼륨 |
| `ca_mg_kg`, `mg_mg_kg` | 권장 | Ca, Mg |
| `cec_cmol_kg` | 필수 | CEC |
| `ec_ds_m` | 필수 | 전기전도도/염류 위험 |
| `moisture_pct` | 권장 | 수분 |
| `bulk_density_g_cm3` | 권장 | 용적밀도 |
| `sand_pct`, `silt_pct`, `clay_pct` | 필수 | 토성 |

### 6. Intervention 테이블

| 컬럼 | 필수 | 설명 |
|---|---|---|
| `intervention_episode_id` | 필수 | 처방 episode ID |
| `plot_id` | 필수 | 처리구 연결 |
| `application_date` | 필수 | 적용일 |
| `intervention_family` | 필수 | lime, compost, fertilizer, biochar, cover_crop, irrigation, tillage, microbial_inoculant 등 |
| `material_name` | 필수 | 실제 투입재 이름 |
| `active_ingredient` | 권장 | 유효성분 |
| `dose_value` | 필수 | 투입량 |
| `dose_unit` | 필수 | 단위 |
| `application_method` | 필수 | 살포, 혼화, 관주, 엽면 등 |
| `application_depth_cm` | 권장 | 혼화 깊이 |
| `cost_per_ha` | 권장 | 비용 |
| `operator_notes` | 권장 | 작업 특이사항 |

처방 모델에는 `intervention_family`만 있으면 부족하다. 같은 퇴비라도 투입량, C/N, 수분, 부숙도, 적용 시점이 다르면 효과가 다르다.

### 7. Weather/Irrigation 테이블

| 컬럼 | 필수 | 설명 |
|---|---|---|
| `site_id` | 필수 | site 연결 |
| `date` | 필수 | 일 단위 |
| `precipitation_mm` | 필수 | 강수량 |
| `tmin_c`, `tmax_c` | 필수 | 최저/최고 기온 |
| `gdd` | 권장 | 생육도일 |
| `evapotranspiration_mm` | 권장 | 증발산 |
| `irrigation_mm` | 필수 | 관개량 |
| `extreme_event_flag` | 권장 | 폭염/가뭄/침수 |

처방 효과는 날씨와 강하게 상호작용하므로, 최소한 intervention 전후 90일의 daily weather가 필요하다.

### 8. Outcome 테이블

| 컬럼 | 필수 | 설명 |
|---|---|---|
| `intervention_episode_id` | 필수 | intervention 연결 |
| `outcome_date` | 필수 | outcome 측정일 |
| `yield_kg_ha` | 필수 | 수확량 |
| `marketable_yield_kg_ha` | 권장 | 상품 수량 |
| `biomass_kg_ha` | 권장 | 생체량 |
| `quality_metric` | 권장 | 당도, 단백질, 등급 등 |
| `disease_score` | 권장 | 병해 지수 |
| `post_ph`, `post_om`, `post_soc`, `post_npk` | 필수 | 처리 후 토양 상태 |
| `net_profit_per_ha` | 권장 | 경제성 |

자동 처방의 target은 raw yield 하나가 아니라, baseline과 control 대비 변화량이어야 한다.

## 모델 target 정의

권장 target은 다음 순서로 만든다.

### 1. 처리 전후 변화량

```text
delta_soil_state = post_treatment_soil_state - pre_treatment_soil_state
delta_yield = observed_yield - expected_baseline_yield
```

### 2. 대조구 대비 효과

같은 site, season, crop, block 안의 control을 기준으로 한다.

```text
treatment_effect = delta_treated_plot - delta_control_plot
```

### 3. 추천용 reward

처방 추천은 단일 지표가 아니라 multi-objective reward로 정의한다.

```text
reward = w1 * yield_gain
       + w2 * soil_health_gain
       - w3 * cost
       - w4 * salinity_or_ph_risk
       - w5 * uncertainty_penalty
```

가중치는 작물/고객 목표별로 따로 설정한다. 예를 들어 토양 회복 목적이면 `soil_health_gain` 비중을 높이고, 상업 농가 목적이면 `yield_gain`과 `net_profit` 비중을 높인다.

## 최소 데이터 수량 기준

### MVP 데이터 gate

이 기준은 “자동 처방”이 아니라 “제한적 처방 후보 추천”을 시작할 수 있는 최소선이다.

| 항목 | 최소 기준 |
|---|---:|
| intervention episode | 1,000개 이상 |
| site 수 | 10개 이상 |
| crop 수 | 2개 이상 |
| season 수 | 2년 이상 |
| intervention family | 5개 이상 |
| family별 episode | 100개 이상 |
| treated-control pair | 500쌍 이상 |
| 처리 전 microbiome + soil chemistry | episode의 90% 이상 |
| 처리 후 3개월 이상 follow-up | episode의 80% 이상 |
| 수확량 또는 biomass outcome | episode의 80% 이상 |
| dose/timing 기록 | episode의 95% 이상 |

### 운영 자동 처방 gate

이 기준을 통과해야 제품에서 “자동 추천”이라는 표현을 쓸 수 있다.

| 항목 | 운영 기준 |
|---|---:|
| intervention episode | 5,000개 이상 |
| site 수 | 30개 이상 |
| crop 수 | 3개 이상 |
| season 수 | 3년 이상 |
| climate/region cluster | 3개 이상 |
| soil texture class | 3개 이상 |
| intervention family | 7개 이상 |
| 주요 family별 episode | 300개 이상 |
| 주요 family별 treated-control pair | 150쌍 이상 |
| dose level | 주요 처방별 3단계 이상 + zero control |
| external holdout site | 전체 site의 20% 이상 |
| prospective validation site | 최소 3개 site |

### 개별 처방별 추천 gate

전체 데이터가 충분해도 특정 처방 데이터가 부족하면 그 처방은 추천하면 안 된다.

| 조건 | 기준 |
|---|---:|
| 비슷한 crop + texture + pH band + climate 조건의 근거 episode | 50개 이상 |
| 같은 intervention family의 treated-control pair | 30쌍 이상 |
| 같은 dose 범위 근거 | 20개 이상 |
| held-out site에서 효과 방향이 맞은 비율 | 60% 이상 |
| 예상 효과 신뢰구간 하한 | 0 이상 또는 고객 목표 기준 충족 |

이 기준 미달이면 출력은 `자동 추천 불가`, `근거 부족`, `추가 검사 필요`로 제한한다.

## 데이터 품질 기준

### Sample 품질

| 항목 | 기준 | 실패 시 처리 |
|---|---|---|
| `sample_id` 중복 | 없어야 함 | 중복 제거 또는 재검토 |
| 채취 깊이 | 동일 task 내 표준화 | depth stratification 또는 제외 |
| pre/post pairing | 필수 | 처방 학습 제외 |
| control pairing | 필수 | evidence ranking만 허용 |
| lab batch | 기록 필수 | batch correction 불가 시 OOD flag |

### Microbiome 품질

| 항목 | 기준 | 실패 시 처리 |
|---|---|---|
| total reads | 10,000 이상 | 제외 |
| classified genera | 20개 이상 | 제외 |
| top-1 genus share | 90% 미만 | contamination flag |
| extraction kit | 기록 필수 | batch correction 불가 |
| taxonomy database | 고정 또는 mapping 가능 | mapping 불가 시 별도 cohort |
| negative/blank control | batch별 필요 | 오염 평가 불가 flag |

### Metadata 품질

| 항목 | 기준 |
|---|---|
| 필수 컬럼 completeness | 95% 이상 |
| outcome completeness | 80% 이상 |
| dose/timing completeness | 95% 이상 |
| weather coverage | intervention 전후 90일 이상 |
| unit 표준화 | SI 또는 프로젝트 표준 단위 |

## 검증 설계

절대 random sample split을 기본 성능으로 쓰면 안 된다. 같은 site, plot, season이 train/test에 섞이면 모델이 처방 효과가 아니라 지역/농장 fingerprint를 외울 수 있다.

필수 split은 다음 4개다.

| 검증 | 목적 | 기준 |
|---|---|---|
| leave-site-out | 새 농장/지역 일반화 | site 단위 holdout |
| leave-season-out | 다음 시즌 예측 | 최신 season holdout |
| leave-region-out | 기후/토양권역 OOD | region cluster holdout |
| prospective validation | 실제 운영 검증 | 모델 추천 후 미래 결과 확인 |

## 모델 평가 지표

### 진단 모델 지표

| 지표 | 통과 기준 |
|---|---:|
| held-out site R2 | 0.20 이상 |
| RMSE improvement over mean baseline | 10% 이상 |
| calibration slope | 0.8-1.2 |
| shortcut accuracy over majority | 0.25 이하 |

진단은 처방의 입력일 뿐이다. 진단 성능이 좋아도 처방 효과 모델이 통과하지 못하면 자동 추천은 불가하다.

### 처방 효과 모델 지표

| 지표 | 의미 | MVP 기준 | 운영 기준 |
|---|---|---:|---:|
| uplift RMSE improvement | control 대비 효과 예측 RMSE가 baseline보다 얼마나 나은지 | 5% 이상 | 10% 이상 |
| treatment effect sign accuracy | 효과 방향을 맞힌 비율 | 60% 이상 | 65% 이상 |
| top-1 regret | 모델이 고른 처방과 실제 최선 처방의 reward 차이 | baseline 대비 10% 감소 | baseline 대비 20% 감소 |
| policy value | 모델 정책을 따랐을 때의 기대 reward | 관행 대비 양수 | 관행 대비 통계적으로 유의한 양수 |
| calibration coverage | 80% 예측구간 실제 포함률 | 70-90% | 75-85% |
| safety violation rate | 금지 조건 위반 추천 비율 | 1% 미만 | 0.5% 미만 |

### Ranking 지표

처방을 하나만 고르는 것이 아니라 후보 순위를 내는 경우:

| 지표 | 기준 |
|---|---:|
| top-3 hit rate | 70% 이상 |
| NDCG@3 | 0.70 이상 |
| Spearman rank correlation | 0.30 이상 |
| negative-effect top recommendation rate | 5% 미만 |

## Baseline 비교 대상

모델은 반드시 다음 baseline을 이겨야 한다.

1. 아무 처방도 하지 않는 control
2. 농장 기존 관행
3. rule-based agronomy baseline
4. site 평균 처방 효과
5. crop/soil texture별 평균 처방 효과
6. soil chemistry only 모델
7. microbiome 없는 management/weather 모델

미생물 모델은 `soil chemistry only`와 `management/weather only`를 이기지 못하면 “미생물 기반 처방”이라고 주장할 수 없다.

## Leakage 금지 규칙

처방 모델 입력에는 다음이 들어가면 안 된다.

| 금지 feature | 이유 |
|---|---|
| 처리 후 토양화학 | outcome leakage |
| 수확량 또는 수확 직전 생육 지표 | target leakage |
| intervention 이후 weather summary만 있고 pre 시점에서 알 수 없는 변수 | 미래 정보 leakage |
| plot_id/site_id를 직접 one-hot으로 넣은 feature | site memorization |
| 같은 plot의 post sample microbiome | 처방 후 결과를 입력에 넣는 문제 |

입력은 추천 시점에 실제로 알 수 있는 정보만 허용한다.

## 추천 차단 조건

다음 조건이면 모델은 자동 처방을 내면 안 된다.

| 조건 | 출력 |
|---|---|
| site/crop/soil texture가 학습 분포 밖 | `OOD: 전문가 검토 필요` |
| 비슷한 조건의 evidence episode가 50개 미만 | `근거 부족` |
| target pH/EC가 안전 범위 밖 | `추가 토양검사 필요` |
| 추천 처방의 신뢰구간 하한이 0 미만 | `추천 보류` |
| dose 범위가 학습 데이터 최대/최소를 벗어남 | `dose extrapolation 금지` |
| 금지 조합 감지: high EC + 염류성 투입재 등 | `안전 차단` |

## 모델 단계별 제품 명칭

| 단계 | 허용 명칭 | 조건 |
|---|---|---|
| Level 0 | 진단 리포트 | 처방 effect 데이터 없음 |
| Level 1 | 처방 후보 근거 제시 | 공개데이터 또는 자체 데이터 일부, control 부족 |
| Level 2 | 제한적 처방 추천 | MVP data gate + held-out 검증 통과 |
| Level 3 | 자동 처방 추천 | 운영 data gate + prospective validation 통과 |
| Level 4 | 자율 최적화 | 다년 현장 A/B + active learning + 안전 모니터링 통과 |

현재 프로젝트는 공개데이터 기준으로 Level 1에 가깝다. Level 2 이상으로 가려면 자체 intervention-response 데이터가 필요하다.

## 수집 우선순위

### 1순위: 처방 전후 paired trial

가장 먼저 모아야 하는 데이터는 같은 plot에서 다음이 모두 연결된 데이터다.

```text
pre microbiome
pre soil chemistry
intervention + dose + timing
post soil chemistry
yield/outcome
matched control
```

### 2순위: 여러 site 반복

단일 농장 데이터는 모델이 site fingerprint를 학습할 위험이 크다. 같은 프로토콜로 최소 10개 site부터 시작하고, 운영 기준은 30개 site 이상으로 둔다.

### 3순위: dose-response

처방 자동 추천에는 “무엇을 넣을까”뿐 아니라 “얼마나 넣을까”가 필요하다. 주요 처방은 최소 3개 dose level과 zero control이 있어야 한다.

### 4순위: prospective validation

과거 데이터에서 성능이 좋아도 실제 추천을 내고 다음 시즌 결과를 맞히지 못하면 자동 처방으로 배포하면 안 된다.

## 실제 수집 설계 예시

초기 MVP 실험 설계:

| 항목 | 권장 설계 |
|---|---|
| site | 10개 농장 |
| crop | 2개 작물 |
| season | 2년 |
| plot/site | 20개 이상 |
| 처방 | control + 5개 family |
| dose | 주요 처방 3단계 |
| sampling | pre, post_3m, harvest |
| 반복 | block별 3반복 이상 |
| 목표 episode | 1,000개 이상 |

운영 전 검증 설계:

| 항목 | 권장 설계 |
|---|---|
| site | 30개 이상 |
| crop | 3개 이상 |
| season | 3년 이상 |
| external holdout | site의 20% |
| prospective validation | 최소 3개 site |
| 목표 episode | 5,000개 이상 |

## 최종 go/no-go 기준

| 판정 | 조건 |
|---|---|
| Go: 자동 처방 | 운영 data gate 통과, policy value 양수, safety violation 0.5% 미만, prospective validation 통과 |
| Limited Go: 제한 추천 | MVP data gate 통과, 특정 crop/site/처방 범위에서만 성능 통과 |
| No-Go: 근거 제시만 | control 또는 follow-up 부족, uplift 성능 미달 |
| No-Go: 진단만 | intervention-response 데이터 없음 |

## 현재 프로젝트에 대한 적용

현재 공개데이터 baseline 결과는 다음 상태다.

- pH 등 토양 상태 진단: 일부 가능성 있음
- drought/control 같은 상태 분류: 가능성 있음
- 수확량/처방 효과 예측: 불합격
- 자동 처방 추천: 불가

따라서 다음 개발 순서는 다음과 같다.

1. 공개데이터 모델은 baseline/evidence ranking으로 고정한다.
2. 자체 데이터 수집 계약은 위 테이블 스키마를 기준으로 잡는다.
3. 첫 자체 데이터는 자동 처방이 아니라 `Level 2 제한적 처방 추천` 통과를 목표로 설계한다.
4. prospective validation을 통과하기 전까지 제품 문구는 “자동 처방”이 아니라 “처방 후보 근거 제시”로 제한한다.

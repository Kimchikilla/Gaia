# Gaia Roadmap (post-2026-05-06)

> 이전 로드맵 (`ProjectGaia_오픈소스_로드맵.md`) 은 폐기됨. 그 로드맵은 "토양 미생물 AlphaFold" 라는 과장된 슬로건과 잘못된 가정 (16S 데이터로 batch effect 없이 큰 모델 만들면 잘 됨) 위에 짜여있었음. 2026-05-04~06 검증으로 그 가정이 틀린 게 확인됨 — v6 의 진단 R² 0.95 가 거의 100% batch shortcut, 진짜 신호는 R² 0.11 수준. 이 로드맵은 그 발견을 출발점으로 다시 짠 것.

---

## 핵심 인식 (정직)

1. **현재 v6/v7 모델의 진단 R² 는 lab/country fingerprint 외운 결과**. 새 lab (예: 한국 포도밭) 에선 거의 안 통함.
2. 16S rRNA 데이터는 "누가 있냐" 만 알려줌 — "걔가 뭘 하냐" 는 안 들어있음. 진짜 미생물 기능 학습하려면 **shotgun metagenome + 기능 vocab** 필요.
3. 토양 미생물 ML 분야에 **표준 평가 인프라가 부재**. 우리가 겪은 함정은 분야 전체의 함정. 이걸 해결하지 않고는 신뢰할 만한 FM 못 만듦.
4. 따라서 Gaia 의 본분 두 트랙으로 재정의:
   - **Track A**: 분야 평가 표준 정립
   - **Track B**: 그 표준 위에서 진짜 기능 모델 (shotgun + KEGG vocab)

---

## Track A — 평가 표준 정립

분야의 sloppy 한 검증 관행을 반복하지 않기 위한 인프라.

### A1. 검증 suite 패키지화 (1~2주)

이미 구현한 검증을 재사용 가능 모듈로:

- `gaia.eval.baselines` — mean baseline, label-shuffle, study-only, RF
- `gaia.eval.probes` — country probe, study probe, biome probe
- `gaia.eval.ood` — leave-one-study-out, leave-one-country-out
- `gaia.eval.report` — 자동 보고서 생성 (한 번 호출하면 표 + 시각화)

학습 후 호출 한 줄로 모든 검증 강제. 통과 못하면 publish 금지.

### A2. 평가 critique 논문 (2~3주)

> *"Soil Microbiome Foundation Models: Quantifying Batch-Effect Shortcut"*

기존 토양 미생물 ML 논문 R² 들이 우리 v6 처럼 inflated 임을 정량화. 필드에 honest evaluation 요청.

구조:
1. 분야 현황 review
2. v6 reproduce → batch shortcut 검출
3. v9 (BERT + adversarial) → 진짜 신호 측정
4. 평가 도구 제안

이 자체가 분야 contribution. 또 다른 inflated FM paper 보다 영향력 큼.

### A3. 공개 leaderboard (3~6개월)

GitHub 에 `gaia-bench` 레포. 누가 모델 제출하면 자동으로 검증 suite 돌아가고 점수 뜸. country probe acc, mean baseline 차이, LOCO 점수 다 표시. inflated R² 자랑 못 하게.

---

## Track B — 진짜 기능 모델 (D + E)

표준 위에서 학습. v9 의 R² 0.11 을 뚫고 가는 길.

### B1. Shotgun 데이터 확보 (2~4주)

- **JGI IMG/M** 가입 → 토양 shotgun + KEGG annotation 표 다운
- 50~100 샘플 (토양 화학 페어 있는 것) 첫 목표
- 안 되면 NCBI SRA raw FASTQ + 자체 처리 (eggNOG-mapper, KofamKOALA)

### B2. 기능 vocab 빌드 (3~5일)

- 현재: 12,916 (속 단위, `g__Bacillus`)
- 새: KEGG ortholog (~10,000), `K00001` 같은 ID
- 또는 COG / Pfam — KEGG 추천 (농학·생태학 표준)

### B3. v10 학습 (1~2주)

- BERT 8L 256d (v8/v9 와 같은 architecture)
- KEGG vocab 위에서 처음부터
- v3-CLR 같은 정규화 적용
- 학습 끝날 때마다 Track A 검증 suite 자동 실행

### B4. 16S + Shotgun 혼합 모델 (1~2개월, 후속)

multi-modal:
- 16S 부분 → 분류 정보 (어떤 미생물이 있냐)
- Shotgun 부분 → 기능 정보 (걔가 뭘 하냐)
- 두 vocab 같이 학습

이게 진짜 본분 모델.

---

## Track C — 응용 / 본분 연결 (장기)

### C1. 한국 wet-lab 데이터 (2~3개월, 비용 동반)

- 친구 포도밭 + 인근 농지 30~50개 직접 시퀀싱
- 비용: 16S 만 약 150~300만원, shotgun 까지면 3000~6000만원
- 진짜 cross-continent 검증 + 본분의 "한국 농지 진단" 데모

### C2. CLI 도구 갱신

`gaia diagnose` 가 v6 헤드 쓰고있음. 이거:
- v9/v10 헤드로 교체
- **신뢰도 표시** 추가 — "이 예측 R² 0.1 수준임. 새 lab 데이터엔 약함"
- 사용자가 자기 데이터 lab 정보 입력하면 그 lab 의 OOD 정도 estimate

### C3. 처방 (inverse design) 재설계

현재 k-NN 검색 기반은 분포 좁아 한계. v10 + multi-task 학습 위에 conditional generation 다시 시도. T5 식 encoder-decoder 또는 diffusion.

---

## Timeline

| 기간 | Track A | Track B | Track C |
|---|---|---|---|
| **5월** | 검증 suite 패키지화, 결과 정리 | JGI 가입, 카탈로그 확보 | — |
| **6월** | critique 논문 1차 draft | shotgun 50개 다운, 기능 vocab | — |
| **7월** | leaderboard scaffold, paper 제출 | v10 학습 + 검증 | wet-lab 견적, 시료 채취 |
| **8월~** | community 확장 | v10 → v11 (multi-modal) | wet-lab 데이터 시퀀싱 |
| **연말** | — | 16S + Shotgun 혼합 모델 | 한국 농지 데모 |

병렬로 가능. Track A 가 B, C 의 토대 — 먼저 무겁게 쌓는다.

---

## 폐기된 가정들 (기록용)

이전 로드맵이 가정했던 것 중 검증 후 폐기:

- ❌ "10,000+ 샘플 모으면 큰 모델 됨" → lab 다양성이 sample 수보다 중요
- ❌ "진단 R² 0.95 면 production 가능" → batch shortcut 빼면 0.11
- ❌ "OOD = 다른 사이트만 다르면 됨" → 같은 lab/protocol 이면 OOD 아님
- ❌ "GPT-2 가 미생물에 fit" → causal mask + 정렬 의존이 batch effect 보존
- ❌ "처방 = autoregressive generation" → 현재 코드는 retrieval 기반, 생성 미사용

---

## 마음가짐

이전 로드맵은 "AlphaFold 만들자" 였음. 이번 로드맵은 **"분야 표준 만들면서 정직한 모델 키우자"** 임.

R² 0.95 자랑하는 영광 대신 R² 0.11 부터 정직히 키우는 길. 그게 본분 (실제 한국 포도밭에서 작동하는 토양 진단) 으로 가는 유일한 honest 경로.

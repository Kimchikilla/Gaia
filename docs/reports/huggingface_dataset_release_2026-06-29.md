---
license: other
language:
- ko
- en
tags:
- soil
- microbiome
- metagenomics
- agriculture
- environmental-science
- tabular
pretty_name: Gaia Public Soil Microbiome Corpus
size_categories:
- 10K<n<100K
---

# Gaia Hugging Face 공개 데이터셋 배포 보고서

배포일: 2026-06-29  
Hugging Face: https://huggingface.co/datasets/Kimchikilla/gaia-corpus

## 결론

이번 배포본은 원본 데이터 전체 미러가 아니라, 라이선스 리스크를 줄인 공개 가공 데이터셋이다. Gaia 코드의 Apache-2.0 라이선스를 원천 데이터에 덮어씌우지 않고, 출처별 라이선스와 이용 조건을 별도로 명시했다.

핵심 정책은 다음과 같다.

| 구분 | 처리 |
|---|---|
| 공개 허용 가공본 | Hugging Face에 업로드 |
| 원본 대형 raw mirror | Hugging Face에서 삭제 |
| Bernburg 원본/가공 테이블 | 라이선스 부재로 삭제 및 제외 |
| JGI 수동 다운로드 원본/v10 파생 파일 | 계정/약관 기반 다운로드라 제외 |
| Dryad placeholder | DOI는 CC0지만 로컬 파일이 0바이트라 제외 |
| benchmark 파일 | Bernburg 행/섹션 제거 후 public release 버전만 업로드 |

## 업로드된 파일

Hugging Face 공개본에는 총 25개 payload 파일과 매니페스트가 올라갔다. 저장소 검증 기준으로 `raw/` 폴더는 삭제되었고, `bernburg`가 들어간 파일명은 남아 있지 않다.

| 경로 | 내용 |
|---|---|
| `processed_real/gaia-abundance-v1.csv` | MGnify 기반 abundance matrix |
| `processed_real/gaia-abundance-v2.csv` | 확장 public abundance matrix |
| `processed_real/gaia-corpus-v1.pkl` | 학습 재현용 serialized corpus |
| `processed_real/gaia-corpus-v2.pkl` | 학습 재현용 serialized corpus |
| `processed_real/gaia-corpus-v3.pkl` | 학습 재현용 serialized corpus |
| `processed_real/gaia-corpus-v7-clr.pkl` | CLR 정규화 corpus |
| `processed_real/gaia-metadata-v1.csv` | MGnify 기반 metadata |
| `processed_real/gaia-metadata-v2.csv` | EMP + v1 metadata |
| `processed_real/gaia-metadata-v3.csv` | EMP + MGnify + NEON metadata |
| `processed_real/gaia-metadata-v7.csv` | v7 corpus metadata |
| `processed_real/soil_state_neon_ph.csv` | NEON site-level pH 진단 테이블 |
| `processed_real/soil_state_westerfeld.csv` | BonaRes Westerfeld paired soil state 테이블 |
| `processed_real/public_soil_*` | Bernburg 행을 제거한 public benchmark/inventory/index |
| `processed_real/tokenizer.json` | public corpus 기반 tokenizer |
| `configs/*.yaml` | 데이터 소스 설정 |
| `README.md` | HF 데이터셋 카드 |
| `SOURCE_LICENSES.md` | 출처별 라이선스 판정 |
| `manifest.csv`, `manifest.json` | 파일별 크기, SHA-256, 출처 그룹 |

`*.pkl` 파일은 pickle 직렬화 파일이므로 신뢰 가능한 환경에서만 로드해야 한다.

## 삭제한 항목

Hugging Face에서 기존에 올라가 있던 `raw/**` 전체를 삭제했다. 특히 아래 항목은 공개 재배포 기준에서 제거했다.

| 항목 | 이유 |
|---|---|
| `raw/bernburg/**` | 원본 GitHub 저장소와 fork 모두 명시 라이선스 없음 |
| `processed_real/bernburg_abundance.csv` | Bernburg 파생 데이터라 제외 |
| `processed_real/bernburg_metadata.csv` | Bernburg 파생 데이터라 제외 |
| `raw/dryad_amendments/**` | 로컬/HF 파일이 실제 데이터가 아닌 placeholder |
| `raw/dryad_vanadium/**` | 로컬/HF 파일이 실제 데이터가 아닌 0바이트 placeholder |
| 대형 raw FASTQ/BIOM/zip mirror | 공개 corpus 재현에 필수 아님, 출처별 조건 혼재 |

## 라이선스 판정

| Source | 공개본 처리 | 라이선스/근거 | 주의 |
|---|---:|---|---|
| Gaia 작성 코드/설정 | 포함 | Apache-2.0 | Gaia 작성물에만 적용 |
| NEON | 포함 | NEON 공개 데이터 정책. 기존 CC0 안내와 2026-06-30 전후 CC BY 4.0 전환을 보수적으로 반영 | attribution 권장 |
| BonaRes Westerfeld | 포함 | DataCite DOI `10.20387/bonares-w669-gdsd` 기준 CC BY 4.0 | attribution 필수 |
| Dryad amendments | 실제 데이터 제외 | DOI `10.5061/dryad.4qrfj6q9n`는 CC0 | 로컬 파일이 placeholder라 미포함 |
| Dryad vanadium | 실제 데이터 제외 | DOI `10.5061/dryad.6wwpzgn52`는 CC0 | 로컬 파일이 0바이트라 미포함 |
| Bernburg GitHub repo | 제외 | GitHub API 기준 fork/upstream 모두 license 없음 | 공개 재배포 가정 금지 |
| JGI manual download | 제외 | 계정/약관 기반 다운로드 | 재배포 조건 확인 전 제외 |
| MGnify / EMP / Naylor 기반 corpus | 가공본 포함 | 공개 연구 DB/데이터 기반. 원 출처 citation과 이용 조건 준수 필요 | Gaia가 원천 데이터를 소유하지 않음 |

## 모델 관점에서의 의미

이번 HF 배포본으로 가능한 것은 “공개 데이터 기반 진단 baseline, corpus 학습, benchmark 재현”이다. 자동 처방 추천을 제품 수준으로 주장하기에는 아직 부족하다. 처방 모델을 위해서는 같은 농지에서 반복 측정된 미생물, 토양 화학성, 시비·관수·기상, 처방 실행, 수확량/병해/토양 개선 결과가 묶인 paired time-series 데이터가 필요하다.

## 출처 URL

- NEON data policy: https://www.neonscience.org/data-samples/guidelines-policies/publishing-research-outputs
- BonaRes Westerfeld DOI: https://doi.org/10.20387/bonares-w669-gdsd
- Dryad amendments DOI: https://doi.org/10.5061/dryad.4qrfj6q9n
- Dryad vanadium DOI: https://doi.org/10.5061/dryad.6wwpzgn52
- MGnify: https://www.ebi.ac.uk/metagenomics/
- EMBL-EBI terms: https://www.ebi.ac.uk/about/terms-of-use
- Earth Microbiome Project: https://earthmicrobiome.org/

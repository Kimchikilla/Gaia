# Gaia 공개 데이터 출처별 라이선스 판정

판정일: 2026-06-29

## 포함

| Source group | 파일 | 판정 | 근거 |
|---|---|---|---|
| Gaia 작성 config/manifest | `configs/*`, `manifest.*` | 포함 | Gaia 작성물이며 프로젝트 라이선스는 Apache-2.0 |
| NEON | `soil_state_neon_ph.csv`, NEON 파생 corpus 행 | 포함 | NEON 공개 데이터 정책. 보수적으로 attribution을 요구하는 방향으로 표기 |
| BonaRes Westerfeld | `soil_state_westerfeld.csv`, Westerfeld 기반 index | 포함 | DataCite DOI `10.20387/bonares-w669-gdsd`가 CC BY 4.0로 명시 |
| MGnify / EMP / Naylor 가공 corpus | `gaia-abundance-v*`, `gaia-metadata-v*`, `gaia-corpus-v*`, `tokenizer.json` | 가공본 포함 | 공개 연구 데이터 기반. 원 출처 citation과 이용 조건 준수 필요 |

## 제외

| Source | 제외 경로 | 이유 |
|---|---|---|
| Bernburg three-year GitHub repository | `raw/bernburg/**`, `processed_real/bernburg_*`, `soil_state_bernburg.csv`, Bernburg benchmark row | fork와 upstream 저장소 모두 명시 라이선스가 없어 공개 재배포를 가정하지 않음 |
| JGI manual download | `raw/jgi_manual/**`, `gaia-corpus-v10-kegg.pkl`, `gaia-metadata-v10.csv` | 계정/약관 기반 다운로드라 재배포 조건 미확인 |
| Dryad placeholder downloads | `raw/dryad_amendments/**`, `raw/dryad_vanadium/**` | DOI는 CC0지만 현재 로컬 파일이 실제 데이터가 아닌 placeholder/0바이트 |
| 대형 raw mirror | raw FASTQ/BIOM/zip/source mirror | 공개 모델 corpus 배포에 필수 아님. 출처별 조건이 섞여 있어 가공본 중심으로 배포 |

## 운영 원칙

1. Gaia의 Apache-2.0 라이선스는 Gaia가 작성한 코드, 설정, 문서에만 적용한다.
2. 외부 데이터는 각 출처 라이선스와 citation 요구사항을 따른다.
3. 라이선스가 없거나 계정/약관 기반 다운로드인 데이터는 공개 HF 배포에서 제외한다.
4. benchmark 결과도 원천 데이터 라이선스가 불명확하면 해당 row/section을 제거한다.
5. 공개 배포본은 원본 아카이브가 아니라 재현 가능한 가공 corpus와 매니페스트 중심으로 유지한다.

## 확인 URL

- NEON data policy: https://www.neonscience.org/data-samples/guidelines-policies/publishing-research-outputs
- BonaRes Westerfeld DOI: https://doi.org/10.20387/bonares-w669-gdsd
- Dryad amendments DOI: https://doi.org/10.5061/dryad.4qrfj6q9n
- Dryad vanadium DOI: https://doi.org/10.5061/dryad.6wwpzgn52
- MGnify: https://www.ebi.ac.uk/metagenomics/
- EMBL-EBI terms: https://www.ebi.ac.uk/about/terms-of-use
- Earth Microbiome Project: https://earthmicrobiome.org/

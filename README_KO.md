# Gaia: 토양 미생물 파운데이션 모델

공개 토양 미생물 데이터(MGnify + EMP)로 사전학습한 트랜스포머 모델. 토양 화학 예측, 가뭄 분류, 컨소시엄 추천을 위한 linear probe 헤드와 CLI 제공. 2026-05-06 이후의 정직한 방향은 **[ROADMAP.md](ROADMAP.md)** 참고.

[English](README.md) | **한국어**

[![HF Model](https://img.shields.io/badge/HuggingFace-Kimchikilla%2Fgaia-yellow)](https://huggingface.co/Kimchikilla/gaia)
[![HF Dataset](https://img.shields.io/badge/HuggingFace_Dataset-Kimchikilla%2Fgaia--corpus-yellow)](https://huggingface.co/datasets/Kimchikilla/gaia-corpus)
[![Dataset Downloads](https://img.shields.io/badge/dataset_downloads-2.7k%2F30d-brightgreen)](https://huggingface.co/datasets/Kimchikilla/gaia-corpus)

`gaia-corpus` 데이터셋은 Hugging Face Hub에서 최근 30일간 약 2,770회 다운로드됨 (2026-05-04 기준).

---

## 주요 기능

- **사전학습 모델**: MGnify + EMP v2 코퍼스 7,170개 시퀀스로 사전학습한 GPT 스타일 Transformer (8 레이어, 256d)
- **토양 화학 예측**: pH R²=0.95, 총탄소 R²=0.88, 총질소 R²=0.88 (Westerfeld 인디스트리뷰션). Bernburg OOD 에서는 pH=0.59, C=0.72, N=0.73
- **사이트 간 OOD**: linear probe 5/6, zero-shot 3/3 에서 RandomForest 보다 R² 높음. 수확량 회귀에서는 RF 에 짐
- **가뭄 분류**: Naylor (미국 캘리포니아 수수) 데이터에서 OOD 검증 — `docs/benchmark_naylor.json`
- **컨소시엄 추천 (역설계)**: 목표 (pH, C, N) 입력 → 임베딩이 가장 가까운 실제 샘플 k개를 가중평균해 추천 속 출력
- **CLI**: `gaia diagnose abundance.csv` (JSON/Markdown 보고서), `gaia design --ph 6.5 --carbon 1.8 --nitrogen 0.18` (컨소시엄 추천)

## 빠른 시작

### 설치

```bash
pip install gaia-soil
```

또는 소스에서 설치:

```bash
git clone https://github.com/Kimchikilla/ProjectGaia.git
cd ProjectGaia
pip install -e ".[dev]"
```

### 기본 사용법

CLI:

```bash
# 토양 건강 진단 (pH, 총탄소, 총질소 + 핵심 속 출력)
gaia diagnose path/to/abundance.csv --markdown report.md

# 역설계 — 목표 토양 상태에 맞는 컨소시엄 추천
gaia design --ph 6.5 --carbon 1.8 --nitrogen 0.18 --top-n 15
```

Python API:

```python
from gaia.cli import diagnose_file
from gaia.inference.inverse_design import DesignTarget, design_consortium

reports = diagnose_file("path/to/abundance.csv")
for r in reports:
    print(r.to_text())

rec = design_consortium(DesignTarget(ph=6.5, total_carbon=1.8, total_nitrogen=0.18))
print(rec.to_text(top_n=15))
```

## 프로젝트 구조

```
gaia/
├── README.md                  # 영문 README
├── README_KO.md               # 한국어 README
├── LICENSE                    # Apache 2.0
├── CONTRIBUTING.md
├── docs/
│   ├── roadmap.md
│   ├── data_standard.md       # 데이터 표준화 가이드
│   └── tutorials/
├── data/
│   ├── scripts/               # 데이터 수집·전처리 스크립트
│   ├── configs/               # 데이터 소스별 설정
│   └── README.md              # 데이터 카탈로그
├── gaia/
│   ├── preprocessing/         # 전처리 모듈
│   ├── models/                # 모델 아키텍처
│   ├── training/              # 학습 스크립트
│   ├── evaluation/            # 평가 모듈
│   └── inference/             # 추론 모듈
├── benchmarks/                # 벤치마크 데이터셋·평가 기준
├── notebooks/                 # 튜토리얼 Jupyter 노트북
└── tests/
```

## 데이터 소스

실제 사전학습 / 벤치마크에 쓰인 처리된 코퍼스:

| 소스 | 설명 | v2 코퍼스 |
|------|------|---:|
| [MGnify](https://www.ebi.ac.uk/metagenomics/) | 속(genus) 단위 풍부도 테이블 | 2,887 |
| [Earth Microbiome Project](https://earthmicrobiome.org/) | EMP release1 에서 토양 샘플 필터 | 4,628 |
| **사전학습 합계 (v2)** | 토큰화 필터 후 (속 ≥5) | **7,170 시퀀스** |
| [Westerfeld LTE](https://www.bonares.de/) | 미생물 + 토양화학 페어 (독일) | held out — 파인튜닝/벤치마크 |
| [Bernburg LTE](https://github.com/raabmarie/Synthesis_Three_Years_Bernburg) | 미생물 + 토양화학 페어 (독일) | held out — OOD 벤치마크 |
| [Naylor 2017](https://github.com/raabmarie/Naylor) | 수수 가뭄 (미국 캘리포니아) | held out — 가뭄 OOD |

NEON 페어 미생물 데이터는 인지된 큐 — 현재 화학·사이트 메타데이터만 다운로드됨.

## 벤치마크 결과

> **먼저 읽기.** v6/v7 의 R² 0.95 같은 헤드라인 수치는 **상당 부분이 lab/country fingerprint 외운 결과** 였다. adversarial 디바이어스 모델 (v9) 을 학습해서 정량 측정함. **batch effect 제거된 정직한 수치는 v9 컬럼이고, 이 값을 새 lab 토양에서의 실제 진단 능력으로 봐야 함.**

각 백본별 결과 — 동결 + linear probe MLP, 5-fold CV:

| Westerfeld (in-dist.) | v6 (GPT2, raw) | v7 (GPT2, CLR) | v8 (BERT, CLR) | **v9 (BERT + adversarial)** |
|---|---|---|---|---|
| pH R² | 0.962 | 0.761 | 0.070 | **0.108** |
| 총탄소 R² | 0.932 | 0.781 | −0.008 | **−0.017** |
| 총질소 R² | 0.905 | 0.683 | 0.097 | **0.084** |

| EMP probes | v6 | v7 | v8 | **v9** |
|---|---|---|---|---|
| country probe acc (낮을수록 batch shortcut 적음) | 0.941 | 0.870 | 0.427 | **0.188** |
| biome random-split acc | 0.935 | 0.897 | 0.632 | **0.305** |
| LOCO mean (cross-country biome acc) | 0.562 | 0.488 | 0.305 | **0.263** |

기타 task (아직 v6/v7 — v9 재실행 대기):

| 작업 | 데이터셋 | v6 점수 | RF |
|---|---|---|---|
| 가뭄 분류 (대륙간) | Naylor (USA Sorghum) 623 | acc 0.944 / AUC 0.970 | acc 0.920 / AUC 0.951 |
| 수확량 회귀 | USDA Potato 423 | R² 0.05 | R² 0.26 |
| pH OOD | Westerfeld → Bernburg 96 | R² 0.39 | R² −0.52 |
| 총탄소 OOD | Westerfeld → Bernburg 96 | R² 0.29 | R² 0.20 |
| 총질소 OOD | Westerfeld → Bernburg 96 | R² 0.52 | R² 0.31 |

v9 가 말해주는 것:

- v9 의 country probe acc 가 **0.19** — v6 의 0.94 에서 거의 추락. CLR + BERT + adversarial 조합으로 **lab/country fingerprint 거의 완전 제거**.
- 그 대가가 큼: Westerfeld pH R² **0.962 → 0.108**, 총탄소는 **거의 0 으로 추락**, 총질소 **0.084**.
- 즉 v6 의 R² 0.95 는 "이건 Westerfeld / 독일 농경지 샘플이네 → Westerfeld 평균 pH 답함" 류의 shortcut 을 크게 포함했다는 뜻. 그 shortcut 빼면 진짜 미생물 → 토양화학 신호는 작음.
- Naylor 가뭄 결과와 Westerfeld→Bernburg OOD 수치들도 v6 에서 나온 거라 같은 shortcut 일부 포함 가능. v9 로 재실행해야 진짜 OOD 주장 가능.

본분에 대한 함의:
- 모델이 **진짜 신호를 갖긴 함** — country probe 가 majority baseline(약 0.06) 보다 훨씬 높고, v9 R² 가 pH/N 에서 양수 — 하지만 그 신호의 크기가 v6 헤드라인 수치가 시사한 것보다 훨씬 작음.
- 본 모델이 **본 적 없는 lab 의 토양** (한국 포도밭, 브라질 Cerrado) 에 대한 예측은 v6 컬럼이 아니라 v9 컬럼처럼 동작할 가능성 높음.
- 이 갭 좁히려면 (a) lab 다양성 확보, (b) abundance-aware 토큰화 (현재 vocab 은 존재 여부만), (c) 1500 step 보다 더 큰 모델·더 긴 학습 필요.

## 알려진 한계

내부 검증 결과(`scripts/validate_diagnostic_heads.py`, `scripts/leave_one_country_out.py`, `scripts/geo_embedding_analysis.py`)에서 나온 솔직한 caveat:

1. **임베딩에 batch / country 신호가 강함.** v6 임베딩 위에서 logistic regression 으로 EMP 샘플의 country 를 분류하면 5-fold CV 정확도 **0.94** (16개국, majority baseline 0.44). UMAP 시각화에서도 country 별로 거의 완벽하게 분리됨 (`docs/geo_umap.png`). 이 신호는 진짜 지리적 미생물 차이라기보단 lab/sequencing-platform fingerprint 일 가능성 높음.

2. **Leave-one-country-out (LOCO) 가 batch shortcut 가설을 확인.** 같은 biome 분류 모델을 한 country 빼고 학습 후 그 country 에서 테스트:
   - Random 80/20: acc **0.96**
   - LOCO 평균: acc **0.63**, balanced acc 0.49
   - 가장 큰 holdout (US, n=834) 에서 acc **0.54 — majority baseline (0.58) 보다 못함**.
   - 결론: in-distribution 정확도의 상당 부분이 "이건 X 나라 샘플이다" 를 인식하는 데서 나옴, 진짜 일반화 가능한 미생물 신호가 아님.

3. **Westerfeld 진단 R² (0.95/0.88/0.88) 자체는 batch shortcut 검증 불가.** Westerfeld 는 단일 사이트라 cross-site holdout 못 만듦. 가장 가까운 OOD 인 Bernburg (같은 독일, 같은 연구진) 에서 R² 가 0.59/0.72/0.73 로 떨어짐 — 진짜 일반화일 수도, 부분적 batch shortcut 일 수도 있음 (현재 데이터로는 분리 불가).

4. **사전학습 코퍼스의 지리·biome 커버리지 불균형.** v3 코퍼스 = MGnify v1 (2,887) + EMP (4,628) + NEON (2,999). NEON 은 24개 사이트 전부 미국. EMP 는 21개국이지만 미국 43%, 유럽 우세. **한국, 열대(아프리카·동남아·라틴아메리카), 건조 토양은 샘플 거의 없음.**

5. **EMP 에서 country 와 biome 이 confounded.** 일본, 호주, 몽골, 탄자니아 등은 EMP 안에서 ENVO biome 이 1개만 나옴. 즉 "country 일반화" 와 "biome 일반화" 를 데이터만으론 분리 못 함.

**실용적 함의**:
- 학습 분포에 가까운 토양(같은 lab, 같은 대륙, 비슷한 biome)에선 강함.
- 진짜 새 지역(한국 포도밭, 브라질 Cerrado)에선 batch effect 단계에서 이미 OOD — 미생물 신호 OOD 보다 더 약할 가능성.
- 다음 시도: (a) 토큰화 전 batch-correction 정규화 (CLR / log-ratio), (b) leave-one-country-out 파인튜닝, (c) 실제로 비-서구 데이터셋 추가 (한국 RDA, MGnify 의 브라질 농경지 study 등).

## 모델 아키텍처

- **베이스**: GPT-2 스타일 Transformer Decoder (HuggingFace `GPT2LMHeadModel`)
- **레이어**: 8
- **어텐션 헤드**: 8
- **임베딩 차원**: 256
- **컨텍스트 길이**: 512 토큰
- **어휘 크기**: 12,916 (특수 4개 + 약 12.9k `g__<속>` 토큰)
- **사전학습**: [MGM](https://github.com/HUST-NingKang-Lab/MGM) 가중치 → `gaia_v4` (현 공개). `gaia_v5` 는 v2(EMP 통합) 코퍼스로 추가 사전학습한 버전.

## 기술 스택

| 영역 | 도구 |
|------|------|
| 언어 | Python 3.10+ |
| 딥러닝 | PyTorch 2.x |
| 트랜스포머 | Hugging Face Transformers |
| 데이터 | Pandas, AnnData, Biom-format |
| 생물정보학 | QIIME2, Kraken2, MetaPhlAn |
| 시각화 | Matplotlib, Seaborn, UMAP |
| 실험 추적 | Weights & Biases |
| 모델 호스팅 | Hugging Face Hub |

## 전처리 파이프라인

```
[원시 데이터]
    │
    ▼
[1] 분류체계 통일 ──── GTDB r220 기준 매핑
    │
    ▼
[2] 풍부도 정규화 ──── TSS (상대적 %) 또는 CLR (로그비)
    │
    ▼
[3] 스파시티 필터링 ── 전체 샘플 0.1% 미만 출현 속 제거
    │
    ▼
[4] 메타데이터 정제 ── ENVO 온톨로지 기반 바이옴 분류
    │
    ▼
[5] 배치 효과 태깅 ── 시퀀싱 플랫폼, 추출 키트 기록
    │
    ▼
[6] 코퍼스 변환 ────── 풍부도 순 정렬 → 토큰화 (길이 512)
```

## 기여하기

[CONTRIBUTING.md](CONTRIBUTING.md) 참고. 받고 싶은 기여: 코드 수정, 표준화된 토양 미생물 데이터셋 추가, 새로운 벤치마크 과제, 생태학적 검증.

## 인용

```bibtex
@software{gaia2026,
  title={Gaia: A Foundation Model for Soil Microbiome Understanding},
  year={2026},
  url={https://github.com/Kimchikilla/ProjectGaia}
}
```

## 라이선스

이 프로젝트는 Apache License 2.0 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE)를 참고하세요.


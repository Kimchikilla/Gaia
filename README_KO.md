# Gaia: 토양 미생물 파운데이션 모델

> *Gaia — 그리스 신화의 대지의 여신. 토양 미생물의 숨겨진 언어를 해독하는 프로젝트.*

**"토양 미생물의 AlphaFold를 오픈소스로 만든다."**

Gaia는 토양 미생물 군집의 "언어"를 이해하는 파운데이션 모델입니다. 공개 메타게놈 데이터로 사전학습하여 토양 건강 진단, 수확량 예측, 미생물 컨소시엄 설계를 가능하게 합니다.

[English](README.md) | **한국어**

---

## 주요 기능

- **사전학습 파운데이션 모델**: MGnify + EMP에서 합쳐 만든 v2 코퍼스 **7,170개 시퀀스** 로 사전학습한 GPT 스타일 Transformer (8 레이어, 256d)
- **토양 건강 진단**: 미생물 프로필로부터 pH (R²=0.95), 총탄소 (R²=0.88), 총질소 (R²=0.88) 예측 — Westerfeld 인디스트리뷰션. Bernburg OOD 에서도 pH=0.59, C=0.72, N=0.73
- **사이트 간 OOD 일반화**: linear probe 5/6, zero-shot 3/3 작업에서 RandomForest 능가
- **가뭄 스트레스 탐지**: Naylor (미국 캘리포니아 수수) 데이터로 OOD 검증 — `docs/benchmark_naylor.json`
- **역설계 (컨소시엄 추천)**: 목표 토양 (pH, C, N) 입력 → 임베딩이 가장 가까운 실제 샘플 k개를 가중평균해 추천 미생물 조합 생성
- **CLI 도구**: `gaia diagnose abundance.csv` 한 줄로 JSON/Markdown 토양 보고서. `gaia design --ph 6.5 --carbon 1.8 --nitrogen 0.18` 으로 컨소시엄 추천

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

v4 백본(동결) + linear probe MLP 헤드 결과:

| 작업 | 데이터셋 | Gaia | RF | 승자 |
|---|---|---|---|---|
| pH 예측 (인디스트) | Westerfeld 192 | R² = **0.95** | — | — |
| 총탄소 (인디스트) | Westerfeld 192 | R² = **0.88** | — | — |
| 총질소 (인디스트) | Westerfeld 192 | R² = **0.88** | — | — |
| pH (OOD) | Bernburg 96 | R² = **0.59** | 0.55 | Gaia |
| 총탄소 (OOD) | Bernburg 96 | R² = **0.72** | 0.36 | Gaia (~2배) |
| 총질소 (OOD) | Bernburg 96 | R² = **0.73** | 0.73 | tie |
| pH zero-shot (Westerfeld→Bernburg) | 96 | R² = **0.39** | −0.52 | Gaia |
| 총탄소 zero-shot | 96 | R² = **0.29** | 0.20 | Gaia |
| 총질소 zero-shot | 96 | R² = **0.52** | 0.31 | Gaia |
| 가뭄 분류 (대륙간 OOD) | Naylor 623 | acc **0.944** / AUC **0.970** | acc 0.920 / AUC 0.951 | Gaia |
| 수확량 회귀 | USDA Potato 423 | R² 0.05 | **R² 0.26** | RF |

정직한 평가: 토양화학·OOD 일반화에서 Gaia 압도. 수확량은 v2 코퍼스로는 RF 에 밀림 —
yield 는 기후·관리 영향이 크고 v2 가 yield-페어 도메인을 충분히 못 봤다는 가설.

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

기여를 환영합니다! 자세한 내용은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.

### 기여 방법

- **코드**: 버그 수정, 새 기능, 파이프라인 개선
- **데이터**: 표준화 프로토콜에 따른 토양 미생물 데이터셋
- **과학**: 새로운 벤치마크 과제 제안, 생태학적 검증, 도메인 전문가 리뷰

## 커뮤니티

- **GitHub Discussions**: 기술 논의 및 Q&A
- **Discord**: 실시간 소통
- **월간 미팅**: 온라인 방향성 논의 (매월 첫째 주 목요일)

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

---

*이 프로젝트는 활발히 개발 중입니다. Star를 눌러 업데이트를 받아보세요!*

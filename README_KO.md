# Gaia: 토양 미생물 파운데이션 모델

> *Gaia — 그리스 신화의 대지의 여신. 토양 미생물의 숨겨진 언어를 해독하는 프로젝트.*

**"토양 미생물의 AlphaFold를 오픈소스로 만든다."**

Gaia는 토양 미생물 군집의 "언어"를 이해하는 파운데이션 모델입니다. 공개 메타게놈 데이터로 사전학습하여 토양 건강 진단, 수확량 예측, 미생물 컨소시엄 설계를 가능하게 합니다.

[English](README.md) | **한국어**

---

## 주요 기능

- **사전학습 파운데이션 모델**: MGnify, NEON, EMP의 10,000개 이상 토양 미생물 샘플로 사전학습한 Transformer 기반 모델
- **토양 건강 진단**: 미생물 프로필로부터 토양 화학 특성(pH, 유기탄소, 총질소) 예측
- **바이옴 분류**: 토양 바이옴 유형 식별 (농경지, 산림, 초원, 사막, 습지)
- **가뭄 스트레스 탐지**: 미생물 시그니처로부터 가뭄 스트레스 이진 분류
- **해석 가능성 도구**: 어텐션 기반 핵심 미생물(keystone genera) 식별
- **합성 데이터 생성**: 목표 토양 조건에 맞는 현실적인 미생물 풍부도 프로필 생성

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

```python
from gaia.inference import GaiaPredictor

# 사전학습 모델 로드
predictor = GaiaPredictor.from_pretrained("gaia-v0.1")

# 미생물 프로필로 토양 특성 예측
result = predictor.diagnose("path/to/abundance_profile.csv")
print(result.soil_health_report)
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

| 소스 | 설명 | 샘플 수 |
|------|------|---------|
| [MGnify](https://www.ebi.ac.uk/metagenomics/) | 토양 바이옴 분류학적 풍부도 테이블 | 5,000~15,000 |
| [NEON](https://www.neonscience.org/) | 미생물 + 환경 데이터 짝 데이터 | ~2,000 |
| [Earth Microbiome Project](https://earthmicrobiome.org/) | 표준 프로토콜 기반 전 세계 토양 샘플 | ~5,000 |
| [SMAG](https://genome.jgi.doe.gov/) | 3,304개 메타게놈에서 복원된 40,039개 토양 MAG | 참조 DB |

## 벤치마크

| 과제 | 지표 | 설명 |
|------|------|------|
| 바이옴 분류 | ROC-AUC, F1 | 미생물 프로필로 토양 바이옴 유형 분류 |
| 토양 화학 예측 | R², RMSE | pH, 유기탄소, 총질소 예측 |
| 경운 방식 분류 | Accuracy, Kappa | 경운 방식 분류 (무경운/최소/관행) |
| 가뭄 스트레스 탐지 | Accuracy, F1 | 가뭄 스트레스 이진 분류 |
| 풍부도 복원 | 코사인 유사도 | 마스킹된 미생물 프로필 복원 |

## 모델 아키텍처

- **베이스**: 다층 트랜스포머 디코더
- **레이어**: 6~12 (조정 가능)
- **어텐션 헤드**: 8~16
- **임베딩 차원**: 256~512
- **어휘 크기**: ~5,000 (토양 관련 속)
- **사전학습**: [MGM](https://github.com/HUST-NingKang-Lab/MGM) 가중치 기반 계속학습 (Continual Pre-training)

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

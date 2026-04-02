# Gaia 가뭄 탐지 기록 — 2026년 4월 2일

---

## 1. 데이터: Naylor et al. (2017)

### 논문 정보
- 제목: "Drought and host selection influence bacterial community dynamics in the grass root microbiome"
- 저널: The ISME Journal
- BioProject: PRJNA369551
- 18종 풀 식물 × 가뭄/정상 처리 × 토양 샘플

### 데이터 수집 과정

| 단계 | 방법 | 결과 |
|------|------|------|
| 메타데이터 수집 | NCBI SRA API | 880개 샘플 메타데이터 (토양 623개) |
| fastq 다운로드 | ENA HTTP 직접 다운로드 (10 병렬) | 623개 성공, 실패 0, 총 7.2GB |
| 품질 필터링 | QIIME2 DADA2 (WSL Ubuntu) | 표준 파이프라인 |
| 분류학 할당 | vsearch + SILVA 138 (99%) | 99,304개 ASV 중 98,481개 매칭 (99.2%) |
| genus 집계 | ASV → genus 합산 | **623개 샘플 × 1,001종 genus** |

### 최종 데이터

| 항목 | 값 |
|------|-----|
| 총 샘플 | 623개 |
| 가뭄 (drought) | 303개 |
| 정상 (control) | 320개 |
| genus 종류 | 1,001종 |
| MGM 어휘 매칭 | 646/1,001 (64.5%) |

---

## 2. QIIME2 처리 과정

### 설치
- WSL (Windows Subsystem for Linux) Ubuntu에 Miniconda + QIIME2 설치
- QIIME2 2024.10.1, DADA2 1.30.0

### DADA2 설정
- trunc_len: 250
- max_expected_errors: 2.0
- chimera_method: consensus
- threads: 8

### 분류학 할당
- QIIME2 내장 sklearn classifier는 메모리 초과로 실패
- vsearch --usearch_global로 대체 (SILVA 138 99% reference)
- --id 0.8, --maxaccepts 1, --top_hits_only
- 98,481/99,304 ASV 매칭 성공 (99.2%)

---

## 3. 가뭄 탐지 결과

### Random Forest (베이스라인)

| 항목 | 값 |
|------|-----|
| feature | 957개 ASV (5% 이상 출현) |
| 학습/테스트 | 498 / 125 |
| **정확도** | **92.0%** |
| **F1** | **92.0%** |

### Gaia (MGM + 미세조정)

| 항목 | 값 |
|------|-----|
| 모델 | MGM + 2,887개 토양 학습 모델 |
| 입력 | genus 이름 토큰 (MGM 어휘 646종 매칭) |
| 분류기 | 256 → 128 → 2 (33,154 파라미터) |
| 학습/테스트 | 493 / 124 |

| 에폭 | 손실 | 정확도 | F1 |
|------|------|--------|-----|
| 1 | 0.588 | 83.1% | 82.5% |
| 5 | 0.289 | 86.3% | 86.2% |
| 10 | 0.215 | 88.7% | 88.7% |
| 15 | 0.177 | 91.1% | 91.1% |
| 20 | 0.124 | **91.9%** | **91.9%** |

### 세부 결과 (에폭 20)

|  | Precision | Recall | F1 |
|---|-----------|--------|-----|
| Control | 92% | 92% | 92% |
| Drought | 92% | 92% | 92% |

### 비교

| 모델 | 정확도 | F1 |
|------|--------|-----|
| Random Forest | **92.0%** | **92.0%** |
| Gaia (MGM + FT) | 91.9% | 91.9% |

**거의 동일한 성능** (0.1%p 차이).

---

## 4. 분석

### Gaia의 강점
- genus 이름의 **64.5%만 매칭**됐는데도 RF와 동등한 성능
- 매칭률을 높이면 (MGM 어휘 확장 또는 SILVA 매핑 개선) 역전 가능성
- MGM이 26만 샘플에서 학습한 미생물 "상식"이 가뭄 탐지에도 유효

### 한계
- RF는 1,001종 전부 사용, Gaia는 646종만 사용 (불리한 조건)
- 아직 RF를 확실히 이기지 못함
- 데이터가 1개 연구(Naylor)에서만 왔으므로 일반화 검증 필요

### 의미
- **실제 실험 데이터**로 검증한 첫 번째 결과
- **QIIME2 표준 파이프라인**으로 처리 → 논문에 사용 가능
- 토양 미생물 조합으로 가뭄 여부를 **92% 정확도로 판별** 가능

---

## 5. 전체 벤치마크 현황

| 과제 | 데이터 | Gaia | RF | 승자 |
|------|--------|------|-----|------|
| 바이옴 분류 (초원 vs 산림) | MGnify 569개 | **90.4%** | 83.8% | **Gaia** |
| pH 예측 (proxy) | MGnify 1,311개 | R²=0.22 | **R²=0.38** | RF |
| **가뭄 탐지 (실제)** | **Naylor 623개** | **91.9%** | **92.0%** | **무승부** |

---

## 6. 기술적 이슈 해결 과정

| 문제 | 해결 |
|------|------|
| SRA Toolkit 다운로드 실패 | ENA HTTP 직접 다운로드로 전환 |
| QIIME2 Windows 미지원 | WSL Ubuntu에 설치 |
| conda PATH 괄호 오류 | -e bash -c 로 직접 실행 |
| conda ToS 미동의 | conda tos accept 실행 |
| Python 3.13 핀 충돌 | 별도 conda 환경 (python=3.10) 생성 |
| SILVA sklearn classifier 메모리 초과 | vsearch --usearch_global로 대체 |
| vsearch QIIME2 래퍼 시간 초과 | vsearch 직접 실행 |
| unzip 미설치 | Python zipfile로 추출 |

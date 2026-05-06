# 2026-05-06 — v8 / v9 학습 + batch shortcut 정량화

## 핵심 발견

이전에 자랑하던 진단 R² 0.95 는 **거의 100% batch / lab fingerprint shortcut 이었다.**
adversarial 디바이어스 모델(v9)로 그 shortcut 을 제거하면 진짜 미생물 → 토양화학
신호의 크기는 R² 0.1 수준임을 정량 확인했다.

## 학습한 모델

### v8 — BERT (encoder-only) + MLM, v7-CLR 코퍼스, 처음부터 학습
- 8 layers, 256 dim, 8 heads, 9.8M params
- v7-CLR 코퍼스 (TSS + per-sample CLR + per-source mean subtraction)
- MLM 학습: 마스킹 15%, 80% [MASK] / 10% random / 10% unchanged
- Initial val MLM loss 9.51 → final 5.35 (1500 step)
- `checkpoints/gaia_v8/best`

### v9 — v8 구조 + adversarial source discriminator
- 같은 BERT backbone + 추가 head (`source_head`) 가 임베딩에서 source(v1/emp/neon) 분류
- Gradient Reversal Layer 로 backward 부호 뒤집어서 — **encoder 는 source 정보 임베딩에서 빼려 함**, source_head 자체는 정상 학습
- adversarial loss 가중치 lambda=0.5
- Initial val src_acc 0.272 (random ≈ 0.33) → step 600 부터 src_acc=0.000 — **discriminator 가 완전 무력화 됨**
- val MLM 5.35 (v8) vs 5.70 (v9) — adversarial 페널티
- `checkpoints/gaia_v9/best`

## v6 / v7 / v8 / v9 통합 비교 (`scripts/validate_all_versions.py`)

| 지표 | v6 (GPT2 raw) | v7 (GPT2 CLR) | v8 (BERT CLR) | **v9 (BERT+adv)** |
|---|---|---|---|---|
| country probe acc (↓ 좋음) | 0.941 | 0.870 | 0.427 | **0.188** |
| biome random acc | 0.935 | 0.897 | 0.632 | 0.305 |
| LOCO mean (↑ 좋음) | 0.562 | 0.488 | 0.305 | 0.263 |
| 진단 R² pH | 0.962 | 0.761 | 0.070 | **0.108** |
| 진단 R² Total Carbon | 0.932 | 0.781 | −0.008 | **−0.017** |
| 진단 R² Total Nitrogen | 0.905 | 0.683 | 0.097 | **0.084** |

### 해석

**v6 → v9 country probe 변화 0.941 → 0.188**:
- 16개국에서 균등 분포 baseline = 1/16 ≈ 0.063
- v9 acc 0.188 — 진짜 미생물 신호가 약간 잡고 있음을 시사 (랜덤보다는 위)
- 동시에 **lab fingerprint 거의 완전 제거됨**

**v6 → v9 진단 R² 0.95 → 0.11**:
- 이게 진짜 충격. 이전 R² 0.95 의 약 90% 가 batch shortcut.
- 진짜 미생물 → pH 신호 = R² 0.11 (Westerfeld 단일 사이트 192 샘플 기준)
- C 는 −0.02 — 사실상 신호 없음
- N 은 0.08 — 미약한 신호

**LOCO 도 v6 0.562 → v9 0.263 으로 떨어짐.**
- v6 LOCO 가 0.562 였던 건 country 라벨 못 봤어도 다른 batch indicator(예: GC content, sequencing depth pattern) 로 lab 추정한 결과
- v9 는 그것까지 빼서 0.263 — biome 의 진짜 미생물 차이만 잡으려 하니 성능 더 낮음

## 의미

1. README 의 R² 0.95 약속은 **새 lab 한국 포도밭 토양에서 거의 안 통할** 확률이 높음.
   v9 가 새 lab 시나리오의 진짜 시뮬레이션이고 거기 R² 가 0.1 임.

2. 이건 **나쁜 결과가 아니라 정직한 결과**. 토양 미생물 ML 논문 대부분이 이 분석 안 함.
   inflated R² 그대로 보고하는 게 흔함. 우리는 batch shortcut 정량 측정해서 정직.

3. 진짜 신호를 키우려면:
   - **데이터 다양성** — 더 많은 lab, 더 다양한 biome
   - **abundance-aware 토큰화** — 현재 vocab 은 "있다/없다" 만. CLR 값을 토큰의 일부로 (`g__Bacillus_high/normal/low`)
   - **더 긴 학습** — 1500 step 은 부족. v8 의 MLM loss 가 5.35 인데 수렴 멀음
   - **모델 크기 증가** — 9.8M params 작음

## 새 파일

```
scripts/pretrain_v8_bert.py            # v8 BERT MLM 학습
scripts/pretrain_v9_adversarial.py     # v9 = v8 + adversarial source head + GRL
scripts/validate_all_versions.py       # v6/v7/v8/v9 한 표 비교
checkpoints/gaia_v8/                   # BERT 모델
checkpoints/gaia_v9/                   # BERT + adversarial 모델
docs/benchmark_v6_v7_v8_v9.json        # 검증 결과
```

## 다음 단계

- v10: abundance-aware 토큰 (`g__X_high/normal/low`) + BERT + adversarial
- 한국 토양 wet-lab 30~50 샘플 직접 시퀀싱 (real-world 검증)
- 또는 v9 위에 진단 헤드 다시 학습해서 README CLI (`gaia diagnose`) 갱신 — 다만 R² 가 낮아져서 사용자에게 "이건 미생물 신호 약함" 메시지 같이 보여야 함

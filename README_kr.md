# 🚗 Jetson Nano 기반 자율주행 RC카 프로젝트

> **딥러닝 + 임베디드 시스템 + 하드웨어 제어**를 통한 라인트래킹 RC카 구현  
> 🎓 졸업 프로젝트 / 공모전 / 포트폴리오용 최적화

---

## 🧠 프로젝트 개요

Jetson Nano와 카메라, 모터를 활용하여 도로의 라인을 실시간으로 인식하고  
CNN 기반의 딥러닝 모델로 조향 각도를 예측하여 RC카를 자율주행시키는 프로젝트입니다.

- 라인 인식 → 조향 제어 → 속도 조절까지 전체 자동화
- 실시간 영상 처리 + 하드웨어 제어 동시 수행 (멀티쓰레딩 기반)

---

## 🎥 시연 영상

> 실제 자율주행에 성공한 결과 영상입니다 (Jetson Nano + 모델 직접 학습)

[![시연 영상](https://img.youtube.com/vi/영상링크ID/0.jpg)](https://github.com/mj99k14/Autonomous-Vehicle-Project/blob/main/KakaoTalk_20241217_204918887.mp4)  
🔗 [영상 파일 직접 보기 (mp4)](https://github.com/mj99k14/self-driving-rc-car/raw/main/video/jetson_autopilot_demo.mp4)


---

## 🛠 기술 스택 및 사용 도구

| 분류 | 도구 |
|------|------|
| 💻 언어 | Python 3.8 |
| 🎥 영상 처리 | OpenCV 4.10.0 |
| 🧠 딥러닝 | PyTorch, PilotNet 구조 |
| ⚙️ 하드웨어 | Jetson Nano, 서보모터, DC모터, L298N |
| 🧪 OS & 라이브러리 | Ubuntu 18.04 + JetPack 4.6.1, CUDA 10.2, cuDNN 8.2 |

---

## 📂 프로젝트 구조

```
Autonomous-Vehicle-Project/
├── model/             # 학습된 모델
├── data/              # 수집한 이미지 및 라벨링 데이터
├── src/
│   ├── camera.py
│   ├── model_infer.py
│   ├── steering.py
│   ├── speed.py
├── utils/
├── test/
└── README.md
```

---

## 💡 핵심 구현 기능

### 📷 실시간 라인 인식
- OpenCV로 영상 수집 및 전처리
- Canny edge detection + CNN 기반 예측

### 🕹 조향/속도 제어
- 서보모터로 방향 조절 (PWM 신호)
- DC모터로 속도 제어 (정방향/역방향 + 가속/감속)

### 🧵 멀티쓰레드 시스템
- 영상 인식과 모터 제어를 동시에 비동기 실행

---

## 📊 학습 및 성능

- 모델 구조: CNN 기반 PilotNet (5 Conv + 4 FC)
- 학습 데이터: 약 13,362장 (5종 조향 각도 라벨)
- 최종 Validation Loss: `0.2107`
- 자율주행 성공률 (테스트 기준): 약 78.8%
- 향후 성능 최적화 완료 모델: `pilotnet_model_optimized.pth`

---

## 📌 프로젝트 성과

✅ 실내 트랙 자율주행 성공  
✅ 곡선 구간 주행 개선 (데이터 증강 + 튜닝)  
✅ GPU 가속 기반 실시간 영상 추론 적용  
✅ Jetson GPIO 제어 + PWM 신호 활용한 정밀 제어

---

## 📎 기타 참고 자료

- GitHub 저장소: [프로젝트 코드 보러가기](https://github.com/mj99k14/Autonomous-Vehicle-Project)

---

## 🙋 만든 사람

| 이름 | 역할 |
|------|------|
| 김민정 | 전체 시스템 설계, 모델 학습, Jetson 제어 |

# **BCCDViwer (C# + ONNX 버전)**

Windows 환경에서 **ONNX Runtime**을 활용하여  
혈액 세포 이미지(BCCD Dataset)를 **실시간 객체 탐지**하고,  
**검출 결과를 시각화**하는 .NET Framework 4.8.1 기반 C# WinForms 툴입니다.  

Python(MMDetection)에서 학습한 모델(`.pth`)을 **ONNX 형식**으로 변환하여,  
C# 환경에서 CPU/GPU 추론이 가능하도록 구현되었습니다.  

---
<img width="1136" height="713" alt="image" src="https://github.com/user-attachments/assets/06fcf62d-9e48-42bb-9cfb-f0e55567d4a7" />

## 📦 프로젝트 개요

- **플랫폼:** Visual Studio (.NET Framework 4.8.1)  
- **추론 엔진:** Microsoft.ML.OnnxRuntime  
- **목적:** 혈액 세포 이미지에서 WBC, RBC, Platelets 객체 탐지 및 시각화  
- **데이터셋:** BCCD Dataset (COCO 형식 변환)  
- **결과물:** 추론 결과 오버레이 이미지, 검출 좌표·클래스 테이블

---

## ✅ 주요 기능

### 1. 📂 ONNX 모델 로드 & 추론
- Python(MMDetection + MMDeploy)에서 변환한 ONNX 모델 불러오기
- CPU/GPU 장치 선택 지원

### 2. 🖼️ 이미지 추론 & 오버레이
- 원본 비율 유지 + 패딩 후 모델 입력 전처리
- **클래스별 색상 고정**:
  - **WBC:** 밝은 초록
  - **RBC:** 빨강
  - **Platelets:** 파랑
- 탐지 결과를 원본 좌표계로 복원하여 오버레이

### 3. 🔍 테이블 결과 표시
- 탐지된 객체별:
  - 클래스 ID / 이름
  - 좌표(x1, y1, x2, y2)
  - 신뢰도(Score)
- DataGridView로 즉시 표시

### 4. ⏱ 추론 시간 측정
- ONNX 추론 실행 시간을 `Trace.WriteLine()`으로 출력하여 성능 분석 가능

### 5. 💾 이미지 로드 & 변환
- Bitmap 기반 이미지 로드
- 자동 전처리(Resize, Padding, Normalize) 포함

---

## 🧰 사용 방법

1. Python(MMDetection)에서 모델 학습 후 `.pth` 생성  
2. MMDeploy를 사용하여 ONNX 모델(`end2end.onnx`)로 변환  
3. C# 프로젝트 실행 후 ONNX 파일 경로 지정  
4. 추론할 이미지 로드 → `Run` 버튼 클릭  
5. 결과 오버레이 및 테이블 확인

---

## 🔧 개발 환경 및 라이브러리

| 구성 요소 | 내용 |
|------------|------|
| 언어 | C# (.NET Framework 4.8.1) |
| 추론 엔진 | Microsoft.ML.OnnxRuntime |
| UI 라이브러리 | WinForms |
| 이미지 처리 | System.Drawing |
| 모델 형식 | ONNX (MMDeploy 변환) |
| 실행 환경 | Windows 10/11 |

---

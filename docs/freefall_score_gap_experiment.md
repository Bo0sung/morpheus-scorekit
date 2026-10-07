# 자유낙하 물리위반 점수 차이 진단안

## 현재 결론

정상/위반 영상의 점수 차이가 작은 현상은 영상에 차이가 없어서라기보다 현재 평가 과정에서
위반 신호가 여러 번 희석되기 때문이라는 가설이 가장 강하다.

1. 중력 개입은 전체 자유낙하 중 일부 프레임에만 존재한다.
2. `falling_ball`의 세 물리 하위 점수 중 중력 위반에 직접 반응하는 값은 주로
   `acceleration_conservation`이다.
3. 세 하위 점수의 평균과 Physical/Statistical 평균을 거치면서 차이가 다시 줄어든다.
4. 각 궤적을 별도로 0~1 시간축에 맞추면 위반 영상의 더 긴 낙하시간 정보가 사라진다.
5. 픽셀 좌표의 고정 스케일과 PINN의 9.8 가정이 맞지 않아 정상 Statistical Score도 바닥에
   가까워질 수 있다.
6. `max(1-NMSE, 0)` 바닥값 때문에 위반 강도 사이의 차이가 더 이상 표현되지 않을 수 있다.

## 바로 제시할 비교표

| 비교 | 시간 | 좌표 | 구간 | 목적 |
|---|---|---|---|---|
| 원본 Morpheus | 궤적별 0~1 | 영상 픽셀/고정 스케일 | 전체 및 최적 창 | 논문 구현 재현 |
| 보정 진단 | 실제 초 | Kubric world z (m) | 두 영상 공통 pre-contact | 시간·스케일 왜곡 제거 |
| 개입 진단 | 실제 초 | Kubric world z (m) | intervention frame만 | 전체 평균 희석 확인 |

보정 진단값은 `exp(-mean(abs(a_z - g))/abs(g))`로 둔다. 이 값은 새 표준 점수가 아니라 원인을
검증하기 위한 보조 지표다. 원본 Morpheus 결과를 반드시 함께 보고한다.

## 실험 순서

1. 동일 seed의 normal/violation pair를 만든다.
2. 원본 `trajectory_freefall_gt.npz`에 Morpheus 점수를 계산한다.
3. `state.npz`로 동일 프레임·실제 시간·월드 좌표 진단을 계산한다.
4. 전체 공통 구간과 개입 구간의 차이를 비교한다.
5. `gravity_scale=0.9, 0.7, 0.5, 0.2, 0.0`으로 dose-response를 반복한다.
6. 원본 점수, acceleration 하위 점수, 보정 진단값의 단조성을 비교한다.

## PhyCo 데이터의 역할

PhyCo `ball_drop_v2`는 정상적인 PyBullet 물리 안에서 restitution을 변화시킨 자유낙하/바운스
영상이다. 따라서 다음에는 적합하다.

- 다양한 외형과 restitution에서 정상 점수 분포 측정
- SAM2 영상 궤적과 제공되는 world-space trajectory의 오차 검증
- 정상 데이터에서 보정 진단값이 안정적으로 높은지 확인

반면 matched physical violation 데이터는 아니므로 정상/위반 민감도 실험을 대체할 수 없다.
그 실험에는 현재 Kubric pair의 개입 영상이 계속 필요하다.

PhyCo 영상 전체에는 바운스와 정지 구간이 들어 있으므로 `falling_ball` 점수에는 첫 바닥 접촉
직전까지만 사용한다. `scripts/run_phyco_sample.sh`가 영상 추적, pre-contact 절단, 점수 계산을
순서대로 실행한다. 영상 기반 접촉 추정은 정적 카메라와 아래 방향 운동을 가정하므로, 최종
분석에서는 `animation_data.pkl`의 월드 좌표로 접촉 프레임을 교차 검증한다.

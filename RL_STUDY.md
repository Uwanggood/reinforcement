# RL Study Notes

이 프로젝트는 강화학습을 이론 암기보다 직접 구현하면서 RL 루프를 이해하기 위한 공부 프로젝트다.

## 진행 방식

- 사용자가 먼저 직접 구현한다.
- 정답 코드를 바로 제시하지 않는다.
- 사용자가 코드를 작성하면 짧게 리뷰한다.
- 틀린 부분은 길게 설명하지 말고 수정 방향만 짧게 알려준다.
- 필요한 학습 상태, 단계, 결정사항은 이 파일에 계속 갱신한다.
- 채팅이 길어지면 이 파일을 최신 상태로 정리한 뒤 새 채팅을 열도록 안내한다.

## 단계 계획

1. 기본 개념 정리: 완료
2. GridWorld 환경 구현: 진행 중
3. 랜덤 에이전트
4. Q-table
5. Q-learning 업데이트
6. epsilon-greedy
7. 파라미터 실험
8. 고철 스크랩 캡처 타이밍 환경 설계

## 1단계 개념

- `agent`: 행동을 고르는 주체
- `environment`: agent가 상호작용하는 세계
- `state`: 현재 상황 정보
- `action`: 선택 가능한 행동
- `reward`: action 후 즉시 받는 점수
- `policy`: state를 보고 action을 고르는 규칙 또는 전략
- `step`: 행동 한 번
- `episode`: 시작부터 종료까지 한 판
- `return`: episode 동안 받은 reward의 누적합

## 2단계: GridWorld 환경

이번 단계에서는 Q-learning, DQN, PPO를 하지 않는다. 오직 환경 인터페이스만 만든다.

### 명세

- `size = 5`
- `start = (0, 0)`
- `goal = (4, 4)`
- `state`: 현재 위치 튜플
- `action`
  - `0`: up
  - `1`: down
  - `2`: left
  - `3`: right

### `reset()` 명세

- `state`를 `start`로 초기화한다.
- 현재 `state`를 반환한다.

### `step(action)` 명세

- `action`에 따라 위치를 이동한다.
- 벽 밖으로 나가면:
  - 위치 그대로
  - `reward = -0.1`
  - `done = False`
- goal에 도착하면:
  - `reward = 1.0`
  - `done = True`
- 그 외 정상 이동이면:
  - `reward = -0.01`
  - `done = False`
- 반환값:
  - `(next_state, reward, done)`

## 현재 상태

- 현재 구현 파일: `grid_world.py`
- `Action` enum과 `GridWorld` 클래스가 있다.
- `reset()`은 구현되어 있다.
- action별 이동 방향은 `(row, col)` 좌표계 기준으로 정리되어 있다.
- `step(action)`은 `(state, reward, done)` 형태를 반환하기 시작했다.
- 현재 `reward`가 즉시 보상이 아니라 누적 보상처럼 구현되어 있어 명세와 다르다.

## 다음 수정 포인트

- `step(action)`의 `reward`는 누적값이 아니라 해당 step의 즉시 보상으로 반환한다.
- 보상 값은 명세대로 goal `1.0`, 벽 `-0.1`, 일반 이동 `-0.01`을 사용한다.
- 벽 밖으로 나가는 경우 위치를 그대로 두고 `reward = -0.1`, `done = False`를 반환한다.
- goal 도착 여부는 이동 후 state 기준으로 판단한다.

## 리뷰 기준

- `state`: 현재 위치가 정확히 관리되는가?
- `action`: `0`, `1`, `2`, `3`의 의미가 명세와 일치하는가?
- `reward`: 벽, goal, 일반 이동 보상이 정확한가?
- `done`: goal에서만 `True`가 되는가?
- 반환값: `(next_state, reward, done)` 순서가 맞는가?

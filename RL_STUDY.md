# RL Study Notes

프로젝트 경로: `C:\Users\thdwo\IdeaProjects\reinforcement`

이 프로젝트는 강화학습을 이론 암기보다 직접 구현하면서 RL 루프를 이해하기 위한 공부 프로젝트다.

## 진행 방식

- 사용자가 먼저 직접 구현한다.
- 정답 코드를 바로 제시하지 않는다.
- 사용자가 코드를 작성하면 짧게 리뷰한다.
- 틀린 부분은 길게 설명하지 말고 수정 방향만 짧게 알려준다.
- 사용자가 원할 때만 이 파일을 정리/갱신한다.
- 채팅이 길어지면 이 파일을 최신 상태로 정리한 뒤 새 채팅을 열도록 안내한다.
- 설명은 강의식으로 길게 하지 말고, 가능한 대화형 질문으로 진행한다.

## 단계 계획

1. 기본 개념 정리: 완료
2. GridWorld 환경 구현: 완료에 가까움
3. 랜덤 에이전트: 진행 중
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
- 좌표계: `(row, col)`
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

## 3단계: 랜덤 에이전트

랜덤 에이전트는 가능한 action 중 하나를 무작위로 고르는 agent다. 아직 학습하지 않는다.

### 현재 이해한 내용

- `environment`: action을 받아 `next_state`, `reward`, `done`을 계산한다.
- `agent`: action을 고른다.
- 랜덤 에이전트는 state를 받아도 사용하지 않는다.
- 랜덤 에이전트는 `Action.UP`, `Action.DOWN`, `Action.LEFT`, `Action.RIGHT` 중 하나를 동일 확률로 고른다.
- episode는 `done == True`가 되거나 `max_steps`에 도달하면 끝난다.
- episode 동안 받은 reward는 `total_reward`로 누적해서 확인한다.

### 현재 구현 상태

- `grid_world_enum.py`
  - `Action`이 `IntEnum`으로 정의되어 있다.
  - `Action.random()`이 랜덤 action을 반환한다.
- `agent.py`
  - `RandomAgent`가 있다.
  - `choose_action()`이 랜덤 action을 반환한다.
  - 현재 `max_step`, `current_step`, `iterable`도 agent 내부에서 관리한다.
- `grid_world.py`
  - `GridWorld`가 있다.
  - `step()`은 `(state, reward, done)`을 반환한다.
  - 아래쪽에 랜덤 에이전트 실행 루프가 직접 들어 있다.
  - `total_reward` 역할의 `t_r`을 누적한다.

### 다음에 볼 점

- `reset()`에서 `self.done = False`도 함께 초기화하는 것이 좋다.
- `RandomAgent`는 action 선택만 담당하고, `max_steps`와 episode 루프는 실행 코드가 관리하는 구조가 더 역할이 분명하다.
- `grid_world.py`는 환경 클래스와 실행 코드가 섞여 있으므로, 다음 단계 전에 실행 코드를 `run_random_agent.py` 같은 파일로 분리하는 것을 고려한다.
- 랜덤 에이전트 한 episode를 여러 번 실행해서 `step_count`, `total_reward`, 성공 여부가 매번 달라지는 것을 관찰한다.

## 다음 공부 주제

다음 순서로 가면 자연스럽다.

1. 랜덤 에이전트 루프 정리
   - `env.reset()`
   - `agent.choose_action()`
   - `env.step(action)`
   - `total_reward += reward`
   - `done` 또는 `max_steps`로 episode 종료
2. 여러 episode 실행
   - 랜덤 에이전트가 얼마나 자주 goal에 도착하는지 보기
   - 평균 step 수와 평균 total reward 보기
3. Q-table 개념
   - `Q[state][action]`이 무엇을 의미하는지 이해하기
   - 처음에는 모든 값을 0으로 시작하기
4. Q-learning 업데이트
   - 한 번의 경험 `(state, action, reward, next_state, done)`으로 Q값을 조금 수정하기
5. epsilon-greedy
   - 가끔 랜덤 탐험하고, 가끔 Q값이 가장 큰 action 선택하기

## 리뷰 기준

- `state`: 현재 위치가 정확히 관리되는가?
- `action`: `0`, `1`, `2`, `3`의 의미가 명세와 일치하는가?
- `reward`: 벽, goal, 일반 이동 보상이 정확한가?
- `done`: goal에서만 `True`가 되는가?
- `return`: episode 동안 받은 reward 누적합을 따로 계산하는가?
- 역할 분리: environment는 결과 계산, agent는 action 선택을 담당하는가?

## 다음 AI에게 전달할 말

아래 내용을 새 채팅에 붙여 넣으면 된다.

```text
나는 강화학습을 직접 구현하면서 공부 중이다. 프로젝트 경로는 C:\Users\thdwo\IdeaProjects\reinforcement 이다.

진행 방식:
- 정답 코드를 바로 주지 말고 내가 먼저 구현하게 해줘.
- 내가 코드를 작성하면 짧게 리뷰해줘.
- 긴 강의식 설명 말고 대화형 질문으로 공부시켜줘.
- 틀린 부분은 짧게 지적하고 수정 방향만 알려줘.
- 내가 요청할 때만 RL_STUDY.md를 갱신해줘.

현재 상태:
- 1단계 기본 개념은 완료했다.
- 2단계 GridWorld 환경은 거의 완료했다.
- 3단계 랜덤 에이전트를 진행 중이다.
- grid_world.py, agent.py, grid_world_enum.py를 봐줘.

다음으로 하고 싶은 것:
- 랜덤 에이전트 루프를 더 깔끔하게 정리하고 싶다.
- environment와 agent 역할 분리를 이해하고 싶다.
- 그 다음 Q-table로 넘어가고 싶다.

먼저 코드 상태를 확인하고, 내가 다음에 무엇을 직접 고치면 좋을지 짧게 물어봐줘.
```

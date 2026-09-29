"""
조건부 제어량의 '분포'를 담는 분위수 앙상블.

왜 점 예측이 아니라 분포인가:
  ESS 흡수량은 sum(min(제어량, cap))이다. min()은 비선형이라 E[min(X, cap)] != min(E[X], cap).
  조건부 평균 하나만 예측하면 꼬리가 눌려(RF shrinkage) 계열이 평탄해지고, 22.5MW 상한에
  덜 걸려 흡수율이 크게 부풀려진다 — 임계값을 어떻게 고르든 고쳐지지 않았다(README (b) 참고).
  그래서 시간별로 분위수 19개(0.05~0.95)를 예측해 분포를 만들고, min()을 분포 위에서 적분한다.

  E[X]          ~= mean_q(quantile_q)
  E[min(X,cap)] ~= mean_q(min(quantile_q, cap))

  분위수 평균은 분포의 평균에 대한 이산 근사다(균등 격자 19개 -> 사실상 5~95% 구간의 평균).
  꼬리 양 끝 5%를 버리므로 아주 극단적인 값은 여전히 과소평가된다.
"""
from __future__ import annotations

import numpy as np

QUANTILES: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05 ~ 0.95


class QuantileEnsemble:
    """분위수별로 따로 학습한 회귀모델 묶음. 예측 후 시간별로 정렬해 분위수 역전(crossing)을 없앤다."""

    def __init__(self, models: dict[float, object], quantiles: tuple[float, ...] = QUANTILES):
        self.models = models
        self.quantiles = tuple(quantiles)

    def predict_quantiles(self, X) -> np.ndarray:
        """(n_samples, n_quantiles) — 행마다 오름차순 정렬되고 0 이상으로 잘린다.

        분위수별 모델을 독립 학습하면 q=0.6 예측이 q=0.7 예측보다 큰 역전이 생길 수 있다.
        단조성은 분위수의 정의라, 행 단위 정렬로 강제한다(rearrangement — 이 보정은
        분위수 손실을 악화시키지 않는 것이 알려져 있다).
        """
        preds = np.column_stack([self.models[q].predict(X) for q in self.quantiles])
        return np.sort(np.clip(preds, 0, None), axis=1)

    def expected_value(self, X) -> np.ndarray:
        """E[X] — 분위수 평균."""
        return self.predict_quantiles(X).mean(axis=1)

    def expected_capped(self, X, cap: float) -> np.ndarray:
        """E[min(X, cap)] — 분포 위에서 min()을 적분한 값."""
        return np.minimum(self.predict_quantiles(X), cap).mean(axis=1)

    # 기존 회귀모델과 같은 인터페이스로 쓰기 위한 별칭 (predict = 조건부 기댓값)
    def predict(self, X) -> np.ndarray:
        return self.expected_value(X)


class ResidualEnsemble:
    """점 예측 + 경험적 잔차 분포. 분위수 방식과 비교하기 위한 대조군.

    보유 구간(2022)에서 얻은 잔차(실측 - 예측)를 예측값에 더해 분포를 만든다.
    잔차가 특징값과 무관하다(등분산)고 가정하는 단순한 방식이다.
    """

    def __init__(self, model, residuals: np.ndarray, center: bool = True):
        """center=True면 잔차의 평균을 빼서 수준(level) 편향을 제거한다.

        잔차를 뽑는 모델(2021 학습)은 배포 모델(2021~2022 학습)보다 약해서 잔차 평균이
        +19.1MWh로 치우쳐 있다. 그대로 더하면 총 제어량이 +48% 부풀고, 0에서 자르는 것까지
        겹쳐 더 커진다. 중심화하면 '퍼짐'만 빌려오고 수준은 배포 모델 예측을 따른다 —
        총 제어량 오차가 +48.0% -> +2.9%로 줄고 흡수율 오차도 함께 줄어든다.
        """
        self.model = model
        residuals = np.asarray(residuals, dtype=float)
        self.residual_mean = float(residuals.mean())
        self.centered = center
        self.residuals = residuals - self.residual_mean if center else residuals

    def _dist(self, X) -> np.ndarray:
        point = np.clip(self.model.predict(X), 0, None)
        return np.clip(point[:, None] + self.residuals[None, :], 0, None)

    def expected_value(self, X) -> np.ndarray:
        return self._dist(X).mean(axis=1)

    def expected_capped(self, X, cap: float) -> np.ndarray:
        return np.minimum(self._dist(X), cap).mean(axis=1)

    def predict(self, X) -> np.ndarray:
        return self.expected_value(X)

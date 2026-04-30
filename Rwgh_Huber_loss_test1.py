import numpy as np
import pandas as pd
from itertools import combinations
from scipy.optimize import least_squares

FILE_PATH = r"median_test.xlsx"
TILE_SIZE = 0.6

anchor_pos_tile = {
    "110394ab": np.array([1.0, 4.0]),
    "e63ce2f": np.array([20.0, 7.0]),
    "8e610981": np.array([5.0, 15.0]),
    "d10485af": np.array([4.0, 27.0]),
    "d1044709(식별잘안됨)": np.array([15.0, 14.0]),
    "4e610206": np.array([14.0, 24.0]),
}

anchor_pos = {
    aid: pos * TILE_SIZE
    for aid, pos in anchor_pos_tile.items()
}

anchor_ids = list(anchor_pos.keys())

df = pd.read_excel(FILE_PATH, header=1)
df.columns = [str(c).strip() for c in df.columns]

NODE_X_COL = "Node_x"
NODE_Y_COL = "Node_y"


def estimate_position_lsm(selected_ids, measured_dist):
    def residual_func(p):
        pos = np.array(p, dtype=float)
        residuals = []

        for aid in selected_ids:
            predicted_dist = np.linalg.norm(pos - anchor_pos[aid])
            residual = measured_dist[aid] - predicted_dist
            residuals.append(residual)

        return np.array(residuals)

    init_pos = np.mean([anchor_pos[aid] for aid in selected_ids], axis=0)

    result = least_squares(residual_func, init_pos)

    return result.x


def estimate_huber_k(residuals, min_k=0.3):
    residuals = np.asarray(residuals, dtype=float)

    median_r = np.median(residuals)
    mad = np.median(np.abs(residuals - median_r))

    sigma = 1.4826 * mad
    k = 1.345 * sigma

    return max(k, min_k)


def estimate_position_huber(selected_ids, measured_dist):
    def residual_func(p):
        pos = np.array(p, dtype=float)
        residuals = []

        for aid in selected_ids:
            predicted_dist = np.linalg.norm(pos - anchor_pos[aid])
            residual = measured_dist[aid] - predicted_dist
            residuals.append(residual)

        return np.array(residuals)

    init_pos = np.mean([anchor_pos[aid] for aid in selected_ids], axis=0)

    initial_result = least_squares(residual_func, init_pos)
    initial_residuals = residual_func(initial_result.x)

    huber_k = estimate_huber_k(initial_residuals, min_k=0.3)

    result = least_squares(
        residual_func,
        init_pos,
        loss="huber",
        f_scale=huber_k
    )

    return result.x


def calculate_total_residual(estimated_pos, measured_dist):
    residuals = []

    for aid, measured in measured_dist.items():
        predicted_dist = np.linalg.norm(estimated_pos - anchor_pos[aid])
        residual = measured - predicted_dist
        residuals.append(residual)

    residuals = np.array(residuals)

    return np.sqrt(np.mean(residuals ** 2))


def estimate_position_all_lsm(measured_dist):
    usable_anchor_ids = list(measured_dist.keys())

    if len(usable_anchor_ids) < 3:
        return None

    return estimate_position_lsm(usable_anchor_ids, measured_dist)


def estimate_position_rwgh_huber(measured_dist, min_anchor_count=3):
    usable_anchor_ids = list(measured_dist.keys())

    if len(usable_anchor_ids) < min_anchor_count:
        return None, 0

    candidate_positions = []
    candidate_weights = []

    for r in range(min_anchor_count, len(usable_anchor_ids) + 1):
        for subset in combinations(usable_anchor_ids, r):
            candidate_pos = estimate_position_huber(subset, measured_dist)

            total_residual = calculate_total_residual(
                candidate_pos,
                measured_dist
            )

            epsilon = 1e-6
            weight = 1.0 / (total_residual + epsilon)

            candidate_positions.append(candidate_pos)
            candidate_weights.append(weight)

    candidate_positions = np.array(candidate_positions)
    candidate_weights = np.array(candidate_weights)

    final_pos = np.sum(
        candidate_positions * candidate_weights[:, None],
        axis=0
    ) / np.sum(candidate_weights)

    return final_pos, len(candidate_positions)


def get_measured_dist_from_row(row):
    measured_dist = {}

    for aid in anchor_ids:
        if aid in df.columns and pd.notna(row[aid]):
            measured_dist[aid] = float(row[aid])

    return measured_dist


def calc_error(estimated_pos, true_pos):
    return np.linalg.norm(estimated_pos - true_pos)


lsm_errors = []
rwgh_huber_errors = []
improvements = []

print("\n========== 샘플별 위치 추정 결과 ==========\n")

for idx, row in df.iterrows():
    if pd.isna(row[NODE_X_COL]) or pd.isna(row[NODE_Y_COL]):
        continue

    true_pos_tile = np.array([row[NODE_X_COL], row[NODE_Y_COL]], dtype=float)
    true_pos_m = true_pos_tile * TILE_SIZE

    measured_dist = get_measured_dist_from_row(row)

    if len(measured_dist) < 3:
        print(f"[샘플 {idx}] 사용 가능한 앵커가 3개 미만이라 계산 불가")
        continue

    lsm_pos = estimate_position_all_lsm(measured_dist)
    rwgh_huber_pos, comb_count = estimate_position_rwgh_huber(measured_dist)

    if lsm_pos is None or rwgh_huber_pos is None:
        continue

    lsm_error = calc_error(lsm_pos, true_pos_m)
    rwgh_huber_error = calc_error(rwgh_huber_pos, true_pos_m)

    if lsm_error > 0:
        improvement = (lsm_error - rwgh_huber_error) / lsm_error * 100
    else:
        improvement = 0

    lsm_errors.append(lsm_error)
    rwgh_huber_errors.append(rwgh_huber_error)
    improvements.append(improvement)

    print(f"[샘플 {idx}]")
    print(f"실제 태그 위치(m):        ({true_pos_m[0]:.3f}, {true_pos_m[1]:.3f})")
    print(f"LSM 추정 위치(m):         ({lsm_pos[0]:.3f}, {lsm_pos[1]:.3f})")
    print(f"Rwgh+Huber 추정 위치(m):  ({rwgh_huber_pos[0]:.3f}, {rwgh_huber_pos[1]:.3f})")
    print(f"LSM 오차:                {lsm_error:.3f} m")
    print(f"Rwgh+Huber 오차:         {rwgh_huber_error:.3f} m")
    print(f"오차 개선율:             {improvement:.2f}%")
    print(f"사용 앵커 수:            {len(measured_dist)}개")
    print(f"사용 조합 수:            {comb_count}개")
    print("-" * 60)


lsm_errors = np.array(lsm_errors)
rwgh_huber_errors = np.array(rwgh_huber_errors)
improvements = np.array(improvements)

print("\n========== 전체 정확도 비교 결과 ==========")

print(f"총 계산 샘플 수: {len(lsm_errors)}개")

print("\n[LSM]")
print(f"평균 오차:   {np.mean(lsm_errors):.3f} m")
print(f"중앙값 오차: {np.median(lsm_errors):.3f} m")
print(f"RMSE:        {np.sqrt(np.mean(lsm_errors ** 2)):.3f} m")
print(f"최대 오차:   {np.max(lsm_errors):.3f} m")

print("\n[Rwgh + Huber]")
print(f"평균 오차:   {np.mean(rwgh_huber_errors):.3f} m")
print(f"중앙값 오차: {np.median(rwgh_huber_errors):.3f} m")
print(f"RMSE:        {np.sqrt(np.mean(rwgh_huber_errors ** 2)):.3f} m")
print(f"최대 오차:   {np.max(rwgh_huber_errors):.3f} m")

mean_improvement = (
    (np.mean(lsm_errors) - np.mean(rwgh_huber_errors))
    / np.mean(lsm_errors)
    * 100
)

rmse_lsm = np.sqrt(np.mean(lsm_errors ** 2))
rmse_rwgh_huber = np.sqrt(np.mean(rwgh_huber_errors ** 2))

rmse_improvement = (
    (rmse_lsm - rmse_rwgh_huber)
    / rmse_lsm
    * 100
)

better_count = np.sum(rwgh_huber_errors < lsm_errors)

print("\n[결론]")
print(f"평균 오차 개선율: {mean_improvement:.2f}%")
print(f"RMSE 개선율:      {rmse_improvement:.2f}%")
print(f"LSM보다 좋아진 샘플 수: {better_count} / {len(lsm_errors)}개")
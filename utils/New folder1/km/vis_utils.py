import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import median_survival_times
import matplotlib.pyplot as plt
import pandas as pd
from lifelines.statistics import logrank_test

def plot_km_curve_from_scores(all_risk_scores, all_event_times, all_censorships, save_path="km_grouped.png"):
    # 清除NaN
    mask = ~np.isnan(all_risk_scores)
    scores = all_risk_scores[mask]
    times = all_event_times[mask]
    events = all_censorships[mask]

    # 按中位数划分高低风险组
    median_score = np.median(scores)
    high_risk = scores >= median_score
    low_risk = ~high_risk

    # 事件标签处理
    observed_events = (1 - events).astype(bool)

    # 新增：执行log-rank检验
    results = logrank_test(
        durations_A=times[high_risk],  # 高风险组时间
        durations_B=times[low_risk],   # 低风险组时间
        event_observed_A=observed_events[high_risk],  # 高风险组事件
        event_observed_B=observed_events[low_risk]    # 低风险组事件
    )
    p_value = results.p_value

    # 构建DataFrame
    df = pd.DataFrame({
        'time': times,
        'event': observed_events,
        'group': np.where(high_risk, 'High Risk', 'Low Risk')
    })

    # KM 拟合器
    kmf = KaplanMeierFitter()

    # 绘图
    plt.figure(figsize=(8, 6))
    for name, grouped_df in sorted(df.groupby("group"), key=lambda x: x[0], reverse=True):
        kmf.fit(durations=grouped_df["time"], event_observed=grouped_df["event"], label=name)
        kmf.plot(ci_show=True, show_censors=True)

    # 新增：在图中添加p值
    plt.text(
        x=0.7, y=0.2,  # 文本位置（相对坐标）
        s=f'Log-rank\np = {p_value:.4f}',  # 科学计数法显示：{p_value:.2e}
        transform=plt.gca().transAxes,  # 使用轴坐标
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8)  # 添加背景框
    )

    plt.title("Kaplan-Meier Curve by Risk Group")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"KM 曲线图已保存到：{save_path}")
    print(f"Log-rank检验p值：{p_value}")  # 新增输出到控制台
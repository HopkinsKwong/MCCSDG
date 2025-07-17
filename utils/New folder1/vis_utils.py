import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def plot_km_curve_from_scores(all_risk_scores, all_event_times, all_censorships, save_path="km_grouped.png"):
    """
    使用风险评分划分高低风险组，并绘制KM曲线（支持删失数据）

    Args:
        all_risk_scores (np.array): 每个样本的风险评分（越高风险越大）
        all_event_times (np.array): 每个样本的生存时间
        all_censorships (np.array): 每个样本的删失标记（1=删失，0=事件）
        save_path (str): 图像保存路径
    """

    # 清除NaN（保持与主函数一致）
    mask = ~np.isnan(all_risk_scores)
    scores = all_risk_scores[mask]
    times = all_event_times[mask]
    events = all_censorships[mask]

    # 按中位数划分高低风险组
    median_score = np.median(scores)
    high_risk = scores >= median_score
    low_risk = ~high_risk

    # 事件标签要变成布尔类型，1=删失 → event=False，0=事件 → event=True
    observed_events = (1 - events).astype(bool)

    # 构建DataFrame便于使用
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

    plt.title("Kaplan-Meier Curve by Risk Group")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"KM 曲线图已保存到：{save_path}")

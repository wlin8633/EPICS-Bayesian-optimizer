import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 設定檔案路徑 (請填入你跑完模擬後下載的 CSV 檔名)
CSV_PATH = os.path.join(os.path.dirname(__file__), 'sim-output.csv')

# 載入數據 (依據之前的預覽，檔案似乎是 Tab 分隔的)
# 若讀取失敗，可將 sep='\t' 改為 sep=','
print(f"Loading data: {CSV_PATH}")
df = pd.read_csv(CSV_PATH, sep='\t')

# 確保數據依照 shot-number 排序
df = df.sort_values('shot-number').reset_index(drop=True)

# 定義要分析的參數名稱
params = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
target = 'obj'

# 創建一個資料夾來存放輸出的圖片
plot_dir = os.path.join(os.path.dirname(__file__), 'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# ==========================================
# 1. 收斂軌跡圖 (Optimization Trace)
# ==========================================
print("Plotting 1. Optimization Trace...")
plt.figure(figsize=(10, 6))

# 計算至今為止的歷史最高分 (Current Best)
df['best_obj'] = df[target].cummax()

# 畫出每一發的真實分數 (散佈點)
plt.scatter(df['shot-number'], df[target], color='gray', alpha=0.5, label='Measured Obj')
# 畫出歷史最高分曲線 (實線)
plt.plot(df['shot-number'], df['best_obj'], color='red', linewidth=2, label='Current Best $f^+$')

plt.title('Bayesian Optimization Trace')
plt.xlabel('Shot Number (Iteration)')
plt.ylabel('Objective Value')
plt.yscale('log') # 依據你的數據，目標值差異很大，使用對數座標可能更好看。若不需要可註解掉
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '1_optimization_trace.png'))
plt.show(block=False)

# ==========================================
# 2. 平行座標圖 (Parallel Coordinates Plot)
# ==========================================
print("Plotting 2. Parallel Coordinates...")
fig, ax = plt.subplots(figsize=(12, 6))

# 為了能在同一個 Y 軸上顯示，將 7 個參數標準化到 [0, 1] 區間
df_norm = df.copy()
for p in params:
    min_val, max_val = df[p].min(), df[p].max()
    if max_val > min_val:
        df_norm[p] = (df[p] - min_val) / (max_val - min_val)
    else:
        df_norm[p] = 0.5

# 設定顏色映射 (分數越高顏色越暖/越亮)
cmap = plt.get_cmap('plasma')
norm = plt.Normalize(df[target].min(), df[target].max())

x_coords = np.arange(len(params))
for _, row in df_norm.iterrows():
    y_coords = row[params].values
    color = cmap(norm(row[target]))
    # 較低分的透明度高一點，高分的不透明，藉此凸顯高分的走勢
    alpha = 0.2 if row[target] < df['best_obj'].quantile(0.8) else 0.8
    ax.plot(x_coords, y_coords, color=color, alpha=alpha, linewidth=1.5)

ax.set_xticks(x_coords)
ax.set_xticklabels(params)
ax.set_ylabel("Normalized Parameter Value [0, 1]")
ax.set_title("Parallel Coordinates (Colored by Objective)")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
fig.colorbar(sm, ax=ax, label='Objective Value')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '2_parallel_coordinates.png'))
plt.show(block=False)

# ==========================================
# 3. 參數探索軌跡 (Parameter Trajectory over Time)
# ==========================================
print("Plotting 3. Parameter Trajectory over Time...")
fig, axes = plt.subplots(7, 1, figsize=(10, 12), sharex=True)
fig.suptitle('Parameter Exploration Trajectory', fontsize=14)

for i, p in enumerate(params):
    # 使用散佈圖，顏色代表該點的 Objective 分數
    sc = axes[i].scatter(df['shot-number'], df[p], c=df[target], cmap='plasma', alpha=0.7, s=20)
    axes[i].set_ylabel(p)
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel('Shot Number')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '3_parameter_trajectory.png'))
plt.show(block=False)

# ==========================================
# 4. 前三大重要參數的 2D 散佈矩陣 (Top 3 Important Parameters)
# ==========================================
print("Plotting 4. Top 3 Important Parameters...")
# 計算各參數與目標值的相關係數絕對值，挑出前三名
correlations = df[params + [target]].corr()[target].abs().drop(target)
top_3_params = correlations.sort_values(ascending=False).head(3).index.tolist()
print(f"Top 3 parameters correlated with Objective are: {top_3_params}")

fig, axes = plt.subplots(1, 3, figsize=(20, 4))
fig.suptitle('2D Scatter for Top 3 Correlated Parameters', fontsize=14)

# 將第一名參數分別與第二、第三名作圖，以及第二與第三名作圖
pairs = [(top_3_params[0], top_3_params[1]), 
         (top_3_params[0], top_3_params[2]), 
         (top_3_params[1], top_3_params[2])]

for i, (px, py) in enumerate(pairs):
    sc = axes[i].scatter(df[px], df[py], c=df[target], cmap='plasma', alpha=0.8, s=30)
    axes[i].set_xlabel(px)
    axes[i].set_ylabel(py)
    axes[i].grid(True, alpha=0.3)

fig.colorbar(sc, ax=axes.ravel().tolist(), label='Objective Value')
plt.savefig(os.path.join(plot_dir, '4_top3_scatter.png'))
plt.show()

print(f"\nAnalysis complete! All plots saved to: {plot_dir}")

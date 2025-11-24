import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib.widgets import Slider

def plot_3d_projection():
    fig = plt.figure(figsize=(12, 6))
    
    # --- 左图：上帝视角 (World Frame) ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("God's View (World Frame)\nGravity (Blue) never moves")
    ax1.set_xlim(-1.5, 1.5); ax1.set_ylim(-1.5, 1.5); ax1.set_zlim(-1.5, 1.5)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    
    # 绘制固定的世界重力向量 (蓝色)
    g_world = np.array([0, 0, -1])
    quiver_world = ax1.quiver(0, 0, 0, g_world[0], g_world[1], g_world[2], color='blue', linewidth=3, label='Gravity (World)')
    
    # 绘制代表机器人的坐标轴 (RGB: X, Y, Z)
    robot_frame_lines = []
    colors = ['r', 'g', 'b'] # Red=X, Green=Y, Blue=Z (Body)
    for c in colors:
        line, = ax1.plot([], [], [], color=c, linewidth=2)
        robot_frame_lines.append(line)

    # --- 右图：机器人视角 (Body Frame) ---
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Robot's View (Body Frame)\nRobot is fixed, Gravity (Magenta) moves")
    ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.5, 1.5); ax2.set_zlim(-1.5, 1.5)
    ax2.set_xlabel('Body X'); ax2.set_ylabel('Body Y'); ax2.set_zlabel('Body Z')
    
    # 绘制机器人的身体坐标轴 (固定不动)
    ax2.quiver(0, 0, 0, 1, 0, 0, color='r', linestyle='--', alpha=0.5) # Body X
    ax2.quiver(0, 0, 0, 0, 1, 0, color='g', linestyle='--', alpha=0.5) # Body Y
    ax2.quiver(0, 0, 0, 0, 0, 1, color='b', linestyle='--', alpha=0.5) # Body Z
    
    # 绘制投影后的重力向量 (品红色) - 这就是 proj！
    quiver_proj = ax2.quiver(0, 0, 0, 0, 0, -1, color='magenta', linewidth=3, label='Proj (Local Gravity)')
    
    # 文本显示 proj 数值
    text_proj = ax2.text2D(0.05, 0.95, "", transform=ax2.transAxes)

    # --- 更新函数 ---
    def update(val):
        roll = s_roll.val
        pitch = s_pitch.val
        yaw = s_yaw.val
        
        # 1. 计算旋转矩阵 (World -> Body 的旋转)
        r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
        
        # === 左图更新：旋转机器人坐标系 ===
        # 机器人的基向量在世界坐标系中的样子
        body_x = r.apply([1, 0, 0])
        body_y = r.apply([0, 1, 0])
        body_z = r.apply([0, 0, 1])
        
        # 更新左图的机器人坐标轴
        origin = [0, 0, 0]
        for i, vec in enumerate([body_x, body_y, body_z]):
            robot_frame_lines[i].set_data([origin[0], vec[0]], [origin[1], vec[1]])
            robot_frame_lines[i].set_3d_properties([origin[2], vec[2]])
            
        # === 右图更新：计算 proj (World Gravity -> Body Frame) ===
        # 这就是代码里的 proj = r.apply([0, 0, -1], inverse=True)
        # inverse=True 意味着我们把一个世界向量转换到身体坐标系下
        g_world_vec = np.array([0, 0, -1])
        proj_vec = r.apply(g_world_vec, inverse=True)
        
        # 更新右图的品红色向量
        # 注意：Matplotlib 的 quiver set_segments 比较麻烦，这里清除重绘是最简单的
        global quiver_proj
        quiver_proj.remove()
        quiver_proj = ax2.quiver(0, 0, 0, proj_vec[0], proj_vec[1], proj_vec[2], color='magenta', linewidth=3)
        
        # 更新数值显示
        text_proj.set_text(f"proj = [{proj_vec[0]:.2f}, {proj_vec[1]:.2f}, {proj_vec[2]:.2f}]")
        
        fig.canvas.draw_idle()

    # --- 滑块 ---
    ax_roll = plt.axes([0.2, 0.05, 0.6, 0.03])
    ax_pitch = plt.axes([0.2, 0.1, 0.6, 0.03])
    ax_yaw = plt.axes([0.2, 0.15, 0.6, 0.03])
    
    s_roll = Slider(ax_roll, 'Roll (X)', -180, 180, valinit=0)
    s_pitch = Slider(ax_pitch, 'Pitch (Y)', -90, 90, valinit=0)
    s_yaw = Slider(ax_yaw, 'Yaw (Z)', -180, 180, valinit=0)
    
    s_roll.on_changed(update)
    s_pitch.on_changed(update)
    s_yaw.on_changed(update)
    
    update(0) # 初始化
    plt.show()

if __name__ == "__main__":
    plot_3d_projection()
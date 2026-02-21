UR 阻抗控制（force_mode）
========================

本项目演示使用 RTDE 对 UR 机器人进行笛卡尔阻抗/柔顺控制。若 TCP 端安装了夹具/工具/工件而未设置负载，常见现象是 Z 轴方向缓慢下沉。解决方案如下：

快速解决方案
-------------
1. 在控制器侧设置负载（推荐）：
	 - 使用 `--payload-mass` 指定总质量（kg），使用 `--payload-cog` 指定质心相对 TCP 的坐标（m）。
	 - 程序启动时会调用 `setPayload(mass, cog)`，UR 控制器会进行重力补偿。
2. 或启用软件侧重力前馈：
	 - 传入 `--use-gravity-ff --payload-mass <kg>`，控制器若未设置负载，也可通过任务坐标系增加上举力抵消重力。

运行示例
--------
在 Windows PowerShell 中：

```powershell
python .\impedance_control.py --ip 192.168.1.100 --duration 60 `
	--axes [1,1,1,1,1,1] --limits [0.05,0.05,0.05,0.5,0.5,0.5] `
	--payload-mass 1.2 --payload-cog [0.0,0.0,0.08]
```

若控制器端无法成功 `setPayload`，可在软件侧加前馈：

```powershell
python .\impedance_control.py --ip 192.168.1.100 --duration 60 `
	--axes [1,1,1,1,1,1] --limits [0.05,0.05,0.05,0.5,0.5,0.5] `
	--payload-mass 1.2 --payload-cog [0.0,0.0,0.08] --use-gravity-ff
```

参数说明（新增）
--------------
- `--payload-mass`：末端负载质量（kg），包含工具+工件。
- `--payload-cog`：末端负载质心相对 TCP 坐标（m）[x,y,z]。
- `--use-gravity-ff`：启用软件重力前馈（在未成功设置控制器负载时可开启）。
- `--gravity`：重力加速度（默认 9.80665 m/s²）。

建议与注意事项
--------------
- 质心坐标务必基于当前 TCP 坐标系；若 TCP 方向改变，UR 控制器将自动处理重力投影。
- 若已成功设置 `setPayload`，通常不需要再启用 `--use-gravity-ff`，避免双重补偿导致上浮。
- 初次使用建议将 `--axes` 在 Z 方向设为 0（位控）验证静止无漂移后，再逐步开放顺从方向并适当提高 `--k-trans`。
- 为避免抖动，代码对力/力矩指令做了简单一阶滤波和饱和，必要时可调整 `--f-max`、`--tau-max`、阻尼 D。


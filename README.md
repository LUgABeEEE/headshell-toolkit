# HeadShell Toolkit

无头 Blender 头壳工具链：**参数契约 → 壳体外壳 → 开孔 / 通风 → 打印检查 → 评审渲染**。
所有几何工作都在脚本里完成，可在没有桌面的机器上跑；不依赖任何在线 3D 服务。

> 这是一个**工具与架构**仓库。具体的设计参数（尺寸、壁厚、孔阵、眼孔位置等）
> 不在仓库里，由使用者自己提供 —— 契约只规定**形状**，不附带可建的值。

## 目录

```
core/       几何内核：参数契约、拟合、打印（外壳/通风/间隙/归一化）
pipeline/   流水线契约与评测阈值（方法可复现，数值留空）
cli/        命令行入口：参数校验、STL 统计
adapters/   外部输入的适配层（图像、重建、壳体驱动）
blender/    在 Blender 内部运行的驱动脚本
tools/      独立小工具：STL 体检 / 四视角渲染 / 效果图 / 连通体审计
tests/      单测（无需 Blender）
params/     契约形状示例（全部取值为 null）
```

## 依赖

- Blender **4.2 LTS**（无头调用；`tools/blender-safe.sh` 会按仓库上一级的
  `.tools/blender-4.2.9-linux-x64/` 查找，也可用 `BLENDER_BIN` 指定）
- Python **3.11+**；`requirements.txt` 里的包装到本地目录即可（例如 `.pylib/`）
- 单测不依赖 Blender：`./run_tests.sh`，或 `PYTHONPATH=.pylib python3 -m unittest discover -s tests -t .`

## 快速开始

```bash
# 1) 描述你自己的参数（字段见 params/schema.example.json；所有值必须自己填）
python3 -m cli.headshell params-validate my_params.json

# 2) 查看一个 STL 的网格事实（面数 / 破洞 / 非流形 / 边长分布 / 连通体）
python3 tools/stl_tools.py stats model.stl

# 3) 四视角 clay 渲染 + 剪影对称（在 Blender 里跑）
tools/blender-safe.sh -b --factory-startup -P tools/stl_review_render.py -- \
    --src model.stl --out out/review --prefix model \
    --views front,three_quarter,side,back --res 1000x1300 --samples 48

# 4) 多件并排的产品图
tools/blender-safe.sh -b --factory-startup -P tools/studio_pair.py -- \
    --src a.stl --src b.stl --out out/two_up.png --res 2200x1250 --samples 160
```

## 设计原则

1. **契约里不含调好的值**：毫米级参数一律由调用方提供，缺项直接报错；
   `core/params.py` 只保留与具体设计无关的宽松合理性区间。
2. **渲染不改变几何**：评审渲染只读网格，投影/材质都在显示层。
3. **可复现**：同样的输入 + 同样的命令 = 同样的输出；渲染参数写进 `review.json`。
4. **许可边界干净**：`core/`、`pipeline/`、`adapters/`、`cli/`、`tools/` 为自有实现；
   参考第三方实现只学行为，不复制代码或网格。

## 许可

`core/`、`pipeline/`、`cli/`、`adapters/`、`tools/` 为本项目自有代码，按
`ARCHITECTURE.md` 的说明以 MIT 发布（正式 LICENSE 文件待补）。
第三方名称仅用于说明兼容性，不含其代码或资产。

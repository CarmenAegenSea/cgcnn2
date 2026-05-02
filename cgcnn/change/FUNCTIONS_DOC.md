# 函数说明 — cgcnn/change 脚本

本文件列出并简要说明 cgcnn/change 目录下脚本中定义的所有函数：作用、主要参数、返回值与副作用（读写文件、外部依赖）。

**filter_candidates.py**
- `filter_and_save(pred_csv, output_dir, attr_file=None, gap_min=1.6, gap_max=2.8)`:
  - 作用：读取预测 CSV（含 `prediction` 列），与属性表（attr_file）按 `material_id` 合并，按条件筛选候选材料并把结果写入 `output_dir/final_candidates.csv`。
  - 参数：`pred_csv`（预测文件路径）、`output_dir`（输出目录）、`attr_file`（属性表，可选，默认 repo/data/tmc_data/tmc_all_materials.csv）、`gap_min/gap_max`（带隙筛选区间）。
  - 返回：`(filtered_df, out_csv_path)`，filtered_df 为筛选后的 pandas.DataFrame，out_csv_path 为写入的 CSV 路径。
  - 副作用：创建输出目录、写 CSV。
  - 异常：若属性文件不存在抛 FileNotFoundError；若未找到形成能列抛 ValueError。

- `_find_id_col(columns)`:
  - 作用：在列名列表中查找可能表示 ID 的列（如 `material_id`, `id`, `mpid` 等），返回匹配的原列名或 `None`。
  - 用途：用于对输入 CSV 的 id 列名做兼容处理（自动重命名为 `material_id`）。

- `_find_formation_energy_col(columns)`:
  - 作用：在列名中查找包含 `formation` 与 `energy` 的列名（用于识别形成能列），返回匹配列或 `None`。

**id_prop_data.py**
- `generate_id_prop(cif_dir, output_file)`:
  - 作用：扫描指定 CIF 目录（`cif_dir`）下的 `.cif` 文件，按文件名（不含扩展名）生成 `id,0` 格式的 `id_prop.csv` 文件（占位值 0）。
  - 副作用：写入 `output_file`。

**id_prop.py**
- `generate_id_prop(cif_dir, output_file, source_csv=None)`:
  - 作用：同样扫描 CIF 目录，但如果提供 `source_csv`（属性表），尝试从中读取真实带隙并映射到每个 material_id；未匹配到的 ID 使用占位 0 并打印警告。
  - 参数：`source_csv`（可选）用于填充真实 target 值。
  - 副作用：写入 `output_file`，在控制台打印加载与缺失警告。

**parityPlot.py**
- `_read_prediction_csv(pred_csv)`:
  - 作用：读取预测 CSV，标准化列名并校验格式。如果含 `target` 与 `prediction` 列直接返回 DataFrame；若为三列无表头则按 `id,target,prediction` 赋列名；否则抛 ValueError。
  - 返回：pandas.DataFrame。

- `plot_predictions(pred_csv, output_dir)`:
  - 作用：基于预测 CSV 生成并保存若干评估图（预测-真实散点、误差分布直方图、训练损失曲线），并返回已保存文件路径列表。
  - 细节：
    - 散点图保存为 `bandgap_prediction_scatter.png`，图上显示 MAE 与 R^2；画 ±0.3 eV 带。
    - 误差分布保存为 `error_distribution.png`，带正态拟合曲线。
    - 损失曲线保存为 `loss_curve.png`，优先尝试从若干候选日志路径加载 `training_log.csv`（查找多处可能路径），若未找到则生成合成示例曲线。
  - 返回：已保存文件路径的列表。
  - 依赖：matplotlib、numpy、pandas、sklearn、scipy。

**pull_data.py**（拉取过渡金属-硫族化合物 TMCs）
- `get_api_key()`:
  - 作用：返回脚本内 `API_KEY`（若已设置且不是默认占位）或从环境变量 `MP_API_KEY` 读取。
  - 返回：API key 字符串或 None。

- `generate_chemsys_combinations(t_metals, chalcogens)`:
  - 作用：为每个过渡金属与硫族元素生成化学体系字符串（按字母排序，例如 `Mo-S`），并跳过被排除的元素。
  - 返回：化学体系字符串列表。

- `query_tmc_by_chemsys(mpr, chemsys, download_cif=True, cif_dir=None)`:
  - 作用：使用 `MPRester` 对指定 `chemsys` 执行 summary.search，提取字段构造统一字典（material_id, formula, band_gap, formation_energy_per_atom, energy_above_hull, volume, density, nsites, elements, crystal_system, spacegroup, is_stable 等）。
  - 可选行为：若 `download_cif` 为真且 `structure` 字段存在，则把 CIF 写入 `cif_dir` 并在返回的数据中加入 `cif_path`。
  - 返回：材料信息字典列表。
  - 副作用：可写 CIF 文件；内部遵循 `REQUEST_DELAY` 以减少 API 频率压力；在异常时打印错误。

- `fetch_all_tmcs(mpr, t_metals, chalcogens, download_cif=True, cif_dir=None)`:
  - 作用：遍历所有由 `generate_chemsys_combinations` 生成的化学体系，调用 `query_tmc_by_chemsys` 收集全部材料并打印进度统计。
  - 返回：合并后的材料信息列表。

- `save_data(data, output_dir)`:
  - 作用：将材料信息写为 CSV（`tmc_all_materials.csv`）和/或 JSON（`tmc_all_materials.json`），会移除不可序列化的 `structure` 字段。
  - 副作用：写入文件（受 `SAVE_CSV`、`SAVE_JSON` 控制）。

- `print_statistics(data)`:
  - 作用：在控制台打印数据概览：总数、按硫族元素计数、带隙范围与平均、形成能范围、热力学稳定计数等。

- `main()`:
  - 作用：脚本入口，获取 API key、创建输出目录、使用 `MPRester` 拉取 TMCs、保存数据并输出统计；包含异常处理与用户提示。

**pull.py**（催化剂数据拉取）
- `get_api_key()`:
  - 与 pull_data.py 中同样逻辑：优先脚本常量 `API_KEY`，其次环境变量 `MP_API_KEY`。

- `element_blacklist_filter(elements)`:
  - 作用：检查 `elements` 列表中是否包含位于 `EXCLUDED_ELEMENTS` 的元素，若有返回 True（表示应排除）。

- `fetch_materials_by_elements(mpr, elements_list, description)`:
  - 作用：按给定的元素组合列表逐个查询 materials.summary.search（限定 `energy_above_hull` 上限），并通过 `process_doc` 统一处理结果；对每个通道应用数量上限与请求延迟。
  - 返回：处理后的材料字典列表。

- `fetch_materials_by_chemsys(mpr, chemsys_list, description)`:
  - 作用：按化学体系列表查询（chemsys 精确匹配），行为与 `fetch_materials_by_elements` 类似。

- `process_doc(doc)`:
  - 作用：从 API 返回的文档对象中提取并规范化字段（元素符号列表、晶系/空间群字符串、band_gap、formation_energy_per_atom、cbm/vbm、structure 等），返回统一字典。
  - 返回：字典，供上层收集函数使用。

- `save_candidates(candidates, output_dir)`:
  - 作用：对候选材料按带隙下限（`BAND_GAP_MIN`）做过滤，写 CIF（若存在结构并且 `DOWNLOAD_CIF`=True）、并将信息分别保存为 `catalysis.csv` 与 `catalysis.json`。
  - 副作用：写入 CIF、CSV、JSON 文件并打印统计。

- `main()`:
  - 作用：脚本入口，组织多个查找通道（氧化物、硫化物、氮化物、LDH、MXene、钙钛矿、g-C3N4 等），合并去重并限制总数，最终调用 `save_candidates` 保存结果。

**split.py**
- `main()`:
  - 作用：为交叉验证将 CIF 文件均匀随机分成 5 份：
    - 验证 `YUAN_DIR`、`MASTER_CSV`、`OUTPUT_ROOT` 存在。
    - 读取 `MASTER_CSV`（无表头，格式 `id,target`），建立 id->target 的映射。
    - 收集 CIF 文件名并过滤出在总表中存在的 ID。
    - 使用 `RANDOM_SEED` 随机打乱并按近似均分方式分成 5 份。
    - 为每一折创建目录、复制对应 CIF、写折内 `id_prop.csv`（无表头），并尝试复制 `atom_init.json`（若存在）。
  - 副作用：创建/删除目录、复制文件、写入每折 `id_prop.csv`。

---

附注与建议
- 运行前请确保已安装需要的依赖（见脚本顶部 import）。
- 若要把函数打包为可复用模块，建议将脚本中带 `_` 前缀的私有辅助函数提取并写入单独的 util 模块，避免重复实现（例如 `get_api_key`、`process_doc`、`_find_id_col` 等在不同文件中有重复逻辑）。
- 注意 API 使用频率与 API_KEY 管理，若频繁拉取建议加大 `REQUEST_DELAY` 或使用本地缓存。


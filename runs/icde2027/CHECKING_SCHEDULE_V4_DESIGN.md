# M2 公平检查调度控制：执行前设计

登记日期：2026-11-05 UTC。状态：设计与合成开发验证；须由父任务独立审查并提供绑定最终文件哈希的批准记录后，才可启动正式计时。本文先于新实现和任何正式计时写入。旧审稿报告、论文、实验和负面结果保持原样。

## 1. 问题、范围与可解释性

本研究询问：在完全相同的阈值语义、候选集合、候选顺序、持久求解器、接收端与服务响应下，收益分别来自浮点预筛选，还是跨查询保留精确 dual 的常数和残差？它不预设存在收益，也不把功能正确、测试通过解释为数据库研究贡献充分。

使用已经观察过的输入，属于 **retrospective controlled timing**，不是新的独立确认样本。不得生成、替换或挑选有利查询。三条原主实验各 1,000 查询流与全部 60 条原阈值输入（各 100 查询）全部纳入。每条输入的 view.json、queries.npy、condition.json 及其 SHA256 固定在配置中；阈值保持 0.001，数组形状与原始文件一致。三条主流来自 primary_aligned_v1；阈值输入来自 threshold_workload_v2 的 54 条 engine 和 6 条 source-derived 输入。

本研究从保存的视图开始。六条原 source-to-answer 输入的视图保持一致，但不会重读 500 万事件或重新物化；它们在本研究明确标为 source-derived fixed-view serving。旧端到端结果不能被本研究取代。

## 2. 核心 2×2 与桥接

| arm | 浮点筛选 S | 精确 dual 缓存 P | 用途 |
| --- | --- | --- | --- |
| s0_p0 | 关闭 | 关闭 | 核心因素控制 |
| s0_p1 | 关闭 | 开启 | 核心因素控制 |
| s1_p0 | 开启 | 关闭 | 核心因素控制 |
| s1_p1 | 开启 | 开启 | 核心因素控制 |
| screened_auto | 开启 | 开启 | 公共 HiGHS strategy=0 强桥接控制 |
| original_prepared | 原实现 | 原实现 | 实际旧 Prepared 的描述桥接 |

四个核心 arm 均使用公共 HiGHS persistent simplex_strategy=1、threads=1、相同 objective_bound、全求解回退及精确检查器。候选固定为 **zero dual → last native dual → initial feasible primal → last native primal**。不增加旧历史池，不根据浮点值重排候选。last 仅在新的 native proposal 经过相同接受规则后更新；命中不更新。初始/zero 不计入各容量为 1 的 last 存储。

P0 与 P1 使用同一个精确核、同一个已表示的 base multiplier 和 scale，不用舍入后的乘子替换 base×scale。P0 每次 exact lower check 重建常数和残差；P1 按需保留本 checker 签发的同一状态。第一次准备、所有失败检查与失效重建都计费。primal 的全视图精确可行性状态在四个 arm 一致保留，不属于本因素。浮点筛选表示从原始浮点 multiplier 和视图独立构造；不得借用 P1 的 exact residual 免费计算，也不得让 P0 暗中保留 exact prepared 状态。

实际旧 Prepared 使用其既有组件/求解路径与历史池，并且只提供正向 lower 证据，不能作为 2×2 因素的一格。screened_auto 改变 backend strategy，也只能作为桥接。对两者完整报告判定差异、未知、成本和不利条件；不得把差异归因于孤立的 S/P 因素。

## 3. 筛选、语义与 complete fallback

S1 使用逐运算向外舍入的 binary64 区间筛选，没有人为 epsilon。对 lower 候选，仅当筛选区间上端 ≤ δ 才暂缓 exact check；对 primal，仅当区间下端 > δ 才暂缓 exact dot。非有限或无法界定的区间立即走精确核。每个加、减、乘与累加使用 nextafter 向外扩张；筛选表示和查询计算成本均计入。

筛选只允许暂缓，不授权答案，也不删除候选：

1. 保持原候选顺序；只有精确证据能返回 true/false。已证明不可能通过的 earlier lower 可在 later lower 给出精确 true 时不再计算。
2. 当 primal 给出 false 时，先精确补查全部被暂缓的 lower，以保留与 S0 相同的最佳 lower 和服务证据。若补查出现 true，与 false 矛盾即硬失败。
3. 若缓存候选尚不能决定，在任何 native call 前按原顺序补查所有暂缓候选。因此 screening 不可使基线可回答的请求落入求解器或 unknown。补查发现答案即返回。
4. 原生求解器输出永远只是 proposal；检查 dual 与全局 primal 可行性后才能更新 last 或返回。cutoff 停止但没有 exact decision 时清除 cutoff 后继续完整求解。即使完整求解后仍不能精确证明，也保留 checked lower/upper bracket 并返回 unknown；不得把 solver flag 当作答案。
5. 数值拒绝、invalid retained state、失败准备、失败检查、unknown 与回退均保留并计费。程序异常或无效证明保留已完成记录并停止，不得删除困难查询后继续统计。

筛选算法的区间必须包住独立 Fraction 计算值；开发测试覆盖随机小实例、抵消、符号变化、零、次正规数及溢出回退。核心 arm 逐请求要求判定、route、证据、native 状态路径、last generation 一致；时间和检查计数允许不同。若该更强同路径门槛失败，先停止并定位，不能按性能好坏放宽门槛。

## 4. 计费、执行顺序与重复单位

主成本：producer 累计时间 + 同一 FIFO16 持久接收端的初始化、JSON 解码和精确验证。producer 包括每 arm 新鲜视图 JSON 解码与验证、engine/native 初始化、筛选、准备、拒绝/失败检查、全部 fallback/native calls、状态更新、证明构造、服务投影和 JSON 序列化。query() 内诊断计数的开销自然计入；原生诊断文件的序列化在所有查询之后单独计量，不进入 wire 成本。嵌套计时不得重复加总；总成本以外层墙钟为准。

视图/查询文件读取、gzip 读写/解压、磁盘和网络成本不在主指标中；gzip 写入单独报告。固定视图读取不声称端到端。另做 fresh per-response stateless receiver 重放，单独报告，不能替换主指标。服务响应使用相同七字段 envelope，完整 multiplier/point 导出。桥接 lower-only 适配不编造 false 或 upper。

每条保存输入重复 3 个新的 timing rounds；每个 arm/round 都重新创建 engine 和 receiver，不跨 arm 复用任何算术或 native 状态。原输入路径中的 r0/r1/r2 是不同的既有查询序列，不是新增独立随机样本。新 timing round 是机器时间重复，不能把查询当独立样本计算 p 值。

执行限制：只用已授权 CPU4，native 与 BLAS 各 1 线程，串行 arm；不使用 GPU，不干扰其他作业。每条输入以固定 seed 2026110504 派生基础 arm 顺序，3 个 round 分别循环移动 0、2、4 位，让六 arm 各处于三个不同位置；输入在每 round 按同一已登记种子洗牌。计时前记录实际 affinity、线程环境、Python/NumPy/SciPy/HiGHS 版本、所有 runtime 哈希和输入哈希。63×3×6＝1,134 个 arm-stream，9,000×3×6＝162,000 条响应；全部执行，不因结果快慢提前停止。

## 5. 预定汇总、门槛和解释

三个组始终分开报告：原主流 3×1,000；engine 54×100；source-derived 6×100。每个新 timing round 内按输入等权几何平均成对 joint-cost ratios；54 engine 输入等价于 18 个条件各 3 个既有序列等权。全量逐输入与 Q=1/10/100（主流另含 1,000）前缀同步报告，不按表现挑维度、结构、阈值或速度更慢的对照。

固定四个条件效应：S1P0/S0P0、S1P1/S0P1、S0P1/S0P0、S1P1/S1P0；报告 log-ratio 交互项 log(S1P1/S0P1)−log(S1P0/S0P0)。桥接单独给出 S1P1/original_prepared 与 S1P1/screened_auto。全部方向、区间范围和各 round 数值公布。

唯一预定工程支持门槛：在原三条主流组，S1P1/S0P1 与 S1P1/S1P0 两个 joint-cost GM ratios 在全部 3 个新 round 均 ≤ 0.8，且完整正确性门槛通过，才称“两个因素各自达到 20% 主流工程收益目标”。否则明确不支持该联合目标，并分别如实报告方向；不得把交互作用或任一桥接替代缺失的因素证据。20% 沿用既有工程目标，不是统计显著性界限。其余组为预定外推边界检查；即使正向也不称新工作负载泛化。无 p 值、无事后功效或独立查询显著性断言。

全量精确重放门槛：每条响应由相同接收端验证当前 view/objective/threshold/outer 字段；另以不使用 producer prepared/cache 状态的 Fraction 实现检查全部已导出 lower/upper 证据。核心四臂必须逐请求相同 verdict、证据和 native/last 路径，无基线可决定答案损失、无相反决定；桥接可有能力差异，但无无效或相反已证明答案。explicit refusal 必须没有 bound/decision 并保留原因。任何无效证据、绑定不符、核心同路径失败、相反已证明答案或代码异常都会保留 partial records 并停止；不覆盖或修补正式结果，修复需要新版本和新 gate。

逐流比较必须显式接收本输入的预定查询数；六臂各自的 service/native 日志行数均须等于该数，且两种日志的 query_index 必须为严格整数并逐位置等于 0…Q−1。先完成这些完整性检查，才比较核心 native 路径与服务证据；不得让 zip 截断、空日志或桥接诊断缺失绕过门槛。

## 6. 开发与独立 gate

只新增 checking_schedule_v4.py、test_checking_schedule_v4.py、configs/checking_schedule_v4.json 及本设计/analysis/checking_schedule_v4_development*。复用既有精确证明核、native 调用和 receiver；不改旧 engine/proofs/runtime，不改原输入、结果、论文或审稿报告。

开发阶段只使用合成小视图和合成目标，覆盖四臂语义、unknown/false/equality、失败候选、筛选的完整回退、issuer/cache 隔离、同 scale 精确核、计费和 gate 拒绝。不得在正式 63 条输入上执行 query()，包括所谓 smoke timing。配置阶段允许只读元数据和计算哈希。

运行器默认不会启动正式计时。正式执行要求独立 gate JSON 的 status=approved，且 design/config/runtime 哈希与当前字节一致。开发结果、代码和 gate payload 先交父任务；父任务独立审查后才能批准。本设计能够隔离“在统一 last-only schedule 中的筛选和 exact preparation”，不能把所得差异直接回填为旧 Prepared 的原有收益分解；这是最小可辨识范围。

每次正式运行在任何查询前保存全部 gate runtime 的逐字节 source_snapshot，以及实际 design、config、gate 和项目环境锁文件。gate 同时绑定 Python、NumPy、SciPy、公共 HiGHS、SciPy 内置 HiGHS 版本及锁文件哈希。开始及结束核验当前代码、配置、输入和保存快照；每次加载输入均对实际待解码字节再次验哈希。manifest 在 complete/failed 时同步最终状态与 UTC 时间，失败保留 partial 响应、已完成 native 诊断和包含失败工作的 elapsed time。

既有 upper 子记录只有 point/claimed_upper；format/view/objective/threshold 由外层 threshold bundle 绑定，独立重放核验这些外层字段后对本次目标精算该点的上界，并严格拒绝 upper 子记录新增或缺失字段。本实验保持该 wire schema，不添加冗余的未登记子证书字段。

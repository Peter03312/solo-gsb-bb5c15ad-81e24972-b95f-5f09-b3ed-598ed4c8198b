# 校准计算不确定度传播 API

基于 FastAPI + SymPy + Pint，对测量模型 $y=f(x_1,\dots,x_n)$ 做 GUM 一阶不确定度传播：
服务端解析仅含白名单运算的公式并进行量纲校验，利用符号偏导与相关系数矩阵计算结果值、
单位、合成标准不确定度，并按输入量列出含相关项的方差贡献。

另提供**多输出联合传播**接口：对同一组输入量与同一输入协方差，按一阶雅可比计算
$C=J\Sigma J^{T}$，一次返回 2～8 个输出量的值、单位、合成标准不确定度、完整输出
协方差矩阵与输出相关系数矩阵。

另提供**不确定度逆分配**接口：把某个输入量的标准不确定度视为未知量，在完整保留相关项
的前提下反求满足目标合成标准不确定度的最大允许取值。

另提供**蒙特卡洛传播校核**接口：对非线性测量模型生成相关正态输入样本（相关矩阵特征
分解，奇异相关矩阵亦可运行），逐样本计算白名单公式，返回样本均值、标准不确定度、
2.5%/50%/97.5% 分位数及相对一阶传播结果的偏差，用于校核 GUM 线性化近似。

另提供**规格符合性判定**接口：在一阶传播结果 $(y,u_c)$ 上假定真值服从 $N(y,u_c^2)$，
稳定计算落入闭规格区间的概率 $p$（保留可表示的非零尾部概率），按置信阈值 $t$ 判定
合格/不合格/待定，并给出固定 $u_c$ 时使 $p\ge t$ 的接受区间。

另提供**重复读数 A 类标准不确定度**接口：对同一被测量的等间隔重复读数先去均值，
按 1/n 归一化计算各滞后自协方差 $\gamma_k$，再用 Bartlett 三角权重估计均值的
A 类标准不确定度 $u_A$（修正序列自相关对朴素 $s/\sqrt n$ 的影响），返回均值、
$u_A$、各滞后 $\gamma_k$ 与有效样本量 $n_{\mathrm{eff}}$；$u_A$ 可直接填入
`/api/v1/propagate` 的 `standard_uncertainty`。

## 启动

单条命令一键启动（Docker Compose）：

```bash
docker compose up --build
```

依赖在**容器启动时**安装；服务随后监听 API 端口 **8000**（交互文档
`http://localhost:8000/docs`）。

## 配置

- 宿主机端口：通过环境变量 `PORT` 覆盖（默认 `8000`），容器内固定监听 `8000`。
  例：`PORT=9000 docker compose up --build`
- 依赖清单：`requirements.txt`（容器启动时由 `docker/entrypoint.sh` 安装）。

## API（端口 8000）

- `GET /health` — 健康检查
- `GET /api/v1/whitelist` — 公式白名单
- `POST /api/v1/propagate` — 不确定度传播；相关矩阵非对称 / 非半正定 / 单位量纲不兼容 / 公式不合法时返回 **422**，错误体 `error.loc` 精确定位（如 `correlation_matrix.0.1`、`inputs.0.unit`、`formula`）
- `POST /api/v1/propagate-joint` — 多输出联合传播（见下）
- `POST /api/v1/allocate-uncertainty` — 不确定度逆分配（见下）
- `POST /api/v1/propagate-monte-carlo` — 非线性模型的蒙特卡洛传播校核（见下）
- `POST /api/v1/conformity` — 规格符合性判定（见下）
- `POST /api/v1/repeatability` — 等间隔重复读数的 A 类标准不确定度（见下）
- 交互文档：`http://localhost:8000/docs`

### 多输出联合传播 `POST /api/v1/propagate-joint`

请求体在 `/api/v1/propagate` 的基础上把单个 `formula` 换成 2～8 个**具名输出公式**
`outputs: [{name, formula}, ...]`；输入量、公式白名单、单位与相关矩阵校验完全复用
单输出接口。设 $J_{ki}=\partial f_k/\partial x_i$ 在输入估计值处取值，输入协方差
$\Sigma=DRD$（$D=\mathrm{diag}(u(x_i))$，$R$ 为输入相关系数矩阵），服务端按一阶
雅可比计算输出协方差矩阵：

$$
C=J\Sigma J^{T},\qquad
C_{kl}=\sum_{i}\sum_{j}c_i^{(k)}u(x_i)\,r_{ij}\,u(x_j)c_j^{(l)}
$$

对角元即各输出量的合成方差 $u_c(y_k)^2$。响应返回每个输出的值、单位、合成标准
不确定度，以及：

- `output_covariance_matrix` — 完整输出协方差矩阵，每个单元给出数值与单位
  （**输出 i 单位 × 输出 j 单位**，如 `Ω * W`）；
- `output_correlation_matrix` — 输出相关系数矩阵 $C_{kl}/\big(u_c(y_k)u_c(y_l)\big)$
  （某输出标准不确定度为 0 时其非对角相关系数无定义，置 0）；
- `sensitivity_coefficients` — 每个输出对各输入量的灵敏度系数（即雅可比各行）。

请求示例（$P=U\cdot I$、$R=U/I$ 联合传播，$r(U,I)=0.5$）：

```bash
curl -s http://localhost:8000/api/v1/propagate-joint \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
      {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.05}
    ],
    "outputs": [
      {"name": "P", "formula": "U*I"},
      {"name": "R", "formula": "U/I"}
    ],
    "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]]
  }'
```

响应（节选）：

```json
{
  "outputs": [
    {"name": "P", "value": 20.0, "unit": "W", "standard_uncertainty": 0.6244997998398398},
    {"name": "R", "value": 5.0, "unit": "Ω", "standard_uncertainty": 0.10897247358851683}
  ],
  "output_covariance_matrix": [
    [{"value": 0.39, "unit": "W ** 2"},  {"value": -0.0525, "unit": "Ω * W"}],
    [{"value": -0.0525, "unit": "Ω * W"}, {"value": 0.011875, "unit": "Ω ** 2"}]
  ],
  "output_correlation_matrix": [[1.0, -0.7714542762891773], [-0.7714542762891773, 1.0]]
}
```

输出数量不在 2～8 范围、输出名重复或公式非法时返回 **422**，`error.loc` 精确定位
（如 `outputs`、`outputs.1.name`、`outputs.1.formula`）；相关矩阵非法、单位量纲
不兼容等沿用传播接口的同一套校验与 code/loc。

### 不确定度逆分配 `POST /api/v1/allocate-uncertainty`

请求体与 `/api/v1/propagate` 相同，额外指定待分配输入量与**带单位的目标合成标准不
确定度**。待分配输入量可以省略 `standard_uncertainty`（其值即待求未知量 $u_k$，
其余输入量仍必须给出）。

设 $c_i=\partial f/\partial x_i$，$r_{ij}$ 为相关系数，总方差完整保留相关项后是关于
$u_k$（以该输入量自身单位计）的二次函数：

$$
u_c^2(u_k)=c_k^2u_k^2+2c_ku_k\sum_{j\ne k}c_ju_jr_{kj}
+\sum_{i\ne k}\sum_{j\ne k}c_iu_i c_ju_jr_{ij}=a\,u_k^2+b\,u_k+c
$$

服务端解二次不等式 $a u_k^2+b u_k+c\le U_t^2$（$u_k\ge0$），返回**最大非负根**
（满足目标的最大标准不确定度）及其单位，并把该值代回，按与传播接口完全相同的结构
给出合成值、灵敏度系数与含相关项的方差贡献；`variance_quadratic` 同时回显带单位的
$a/b/c$ 三个系数。

请求示例（$P=U\cdot I$，$u(U)=0.1\,\mathrm{V}$，$r=0.5$，目标
$u_c(P)=0.5\,\mathrm{W}$，反求 $u(I)$）：

```bash
curl -s http://localhost:8000/api/v1/allocate-uncertainty \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
      {"name": "I", "value": 2.0, "unit": "A"}
    ],
    "formula": "U*I",
    "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
    "allocate_input": "I",
    "target_combined_uncertainty": {"value": 0.5, "unit": "W"}
  }'
```

响应（节选）：

```json
{
  "result": {"value": 20.0, "unit": "W", "standard_uncertainty": 0.5},
  "allocate_input": "I",
  "allocated_standard_uncertainty": 0.036904157598234295,
  "allocated_standard_uncertainty_unit": "A",
  "target_combined_uncertainty": {"value": 0.5, "unit": "W", "standard_uncertainty": 0.5},
  "variance_quadratic": {
    "a": 100.0, "a_unit": "W ** 2 / A ** 2",
    "b": 2.0,   "b_unit": "W ** 2 / A",
    "c": 0.04,  "c_unit": "W ** 2",
    "unknown_uncertainty_unit": "A"
  }
}
```

目标单位须与公式输出同量纲（支持单位换算，如目标填 `{"value": 500, "unit": "mW"}`）。
以下情况返回 **422**，错误体带明确 `code` 与 `loc`：

| code | loc | 含义 |
| --- | --- | --- |
| `target_unreachable` | `target_combined_uncertainty` | 无论 $u_k$ 取何非负值，最小合成标准不确定度仍高于目标 |
| `zero_sensitivity` | `allocate_input` | $c_k=0$（该输入量不影响结果）且当前值已不满足目标 |
| `no_finite_bound` | `allocate_input` | $c_k=0$ 而目标已满足，任意大的 $u_k$ 都可行，不存在有限上界 |
| `unknown_allocate_input` | `allocate_input` | 待分配输入量未在 `inputs` 中声明 |
| `missing_uncertainty` | `inputs.<i>.standard_uncertainty` | 非待分配输入量缺少标准不确定度 |
| `incompatible_unit` | `target_combined_uncertainty.unit` | 目标单位与公式输出量纲不一致 |

相关矩阵非法、公式不合法、其余单位错误沿用传播接口的同一套校验与 code/loc。

### 蒙特卡洛传播校核 `POST /api/v1/propagate-monte-carlo`

请求体复用 `/api/v1/propagate` 的 `inputs`、`formula` 与 `correlation_matrix`
校验（同样的白名单、量纲/单位检查与相关矩阵检查），额外要求：

- `samples` — 蒙特卡洛抽样次数，整数 **1000～100000**；
- `seed` — 可选随机数种子（非负整数，上限 $2^{63}-1$）；**给定后结果可复现**，
  缺省（或显式 `null`）时每次抽样不同，响应中回显该值。

输入量按**联合正态**建模：每个输入量的标准不确定度按其声明单位解释（支持
`{value, unit}` 显式单位与前缀、角度、`degC/degF` 等偏移单位）；采样器对相关
系数矩阵 $R$ 做特征分解 $R=V\,\mathrm{diag}(\lambda)\,V^{T}$，由标准正态样本 $Z$ 经

$$
X_{\mathrm{std}} = Z\,\mathrm{diag}(\sqrt{\lambda})\,V^{T}
$$

生成相关标准正态样本。零特征值方向不产生随机扰动，因此**半正定但奇异**的相关
矩阵（如完全相关 $r=\pm1$）也可运行；非半正定矩阵仍返回 422。随后逐样本在 SI
基准单位下计算白名单公式，结果换算回与一阶传播一致的显示单位。

响应返回样本均值、样本标准不确定度（标准差，ddof=1）、2.5%/50%/97.5% 分位数
（线性插值），并在 `first_order` 中给出同一请求的 GUM 一阶传播结果值与
$u_c$，`deviation` 给出蒙特卡洛统计量相对一阶结果的绝对与相对偏差（分母为 0
时相对偏差为 `null`）。非线性模型的均值偏移（如 $E[UI]=\mu_U\mu_I+\mathrm{Cov}(U,I)$、
$E[e^X]=e^{\mu+\sigma^2/2}$）可由此直接校核。

请求示例（$P=U\cdot I$，$r(U,I)=0.5$，50000 样本，固定 seed）：

```bash
curl -s http://localhost:8000/api/v1/propagate-monte-carlo \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
      {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.05}
    ],
    "formula": "U*I",
    "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
    "samples": 50000,
    "seed": 20260920
  }'
```

响应（节选；同一 seed 下逐位可复现）：

```json
{
  "result": {"value": 20.003213420755007, "unit": "W", "standard_uncertainty": 0.6246478040599004},
  "quantiles": {"2.5%": 18.77668552065427, "50%": 20.003869077776333, "97.5%": 21.22478366919415},
  "samples": 50000,
  "seed": 20260920,
  "first_order": {"value": 20.0, "unit": "W", "standard_uncertainty": 0.6244997998398398},
  "deviation": {
    "mean_absolute": 0.003213420755006524,
    "mean_relative": 0.0001606710377503262,
    "standard_uncertainty_absolute": 0.00014800422006056024,
    "standard_uncertainty_relative": 0.00023699642513659352
  }
}
```

若**任一样本**落入公式无效域（`sqrt`/`log` 取非正、`asin`/`acos` 越界、零的负
次幂、数值溢出等导致结果非有限），返回 **422**，`code` 为
`invalid_sample_domain`，`loc` 为 `["formula"]`，并在 `error.details` 中给出
失败样本数量 `failed_samples` 与首个失败样本索引 `first_invalid_sample_index`
（从 0 开始；同时回显 `samples` 与 `seed` 便于复现）。例如 `log(x)` 在
$x=0.01$、$u(x)=1$ 时：

```json
{
  "error": {
    "code": "invalid_sample_domain",
    "message": "5012 个蒙特卡洛样本落入公式无效域（首个为第 2 个样本）；请检查公式定义域与输入不确定度",
    "loc": ["formula"],
    "details": {"failed_samples": 5012, "first_invalid_sample_index": 2, "samples": 10000, "seed": 7}
  }
}
```

`samples` 越界、`seed` 非法（负数或超过 $2^{63}-1$）返回 **422** 并定位到对应
字段；相关矩阵非法、单位/量纲不兼容、公式不合法等沿用传播接口的同一套校验与
code/loc。

### 规格符合性判定 `POST /api/v1/conformity`

请求体复用 `/api/v1/propagate` 的 `inputs`、`formula` 与 `correlation_matrix`
（同样的白名单、量纲/单位检查与相关矩阵检查），额外要求：

- `lower_limit` / `upper_limit` — **带单位的规格界限** `{value, unit}`，至少给出
  一个；单位须与公式输出同量纲（支持换算，如 `mW`→`W`），双侧时下限须小于上限；
- `confidence_threshold` — 置信阈值 $t$，须严格满足 $0.5<t<1$。

服务端先做 GUM 一阶传播得到 $(y,u_c)$，再假定真值服从 $N(y,u_c^2)$，计算落入
**闭规格区间**的概率

$$
p=P(L\le Y\le U)=\Phi\!\left(\tfrac{U-y}{u_c}\right)-\Phi\!\left(\tfrac{L-y}{u_c}\right)
$$

（单侧界限时退化为相应的单尾概率）。$\Phi$ 一律经 `erfc` 计算，双侧概率在区间整体
位于均值上方时改用上尾差 $Q(z_L)-Q(z_U)$，避免 $1-\Phi$ 在远尾饱和为 1——可表示的
**非零尾部概率**（约到 $10^{-308}$ 量级）得以保留。判定规则：

| 条件 | `decision` | 含义 |
| --- | --- | --- |
| $p \ge t$ | `"pass"` | 合格 |
| $p \le 1-t$ | `"fail"` | 不合格 |
| 其余 | `"indeterminate"` | 待定 |

响应返回 `probability`（$p$）、`result`（$y$、单位、$u_c$）、换算到结果显示单位的
`lower_limit`/`upper_limit`，以及 `acceptance_interval`——**固定 $u_c$ 时使
$p(y)\ge t$ 的全部 $y$ 构成的闭接受区间**：

- 双侧界限：$p(y)$ 在中点 $(L+U)/2$ 处取最大值 $2\Phi\!\left(\tfrac{U-L}{2u_c}\right)-1$；
  最大概率仍低于 $t$ 时接受区间为空（`is_empty: true`，两端为 `null`），否则在
  中点两侧分别二分求解 $p(y)=t$ 得到两个边界；
- 单侧界限：直接由分位数给出半无限区间，如仅上限时为
  $\big(-\infty,\;U-u_c\,\Phi^{-1}(t)\big]$（无穷端为 `null`）；
- $u_c=0$ 按确定值处理：$p\in\{0,1\}$，接受区间即闭规格区间本身。

请求示例（$P=U\cdot I$，$r(U,I)=0.5$，双侧规格限 $18.5\sim21.5\,\mathrm{W}$，
$t=0.95$；服务地址 `http://localhost:8000`）：

```bash
curl -s http://localhost:8000/api/v1/conformity \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
      {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.05}
    ],
    "formula": "U*I",
    "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
    "lower_limit": {"value": 18.5, "unit": "W"},
    "upper_limit": {"value": 21.5, "unit": "W"},
    "confidence_threshold": 0.95
  }'
```

响应（$u_c=\sqrt{0.39}\approx0.6245\,\mathrm{W}$，$p\approx0.9837\ge0.95$ 判合格）：

```json
{
  "decision": "pass",
  "probability": 0.983690828122245,
  "confidence_threshold": 0.95,
  "result": {"value": 20.0, "unit": "W", "standard_uncertainty": 0.6244997998398398},
  "lower_limit": {"value": 18.5, "unit": "W"},
  "upper_limit": {"value": 21.5, "unit": "W"},
  "acceptance_interval": {
    "lower": 19.53216838235092,
    "upper": 20.46783161764908,
    "unit": "W",
    "is_empty": false
  },
  "formula": "U*I"
}
```

以下情况返回 **422**，错误体带明确 `code` 与 `loc`：

| code | loc | 含义 |
| --- | --- | --- |
| `missing_spec_limits` | `[]` | 未给出任何规格界限 |
| `incompatible_unit` | `lower_limit.unit` / `upper_limit.unit` | 界限单位与公式输出量纲不一致 |
| `unknown_unit` | `lower_limit.unit` / `upper_limit.unit` | 界限单位无法识别 |
| `invalid_spec_limits` | `lower_limit.value` | 双侧界限换算后下限不小于上限 |
| `greater_than` / `less_than` | `confidence_threshold` | 置信阈值未严格落在 $(0.5,1)$ 内 |

相关矩阵非法、公式不合法、输入量单位错误等沿用传播接口的同一套校验与 code/loc。

### 重复读数 A 类标准不确定度 `POST /api/v1/repeatability`

对同一被测量按等时间间隔取得的 $n$ 个同单位读数 $x_1,\dots,x_n$（请求字段
`readings`，**至少 3 个**；`unit` 为全部读数的共同单位），先求均值
$\bar x=\frac1n\sum_i x_i$ 并去均值 $d_i=x_i-\bar x$，按 **1/n 归一化**
（有偏样本自协方差）计算滞后 $k$ 的自协方差：

$$
\gamma_k=\frac1n\sum_{i=1}^{n-k} d_i\,d_{i+k},\qquad k=0,1,\dots,L
$$

再用 Bartlett（三角）权重 $w_k=1-\frac{k}{L+1}$ 估计均值的 A 类方差：

$$
u_A^2=\frac{1}{n}\left[\gamma_0+2\sum_{k=1}^{L}\left(1-\frac{k}{L+1}\right)\gamma_k\right]
$$

其中 `max_lag` 即最大滞后 $L$，须满足整数 $0\le L<n$。等间隔读数若存在自相关，
朴素 $s/\sqrt n$ 不再适用：**正自相关**使相邻读数携带重复信息，$u_A$ 应增大、
有效样本量减小；负自相关则相反。响应同时返回有效样本量

$$
n_{\mathrm{eff}}=\frac{n\,\gamma_0}{\gamma_0+2\sum_{k=1}^{L}(1-\frac{k}{L+1})\gamma_k}
$$

（即加窗长程方差括号项对应的等效独立读数数量）；**分母为零时 $n_{\mathrm{eff}}$
为 `null`**（如全部读数相同）。各 $\gamma_k$ 与 $u_A^2$ 的单位为读数单位的平方，
均值与 $u_A$ 保持输入单位；`result.standard_uncertainty`（即 $u_A$）可直接填入
`/api/v1/propagate` 输入量的 `standard_uncertainty`（同一 `unit`）。

请求示例（8 个等间隔电压读数，$L=2$）：

```bash
curl -s http://localhost:8000/api/v1/repeatability \
  -H 'Content-Type: application/json' \
  -d '{
    "readings": [10.00, 10.01, 10.03, 10.02, 10.00, 9.98, 9.99, 10.01],
    "unit": "V",
    "max_lag": 2
  }'
```

响应（$\bar x=10.005\,\mathrm{V}$，$\gamma_0=0.000225$、$\gamma_1=0.000103125$、
$\gamma_2=-0.000075\,\mathrm{V^2}$，$u_A=0.00625\,\mathrm{V}$；正自相关使
$n_{\mathrm{eff}}=5.76<n=8$。JSON 数值为双精度浮点，末位有舍入噪声）：

```json
{
  "result": {"value": 10.004999999999999, "unit": "V", "standard_uncertainty": 0.006249999999999875},
  "number_of_readings": 8,
  "max_lag": 2,
  "autocovariances": [
    {"lag": 0, "value": 0.00022499999999999043, "unit": "V ** 2"},
    {"lag": 1, "value": 0.00010312499999999561, "unit": "V ** 2"},
    {"lag": 2, "value": -0.0000749999999999957, "unit": "V ** 2"}
  ],
  "variance": 0.00003906249999999843,
  "variance_unit": "V ** 2",
  "effective_sample_size": 5.7599999999999865
}
```

取 `max_lag: 0` 时 $u_A^2=\gamma_0/n$（1/n 归一化的方差再除以 $n$，等价于
$\frac{n-1}{n}\cdot\frac{s^2}{n}$），且恒有 $n_{\mathrm{eff}}=n$。以下情况返回
**422**，错误体带明确 `code` 与 `loc`：

| code | loc | 含义 |
| --- | --- | --- |
| `too_short` / `finite_number` 等 Pydantic 错误 | `readings` / `readings.<i>` | 读数少于 3 个，或含 `NaN`/`Infinity` 等非有限值 |
| `empty_unit` / `unknown_unit` | `unit` | 单位为空或无法识别 |
| `greater_than_equal` / `int_from_float` 等 Pydantic 错误 | `max_lag` | $L$ 非整数或为负数（须 $L\ge0$） |
| `lag_too_large` | `max_lag` | $L\ge n$（最大滞后须严格小于读数数量） |
| `negative_variance_estimate` | `max_lag` | 加窗长程方差显著为负（自协方差估计数值不自洽，建议减小 $L$） |


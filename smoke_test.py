"""端到端快速验证（不经过 HTTP，直接调传播函数与 API 客户端）。"""
import json
import math
import sys

sys.path.insert(0, "/workspace")

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def show(title, resp):
    print("=" * 70)
    print(title)
    print(json.dumps(resp, ensure_ascii=False, indent=2))


# 1) 欧姆定律 P = U*I，单位校验 + 不相关
r = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [
            {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
            {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.02},
        ],
        "formula": "U*I",
    },
)
show("1) P=U*I (expect 20 W, uc=sqrt(0.2^2+0.2^2)=0.2828 W)", r.json())
assert r.status_code == 200, r.text
b = r.json()
assert abs(b["result"]["value"] - 20.0) < 1e-12
assert b["result"]["unit"] == "W"
assert abs(b["result"]["standard_uncertainty"] - (0.08 ** 0.5)) < 1e-12

# 2) 完全正相关，相关项使不确定度为 0.4
r = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [
            {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
            {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.02},
        ],
        "formula": "U*I",
        "correlation_matrix": [[1, 1], [1, 1]],
    },
)
show("2) P=U*I, r=1 (expect uc=0.4 W, share=0.5 each)", r.json())
assert r.status_code == 200
b = r.json()
assert abs(b["result"]["standard_uncertainty"] - 0.4) < 1e-12
assert abs(b["variance_contributions"][0]["share"] - 0.5) < 1e-9

# 3) 负相关抵消
r = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [
            {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
            {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.02},
        ],
        "formula": "U*I",
        "correlation_matrix": [[1, -1], [-1, 1]],
    },
)
show("3) P=U*I, r=-1 (expect uc=0)", r.json())
assert r.status_code == 200
assert abs(r.json()["result"]["standard_uncertainty"]) < 1e-9

# 4) 量纲不一致：V + A
r = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [
            {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
            {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.02},
        ],
        "formula": "U + I",
    },
)
show("4) V + A -> 422 dimension_mismatch", r.json())
assert r.status_code == 422
assert r.json()["error"]["code"] == "dimension_mismatch"
assert r.json()["error"]["loc"] == ["formula"]

# 5) 非对称矩阵，定位到 [1,0]
r = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [
            {"name": "x", "value": 1, "unit": "m", "standard_uncertainty": 0.1},
            {"name": "y", "value": 2, "unit": "m", "standard_uncertainty": 0.1},
        ],
        "formula": "x+y",
        "correlation_matrix": [[1, 0.5], [0.3, 1]],
    },
)
show("5) asymmetric -> 422 loc [correlation_matrix,1,1?]", r.json())
assert r.status_code == 422
assert r.json()["error"]["code"] == "asymmetric_matrix"
assert r.json()["error"]["loc"][0] == "correlation_matrix"

# 6) 非半正定（三个相关系数互相矛盾）
r = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [
            {"name": "x", "value": 1, "unit": "m", "standard_uncertainty": 0.1},
            {"name": "y", "value": 2, "unit": "m", "standard_uncertainty": 0.1},
            {"name": "z", "value": 3, "unit": "m", "standard_uncertainty": 0.1},
        ],
        "formula": "x+y+z",
        "correlation_matrix": [
            [1, 0.9, 0.9],
            [0.9, 1, -0.9],
            [0.9, -0.9, 1],
        ],
    },
)
show("6) non-PSD -> 422", r.json())
assert r.status_code == 422
assert r.json()["error"]["code"] == "not_positive_semidefinite"

# 7) 白名单外函数 __import__
r = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [{"name": "x", "value": 1, "unit": "m", "standard_uncertainty": 0.1}],
        "formula": "__import__('os').system('id')",
    },
)
show("7) injection -> 422", r.json())
assert r.status_code == 422

# 8) 三角函数 + 角度单位，ln/log10
r = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [
            {"name": "theta", "value": 30, "unit": "deg", "standard_uncertainty": 1},
            {"name": "L", "value": 2.0, "unit": "m", "standard_uncertainty": 0.01},
        ],
        "formula": "L * cos(theta)",
    },
)
show("8) L*cos(30 deg) (expect ~1.732 m)", r.json())
assert r.status_code == 200, r.text
b = r.json()
assert abs(b["result"]["value"] - 2.0 * (3 ** 0.5) / 2) < 1e-9
assert b["result"]["unit"] == "m"

# 9) 不确定度显式单位（mV 与 V 量纲一致）
r = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [
            {
                "name": "U",
                "value": 10.0,
                "unit": "V",
                "standard_uncertainty": {"value": 100, "unit": "mV"},
            },
        ],
        "formula": "U*2",
    },
)
show("9) u=100 mV for 10 V (expect uc=0.2 V)", r.json())
assert r.status_code == 200, r.text
assert abs(r.json()["result"]["standard_uncertainty"] - 0.2) < 1e-12

# 10) 不确定度单位量纲不一致 -> 422 loc 到字段
r = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [
            {
                "name": "U",
                "value": 10.0,
                "unit": "V",
                "standard_uncertainty": {"value": 0.1, "unit": "A"},
            },
        ],
        "formula": "U*2",
    },
)
show("10) u unit A vs V -> 422", r.json())
assert r.status_code == 422
assert r.json()["error"]["loc"][-1] == "unit"

# 11) log 量纲错误：log(m)
r = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [{"name": "x", "value": 1, "unit": "m", "standard_uncertainty": 0.1}],
        "formula": "log(x)",
    },
)
show("11) log(m) -> 422", r.json())
assert r.status_code == 422

# 12) 未声明符号
r = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [{"name": "x", "value": 1, "unit": "m", "standard_uncertainty": 0.1}],
        "formula": "x + y",
    },
)
show("12) unknown symbol y -> 422", r.json())
assert r.status_code == 422 and r.json()["error"]["code"] == "unknown_symbol"

# 13) 幂次为变量
r = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [
            {"name": "x", "value": 2, "unit": "m", "standard_uncertainty": 0.1},
            {"name": "n", "value": 2, "unit": "dimensionless", "standard_uncertainty": 0.0},
        ],
        "formula": "x**n",
    },
)
show("13) x**n -> 422 non_constant_exponent", r.json())
assert r.status_code == 422 and r.json()["error"]["code"] == "non_constant_exponent"

# 14) 未知单位
r = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [{"name": "x", "value": 1, "unit": "frobnicate", "standard_uncertainty": 0.1}],
        "formula": "x",
    },
)
show("14) unknown unit -> 422 loc inputs/0/unit", r.json())
assert r.status_code == 422 and r.json()["error"]["loc"] == ["inputs", 0, "unit"]

# 15) 相关系数越界
r = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [
            {"name": "x", "value": 1, "unit": "m", "standard_uncertainty": 0.1},
            {"name": "y", "value": 2, "unit": "m", "standard_uncertainty": 0.1},
        ],
        "formula": "x+y",
        "correlation_matrix": [[1, 1.2], [1.2, 1]],
    },
)
show("15) r=1.2 -> 422 loc element", r.json())
assert r.status_code == 422 and r.json()["error"]["code"] == "correlation_out_of_range"
assert r.json()["error"]["loc"] == ["correlation_matrix", 0, 1]

# 16) 对角线 != 1
r = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [
            {"name": "x", "value": 1, "unit": "m", "standard_uncertainty": 0.1},
            {"name": "y", "value": 2, "unit": "m", "standard_uncertainty": 0.1},
        ],
        "formula": "x+y",
        "correlation_matrix": [[0.9, 0], [0, 1]],
    },
)
show("16) diag=0.9 -> 422", r.json())
assert r.status_code == 422 and r.json()["error"]["code"] == "diagonal_not_one"

# 17) 矩阵形状不符
r = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [
            {"name": "x", "value": 1, "unit": "m", "standard_uncertainty": 0.1},
            {"name": "y", "value": 2, "unit": "m", "standard_uncertainty": 0.1},
        ],
        "formula": "x+y",
        "correlation_matrix": [[1]],
    },
)
show("17) bad shape -> 422", r.json())
assert r.status_code == 422 and r.json()["error"]["code"] == "bad_matrix_shape"

# 18) sqrt(m^2) 量纲
r = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [
            {"name": "a", "value": 3, "unit": "m", "standard_uncertainty": 0.1},
            {"name": "b", "value": 4, "unit": "m", "standard_uncertainty": 0.1},
        ],
        "formula": "sqrt(a^2 + b^2)",
    },
)
show("18) sqrt(a^2+b^2) (expect 5 m)", r.json())
assert r.status_code == 200, r.text
b = r.json()
assert abs(b["result"]["value"] - 5.0) < 1e-9
assert b["result"]["unit"] == "m"

# 19) 重复名称
r = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [
            {"name": "x", "value": 1, "unit": "m", "standard_uncertainty": 0.1},
            {"name": "x", "value": 2, "unit": "m", "standard_uncertainty": 0.1},
        ],
        "formula": "x",
    },
)
show("19) duplicate name -> 422", r.json())
assert r.status_code == 422 and r.json()["error"]["code"] == "duplicate_name"

# 20) 常数公式
r = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [{"name": "x", "value": 1, "unit": "m", "standard_uncertainty": 0.1}],
        "formula": "pi*E",
    },
)
show("20) pi*E, no x dependence (uc=0)", r.json())
assert r.status_code == 200, r.text
assert abs(r.json()["result"]["standard_uncertainty"]) < 1e-12

# --- POST /api/v1/allocate-uncertainty ------------------------------------

# 21) P=U*I，r=0.5，把 u(I) 当未知量，目标 uc=0.5 W
#   Var = 100 uI^2 + 2 uI + 0.04 = 0.25
#   uI = (-2 + sqrt(88)) / 200
r = client.post(
    "/api/v1/allocate-uncertainty",
    json={
        "inputs": [
            {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
            {"name": "I", "value": 2.0, "unit": "A"},
        ],
        "formula": "U*I",
        "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
        "allocate_input": "I",
        "target_combined_uncertainty": {"value": 0.5, "unit": "W"},
    },
)
show("21) allocate u(I) for target uc=0.5 W", r.json())
assert r.status_code == 200, r.text
b = r.json()
assert abs(b["allocated_standard_uncertainty"] - ((88 ** 0.5) - 2.0) / 200.0) < 1e-12
assert b["allocated_standard_uncertainty_unit"] == "A"
assert b["result"]["unit"] == "W"
assert abs(b["result"]["standard_uncertainty"] - 0.5) < 1e-9
q = b["variance_quadratic"]
assert (q["a"], q["b"], q["c"]) == (100.0, 2.0, 0.04000000000000001)
assert q["unknown_uncertainty_unit"] == "A"

# 21b) 把求得的 u(I) 代回 /propagate，合成值应恰为目标
r = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [
            {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
            {
                "name": "I",
                "value": 2.0,
                "unit": "A",
                "standard_uncertainty": b["allocated_standard_uncertainty"],
            },
        ],
        "formula": "U*I",
        "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
    },
)
assert r.status_code == 200
assert abs(r.json()["result"]["standard_uncertainty"] - 0.5) < 1e-9

# 22) 目标不可达：r=-0.5 时最小总方差 0.03（x=0.01），目标 uc=0.1 不可达
r = client.post(
    "/api/v1/allocate-uncertainty",
    json={
        "inputs": [
            {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
            {"name": "I", "value": 2.0, "unit": "A"},
        ],
        "formula": "U*I",
        "correlation_matrix": [[1.0, -0.5], [-0.5, 1.0]],
        "allocate_input": "I",
        "target_combined_uncertainty": {"value": 0.1, "unit": "W"},
    },
)
show("22) target unreachable -> 422 target_unreachable", r.json())
assert r.status_code == 422
assert r.json()["error"]["code"] == "target_unreachable"
assert r.json()["error"]["loc"] == ["target_combined_uncertainty"]

# 23) 完全负相关、目标 uc=0：可行集仅一个点 uI=0.02
r = client.post(
    "/api/v1/allocate-uncertainty",
    json={
        "inputs": [
            {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
            {"name": "I", "value": 2.0, "unit": "A"},
        ],
        "formula": "U*I",
        "correlation_matrix": [[1, -1], [-1, 1]],
        "allocate_input": "I",
        "target_combined_uncertainty": {"value": 0.0, "unit": "W"},
    },
)
show("23) r=-1, target uc=0 -> uI=0.02 A", r.json())
assert r.status_code == 200, r.text
assert abs(r.json()["allocated_standard_uncertainty"] - 0.02) < 1e-9

# 24) 灵敏度为零且目标已满足 -> 无有限上界
r = client.post(
    "/api/v1/allocate-uncertainty",
    json={
        "inputs": [
            {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
            {"name": "I", "value": 2.0, "unit": "A"},
        ],
        "formula": "U*2",
        "allocate_input": "I",
        "target_combined_uncertainty": {"value": 1.0, "unit": "V"},
    },
)
show("24) zero sensitivity, target met -> 422 no_finite_bound", r.json())
assert r.status_code == 422
assert r.json()["error"]["code"] == "no_finite_bound"
assert r.json()["error"]["loc"] == ["allocate_input"]

# 25) 灵敏度为零且目标不满足 -> zero_sensitivity
r = client.post(
    "/api/v1/allocate-uncertainty",
    json={
        "inputs": [
            {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
            {"name": "I", "value": 2.0, "unit": "A"},
        ],
        "formula": "U*2",
        "allocate_input": "I",
        "target_combined_uncertainty": {"value": 0.01, "unit": "V"},
    },
)
show("25) zero sensitivity, target missed -> 422 zero_sensitivity", r.json())
assert r.status_code == 422
assert r.json()["error"]["code"] == "zero_sensitivity"
assert r.json()["error"]["loc"] == ["allocate_input"]

# 26) 待分配输入量未声明
r = client.post(
    "/api/v1/allocate-uncertainty",
    json={
        "inputs": [
            {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
        ],
        "formula": "U*2",
        "allocate_input": "Q",
        "target_combined_uncertainty": {"value": 1.0, "unit": "V"},
    },
)
show("26) unknown allocate_input -> 422", r.json())
assert r.status_code == 422
assert r.json()["error"]["code"] == "unknown_allocate_input"
assert r.json()["error"]["loc"] == ["allocate_input"]

# 27) 非待分配输入量也缺不确定度
r = client.post(
    "/api/v1/allocate-uncertainty",
    json={
        "inputs": [
            {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
            {"name": "I", "value": 2.0, "unit": "A"},
        ],
        "formula": "U*I",
        "allocate_input": "U",
        "target_combined_uncertainty": {"value": 1.0, "unit": "W"},
    },
)
show("27) other input missing uncertainty -> 422", r.json())
assert r.status_code == 422
assert r.json()["error"]["code"] == "missing_uncertainty"
assert r.json()["error"]["loc"] == ["inputs", 1, "standard_uncertainty"]

# 28) 目标单位量纲不一致
r = client.post(
    "/api/v1/allocate-uncertainty",
    json={
        "inputs": [
            {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
        ],
        "formula": "U*2",
        "allocate_input": "U",
        "target_combined_uncertainty": {"value": 1.0, "unit": "A"},
    },
)
show("28) target unit A vs output V -> 422", r.json())
assert r.status_code == 422
assert r.json()["error"]["code"] == "incompatible_unit"
assert r.json()["error"]["loc"] == ["target_combined_uncertainty", "unit"]

# 29) 目标单位换算：500 mW = 0.5 W；uU=0 时 uI = 0.05 A
r = client.post(
    "/api/v1/allocate-uncertainty",
    json={
        "inputs": [
            {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.0},
            {"name": "I", "value": 2.0, "unit": "A"},
        ],
        "formula": "U*I",
        "allocate_input": "I",
        "target_combined_uncertainty": {"value": 500, "unit": "mW"},
    },
)
show("29) target 500 mW -> uI=0.05 A", r.json())
assert r.status_code == 200, r.text
assert abs(r.json()["allocated_standard_uncertainty"] - 0.05) < 1e-12
assert abs(r.json()["result"]["standard_uncertainty"] - 0.5) < 1e-9

# 30) 相关矩阵非法仍走同一套校验
r = client.post(
    "/api/v1/allocate-uncertainty",
    json={
        "inputs": [
            {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
            {"name": "I", "value": 2.0, "unit": "A"},
        ],
        "formula": "U*I",
        "correlation_matrix": [[1, 2], [2, 1]],
        "allocate_input": "I",
        "target_combined_uncertainty": {"value": 1.0, "unit": "W"},
    },
)
show("30) bad correlation -> 422", r.json())
assert r.status_code == 422
assert r.json()["error"]["code"] == "correlation_out_of_range"

# 31) 不相关、取 0 已超目标 -> target_unreachable
r = client.post(
    "/api/v1/allocate-uncertainty",
    json={
        "inputs": [
            {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
            {"name": "I", "value": 2.0, "unit": "A"},
        ],
        "formula": "U*I",
        "allocate_input": "I",
        "target_combined_uncertainty": {"value": 0.1, "unit": "W"},
    },
)
show("31) x=0 already above target -> 422", r.json())
assert r.status_code == 422 and r.json()["error"]["code"] == "target_unreachable"

# --- POST /api/v1/propagate-joint -------------------------------------------

# 32) 联合传播 P=U*I 与 R=U/I，r=0.5
#   P: cu=[0.2, 0.5]  -> Var=0.04+0.25+2*0.5*0.2*0.5=0.39
#   R: cu=[0.05, -0.125] -> Var=0.0025+0.015625-2*0.5*0.05*0.125=0.011875
#   Cov(P,R)=0.01-0.0125+0.0125-0.0625=-0.0525
r = client.post(
    "/api/v1/propagate-joint",
    json={
        "inputs": [
            {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
            {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.05},
        ],
        "outputs": [
            {"name": "P", "formula": "U*I"},
            {"name": "R", "formula": "U/I"},
        ],
        "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
    },
)
show("32) joint P=U*I, R=U/I (Var_P=0.39, Var_R=0.011875, Cov=-0.0525)", r.json())
assert r.status_code == 200, r.text
b = r.json()
assert b["number_of_inputs"] == 2 and b["number_of_outputs"] == 2
assert b["outputs"][0] == {
    "name": "P",
    "value": 20.0,
    "unit": "W",
    "standard_uncertainty": 0.39 ** 0.5,
}
assert b["outputs"][1]["name"] == "R"
assert abs(b["outputs"][1]["value"] - 5.0) < 1e-12
assert abs(b["outputs"][1]["standard_uncertainty"] - 0.011875 ** 0.5) < 1e-12
cov = b["output_covariance_matrix"]
assert abs(cov[0][0]["value"] - 0.39) < 1e-12 and cov[0][0]["unit"] == "W ** 2"
assert abs(cov[0][1]["value"] - (-0.0525)) < 1e-12
assert abs(cov[1][0]["value"] - (-0.0525)) < 1e-12
assert abs(cov[1][1]["value"] - 0.011875) < 1e-12
oc = b["output_correlation_matrix"]
assert oc[0][0] == 1.0 and oc[1][1] == 1.0
expected_r = -0.0525 / (0.39 ** 0.5 * 0.011875 ** 0.5)
assert abs(oc[0][1] - expected_r) < 1e-12 and abs(oc[1][0] - expected_r) < 1e-12
# 每个输出的灵敏度系数（雅可比行）
sens = {s["output"]: s["coefficients"] for s in b["sensitivity_coefficients"]}
assert [c["value"] for c in sens["P"]] == [2.0, 10.0]
assert [c["value"] for c in sens["R"]] == [0.5, -2.5]

# 32b) 与单输出接口一致：联合对角元 = /propagate 的合成不确定度
r1 = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [
            {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
            {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.05},
        ],
        "formula": "U*I",
        "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
    },
)
assert r1.status_code == 200
assert abs(r1.json()["result"]["standard_uncertainty"] - b["outputs"][0]["standard_uncertainty"]) < 1e-15

# 33) 输出名重复 -> 422 loc outputs/1/name
r = client.post(
    "/api/v1/propagate-joint",
    json={
        "inputs": [{"name": "x", "value": 1, "unit": "m", "standard_uncertainty": 0.1}],
        "outputs": [{"name": "a", "formula": "x"}, {"name": "a", "formula": "x*2"}],
    },
)
show("33) duplicate output name -> 422", r.json())
assert r.status_code == 422
assert r.json()["error"]["code"] == "duplicate_name"
assert r.json()["error"]["loc"] == ["outputs", 1, "name"]

# 34) 第二个输出公式非法（未声明符号）-> 422 loc outputs/1/formula
r = client.post(
    "/api/v1/propagate-joint",
    json={
        "inputs": [{"name": "x", "value": 1, "unit": "m", "standard_uncertainty": 0.1}],
        "outputs": [{"name": "a", "formula": "x"}, {"name": "b", "formula": "x+y"}],
    },
)
show("34) unknown symbol in output 1 -> 422", r.json())
assert r.status_code == 422
assert r.json()["error"]["code"] == "unknown_symbol"
assert r.json()["error"]["loc"] == ["outputs", 1, "formula"]

# 34b) 输出公式量纲不一致 -> 422 loc 同样精确定位
r = client.post(
    "/api/v1/propagate-joint",
    json={
        "inputs": [
            {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
            {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.05},
        ],
        "outputs": [{"name": "P", "formula": "U*I"}, {"name": "bad", "formula": "U+I"}],
    },
)
assert r.status_code == 422
assert r.json()["error"]["code"] == "dimension_mismatch"
assert r.json()["error"]["loc"] == ["outputs", 1, "formula"]

# 35) 输出数量越界：1 个或 9 个 -> 422 loc outputs
r = client.post(
    "/api/v1/propagate-joint",
    json={
        "inputs": [{"name": "x", "value": 1, "unit": "m", "standard_uncertainty": 0.1}],
        "outputs": [{"name": "a", "formula": "x"}],
    },
)
show("35) single output -> 422", r.json())
assert r.status_code == 422 and r.json()["error"]["loc"] == ["outputs"]
r = client.post(
    "/api/v1/propagate-joint",
    json={
        "inputs": [{"name": "x", "value": 1, "unit": "m", "standard_uncertainty": 0.1}],
        "outputs": [{"name": f"y{k}", "formula": f"x*{k+1}"} for k in range(9)],
    },
)
show("35b) nine outputs -> 422", r.json())
assert r.status_code == 422 and r.json()["error"]["loc"] == ["outputs"]

# 36) 8 个输出（上限）可正常计算
r = client.post(
    "/api/v1/propagate-joint",
    json={
        "inputs": [{"name": "x", "value": 2, "unit": "m", "standard_uncertainty": 0.1}],
        "outputs": [{"name": f"y{k}", "formula": f"x*{k+1}"} for k in range(8)],
    },
)
show("36) eight outputs -> 200", r.json())
assert r.status_code == 200, r.text
b = r.json()
assert b["number_of_outputs"] == 8
# y_k = (k+1) x，完全相关：输出相关系数矩阵应全为 1
assert all(abs(v - 1.0) < 1e-12 for row in b["output_correlation_matrix"] for v in row)

# 37) 相关矩阵非法仍走同一套校验
r = client.post(
    "/api/v1/propagate-joint",
    json={
        "inputs": [
            {"name": "x", "value": 1, "unit": "m", "standard_uncertainty": 0.1},
            {"name": "y", "value": 2, "unit": "m", "standard_uncertainty": 0.1},
        ],
        "outputs": [{"name": "s", "formula": "x+y"}, {"name": "d", "formula": "x-y"}],
        "correlation_matrix": [[1, 1.2], [1.2, 1]],
    },
)
show("37) bad correlation -> 422", r.json())
assert r.status_code == 422
assert r.json()["error"]["code"] == "correlation_out_of_range"
assert r.json()["error"]["loc"] == ["correlation_matrix", 0, 1]

# 38) 零不确定度输出：非对角相关系数无定义置 0，对角为 1
r = client.post(
    "/api/v1/propagate-joint",
    json={
        "inputs": [{"name": "x", "value": 1, "unit": "m", "standard_uncertainty": 0.1}],
        "outputs": [{"name": "c", "formula": "pi*E"}, {"name": "y", "formula": "x*2"}],
    },
)
show("38) zero-variance output -> off-diagonal corr 0", r.json())
assert r.status_code == 200, r.text
b = r.json()
assert b["outputs"][0]["standard_uncertainty"] == 0.0
assert b["output_correlation_matrix"] == [[1.0, 0.0], [0.0, 1.0]]

# 39) 缺省相关矩阵 = 单位阵；完全负相关输入使两输出协方差为负
r = client.post(
    "/api/v1/propagate-joint",
    json={
        "inputs": [
            {"name": "x", "value": 1, "unit": "m", "standard_uncertainty": 0.1},
            {"name": "y", "value": 2, "unit": "m", "standard_uncertainty": 0.2},
        ],
        "outputs": [{"name": "s", "formula": "x+y"}, {"name": "d", "formula": "x-y"}],
    },
)
show("39) default identity corr; s=x+y, d=x-y", r.json())
assert r.status_code == 200, r.text
b = r.json()
# Var(s)=0.01+0.04=0.05, Var(d)=0.05, Cov(s,d)=0.01-0.04=-0.03
cov = b["output_covariance_matrix"]
assert abs(cov[0][0]["value"] - 0.05) < 1e-12
assert abs(cov[0][1]["value"] - (-0.03)) < 1e-12
assert abs(b["output_correlation_matrix"][0][1] - (-0.6)) < 1e-12
assert cov[0][1]["unit"] == "m ** 2"

# --- POST /api/v1/propagate-monte-carlo -------------------------------------

_MC_BODY = {
    "inputs": [
        {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
        {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.05},
    ],
    "formula": "U*I",
    "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
    "samples": 50000,
    "seed": 20260920,
}

# 40) 基本结构 + 固定 seed 可复现 + 一阶 uc=sqrt(0.39)≈0.6245 W
r = client.post("/api/v1/propagate-monte-carlo", json=_MC_BODY)
show("40) MC P=U*I, seeded", r.json())
assert r.status_code == 200, r.text
b = r.json()
assert b["samples"] == 50000 and b["seed"] == 20260920 and b["number_of_inputs"] == 2
assert b["result"]["unit"] == "W"
# E[UI] = 20 + cov(U,I) = 20 + 0.5*0.1*0.05 = 20.0025（非线性项带来均值偏移）
assert abs(b["result"]["value"] - 20.0025) < 0.005
assert abs(b["result"]["standard_uncertainty"] - 0.39 ** 0.5) < 0.02
q = b["quantiles"]
assert q["2.5%"] < q["50%"] < q["97.5%"]
assert set(q) == {"2.5%", "50%", "97.5%"}
# 一阶摘要与偏差
fo = b["first_order"]
assert fo == {"value": 20.0, "unit": "W", "standard_uncertainty": 0.39 ** 0.5}
assert abs(b["deviation"]["mean_absolute"] - (b["result"]["value"] - 20.0)) < 1e-12
assert b["deviation"]["mean_relative"] == b["deviation"]["mean_absolute"] / 20.0
assert abs(
    b["deviation"]["standard_uncertainty_absolute"]
    - (b["result"]["standard_uncertainty"] - 0.39 ** 0.5)
) < 1e-12
# 相同 seed 结果逐位一致
r2 = client.post("/api/v1/propagate-monte-carlo", json=_MC_BODY)
assert r2.json() == b

# 40b) 不指定 seed：回显 null，两次抽样（几乎必然）不同
b1 = client.post(
    "/api/v1/propagate-monte-carlo",
    json={
        "inputs": [{"name": "x", "value": 1.0, "unit": "m", "standard_uncertainty": 0.3}],
        "formula": "x^3",
        "samples": 5000,
    },
).json()
b2 = client.post(
    "/api/v1/propagate-monte-carlo",
    json={
        "inputs": [{"name": "x", "value": 1.0, "unit": "m", "standard_uncertainty": 0.3}],
        "formula": "x^3",
        "samples": 5000,
        "seed": None,
    },
).json()
assert b1["seed"] is None and b2["seed"] is None
assert b1["result"]["value"] != b2["result"]["value"]

# 41) 奇异（完全相关）相关矩阵可运行；x+y, ux=0.2, uy=0.3, r=1 -> uc=0.5
r = client.post(
    "/api/v1/propagate-monte-carlo",
    json={
        "inputs": [
            {"name": "x", "value": 1.0, "unit": "m", "standard_uncertainty": 0.2},
            {"name": "y", "value": 2.0, "unit": "m", "standard_uncertainty": 0.3},
        ],
        "formula": "x+y",
        "correlation_matrix": [[1.0, 1.0], [1.0, 1.0]],
        "samples": 50000,
        "seed": 2,
    },
)
show("41) singular corr r=1", r.json())
assert r.status_code == 200
assert abs(r.json()["result"]["standard_uncertainty"] - 0.5) < 0.01
# 完全负相关且不确定度相等 -> uc=0
r = client.post(
    "/api/v1/propagate-monte-carlo",
    json={
        "inputs": [
            {"name": "x", "value": 1.0, "unit": "m", "standard_uncertainty": 0.2},
            {"name": "y", "value": 2.0, "unit": "m", "standard_uncertainty": 0.2},
        ],
        "formula": "x+y",
        "correlation_matrix": [[1.0, -1.0], [-1.0, 1.0]],
        "samples": 50000,
        "seed": 2,
    },
)
assert r.status_code == 200
assert abs(r.json()["result"]["standard_uncertainty"]) < 1e-12

# 42) 无效域：log(x) 有样本 <= 0 -> 422 invalid_sample_domain + 数量/首个索引
r = client.post(
    "/api/v1/propagate-monte-carlo",
    json={
        "inputs": [
            {"name": "x", "value": 0.01, "unit": "dimensionless", "standard_uncertainty": 1.0}
        ],
        "formula": "log(x)",
        "samples": 10000,
        "seed": 7,
    },
)
show("42) log(x) invalid domain -> 422", r.json())
assert r.status_code == 422
err = r.json()["error"]
assert err["code"] == "invalid_sample_domain"
assert err["loc"] == ["formula"]
d = err["details"]
assert d["failed_samples"] > 0
assert 0 <= d["first_invalid_sample_index"] < 10000
assert d["samples"] == 10000 and d["seed"] == 7

# 42b) asin 越界同样报 invalid_sample_domain
r = client.post(
    "/api/v1/propagate-monte-carlo",
    json={
        "inputs": [
            {"name": "x", "value": 0.999, "unit": "dimensionless", "standard_uncertainty": 0.01}
        ],
        "formula": "asin(x)",
        "samples": 5000,
        "seed": 3,
    },
)
assert r.status_code == 422
assert r.json()["error"]["code"] == "invalid_sample_domain"

# 43) samples 越界 / seed 非法 -> 422（Pydantic 定位到字段）
for bad_samples in (999, 100001):
    r = client.post(
        "/api/v1/propagate-monte-carlo",
        json={
            "inputs": [
                {"name": "x", "value": 1.0, "unit": "m", "standard_uncertainty": 0.1}
            ],
            "formula": "x",
            "samples": bad_samples,
        },
    )
    assert r.status_code == 422 and r.json()["error"]["loc"] == ["samples"]
r = client.post(
    "/api/v1/propagate-monte-carlo",
    json={
        "inputs": [
            {"name": "x", "value": 1.0, "unit": "m", "standard_uncertainty": 0.1}
        ],
        "formula": "x",
        "samples": 2000,
        "seed": -1,
    },
)
assert r.status_code == 422 and r.json()["error"]["loc"] == ["seed"]
# 缺 samples 字段
r = client.post(
    "/api/v1/propagate-monte-carlo",
    json={
        "inputs": [
            {"name": "x", "value": 1.0, "unit": "m", "standard_uncertainty": 0.1}
        ],
        "formula": "x",
    },
)
assert r.status_code == 422 and r.json()["error"]["loc"] == ["samples"]

# 44) 偏移单位 degC/degF 与角度 deg：不确定度按声明单位解释
r = client.post(
    "/api/v1/propagate-monte-carlo",
    json={
        "inputs": [
            {"name": "T", "value": 25.0, "unit": "degC", "standard_uncertainty": 1.0}
        ],
        "formula": "T",
        "samples": 5000,
        "seed": 11,
    },
)
show("44) degC affine", r.json())
assert r.status_code == 200, r.text
b = r.json()
assert abs(b["result"]["value"] - 25.0) < 0.05
assert abs(b["result"]["standard_uncertainty"] - 1.0) < 0.05
assert b["result"]["unit"] == "°C"
r = client.post(
    "/api/v1/propagate-monte-carlo",
    json={
        "inputs": [
            {"name": "theta", "value": 30.0, "unit": "deg", "standard_uncertainty": 0.5},
            {"name": "L", "value": 2.0, "unit": "m", "standard_uncertainty": 0.0},
        ],
        "formula": "L*cos(theta)",
        "samples": 10000,
        "seed": 5,
    },
)
assert r.status_code == 200
assert abs(r.json()["result"]["value"] - 3 ** 0.5) < 0.02

# 45) 不确定度显式单位（100 mV = 0.1 V）；一阶/蒙特卡洛一致
r = client.post(
    "/api/v1/propagate-monte-carlo",
    json={
        "inputs": [
            {
                "name": "U",
                "value": 10.0,
                "unit": "V",
                "standard_uncertainty": {"value": 100, "unit": "mV"},
            }
        ],
        "formula": "U*2",
        "samples": 5000,
        "seed": 1,
    },
)
assert r.status_code == 200
b = r.json()
assert abs(b["result"]["standard_uncertainty"] - 0.2) < 0.01
assert abs(b["first_order"]["standard_uncertainty"] - 0.2) < 1e-12

# 46) 非线性 exp(x)：E[exp(X)] = e^(μ+σ²/2)，与一阶结果存在可见偏差
r = client.post(
    "/api/v1/propagate-monte-carlo",
    json={
        "inputs": [
            {"name": "x", "value": 1.0, "unit": "dimensionless", "standard_uncertainty": 0.1}
        ],
        "formula": "exp(x)",
        "samples": 100000,
        "seed": 7,
    },
)
show("46) exp(x) nonlinear mean shift", r.json())
assert r.status_code == 200
b = r.json()
assert abs(b["result"]["value"] - math.e * math.exp(0.005)) < 0.01
assert abs(b["deviation"]["mean_relative"]) > 1e-4

# 47) 常量公式：uc=0，分位数三点相同，一阶 uc=0 时相对偏差为 null
r = client.post(
    "/api/v1/propagate-monte-carlo",
    json={
        "inputs": [
            {"name": "x", "value": 1.0, "unit": "m", "standard_uncertainty": 0.1}
        ],
        "formula": "pi*E",
        "samples": 2000,
        "seed": 1,
    },
)
assert r.status_code == 200
b = r.json()
assert b["result"]["standard_uncertainty"] == 0.0
assert b["quantiles"]["2.5%"] == b["quantiles"]["97.5%"]
assert b["deviation"]["standard_uncertainty_relative"] is None

# 48) 复用既有校验：相关矩阵越界 / 非半正定 / 未知符号 / 名称重复
r = client.post(
    "/api/v1/propagate-monte-carlo",
    json={
        "inputs": [
            {"name": "x", "value": 1, "unit": "m", "standard_uncertainty": 0.1},
            {"name": "y", "value": 2, "unit": "m", "standard_uncertainty": 0.1},
        ],
        "formula": "x+y",
        "correlation_matrix": [[1, 1.2], [1.2, 1]],
        "samples": 2000,
    },
)
assert r.status_code == 422 and r.json()["error"]["code"] == "correlation_out_of_range"
r = client.post(
    "/api/v1/propagate-monte-carlo",
    json={
        "inputs": [
            {"name": "x", "value": 1, "unit": "m", "standard_uncertainty": 0.1},
            {"name": "y", "value": 2, "unit": "m", "standard_uncertainty": 0.1},
            {"name": "z", "value": 3, "unit": "m", "standard_uncertainty": 0.1},
        ],
        "formula": "x+y+z",
        "correlation_matrix": [
            [1, 0.9, 0.9],
            [0.9, 1, -0.9],
            [0.9, -0.9, 1],
        ],
        "samples": 2000,
    },
)
assert r.status_code == 422 and r.json()["error"]["code"] == "not_positive_semidefinite"
r = client.post(
    "/api/v1/propagate-monte-carlo",
    json={
        "inputs": [
            {"name": "x", "value": 1, "unit": "m", "standard_uncertainty": 0.1}
        ],
        "formula": "x+y",
        "samples": 2000,
    },
)
assert r.status_code == 422 and r.json()["error"]["code"] == "unknown_symbol"
r = client.post(
    "/api/v1/propagate-monte-carlo",
    json={
        "inputs": [
            {"name": "x", "value": 1, "unit": "m", "standard_uncertainty": 0.1},
            {"name": "x", "value": 2, "unit": "m", "standard_uncertainty": 0.1},
        ],
        "formula": "x",
        "samples": 2000,
    },
)
assert r.status_code == 422 and r.json()["error"]["code"] == "duplicate_name"

# 49) 缺省相关矩阵回显单位阵；自定义 log10/log2 正常
r = client.post(
    "/api/v1/propagate-monte-carlo",
    json={
        "inputs": [
            {"name": "x", "value": 100.0, "unit": "dimensionless", "standard_uncertainty": 1.0}
        ],
        "formula": "log10(x) + log2(x)",
        "samples": 4000,
        "seed": 1,
    },
)
assert r.status_code == 200, r.text
b = r.json()
assert b["correlation_matrix"] == [[1.0]]
assert abs(b["result"]["value"] - (2.0 + math.log2(100.0))) < 0.02

# --- POST /api/v1/conformity ----------------------------------------------

_Phi = lambda z: 0.5 * math.erfc(-z / math.sqrt(2.0))
_Q = lambda z: 0.5 * math.erfc(z / math.sqrt(2.0))
_UC = 0.39 ** 0.5  # P=U*I, r=0.5 时的合成标准不确定度

_CONF_BASE = {
    "inputs": [
        {"name": "U", "value": 10.0, "unit": "V", "standard_uncertainty": 0.1},
        {"name": "I", "value": 2.0, "unit": "A", "standard_uncertainty": 0.05},
    ],
    "formula": "U*I",
    "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
    "confidence_threshold": 0.95,
}

# 50) 双侧规格限 18.5~21.5 W，t=0.95 -> 合格；接受区间边界处 p=t
r = client.post(
    "/api/v1/conformity",
    json={
        **_CONF_BASE,
        "lower_limit": {"value": 18.5, "unit": "W"},
        "upper_limit": {"value": 21.5, "unit": "W"},
    },
)
show("50) conformity pass, two-sided 18.5~21.5 W", r.json())
assert r.status_code == 200, r.text
b = r.json()
assert b["decision"] == "pass"
assert abs(b["probability"] - (2 * _Phi(1.5 / _UC) - 1)) < 1e-12
assert b["confidence_threshold"] == 0.95
assert b["result"] == {"value": 20.0, "unit": "W", "standard_uncertainty": _UC}
assert b["lower_limit"] == {"value": 18.5, "unit": "W"}
assert b["upper_limit"] == {"value": 21.5, "unit": "W"}
ai = b["acceptance_interval"]
assert ai["unit"] == "W" and not ai["is_empty"]
assert ai["lower"] < 20.0 < ai["upper"]
# 边界关于中点对称，且边界处概率恰为 t
assert abs(ai["lower"] + ai["upper"] - 40.0) < 1e-9
for yb in (ai["lower"], ai["upper"]):
    assert abs((_Phi((21.5 - yb) / _UC) - _Phi((18.5 - yb) / _UC)) - 0.95) < 1e-12

# 51) 窄区间 -> 待定；中点最大概率仍低于阈值 -> 接受区间为空
r = client.post(
    "/api/v1/conformity",
    json={
        **_CONF_BASE,
        "lower_limit": {"value": 19.9, "unit": "W"},
        "upper_limit": {"value": 20.1, "unit": "W"},
    },
)
show("51) conformity indeterminate, empty acceptance", r.json())
assert r.status_code == 200
b = r.json()
assert b["decision"] == "indeterminate"
assert abs(b["probability"] - (_Phi(0.1 / _UC) - _Phi(-0.1 / _UC))) < 1e-12
assert b["acceptance_interval"] == {
    "lower": None,
    "upper": None,
    "unit": "W",
    "is_empty": True,
}

# 52) 区间整体偏上 -> 不合格（上尾差保留非零概率）
r = client.post(
    "/api/v1/conformity",
    json={
        **_CONF_BASE,
        "lower_limit": {"value": 21.5, "unit": "W"},
        "upper_limit": {"value": 22.5, "unit": "W"},
    },
)
show("52) conformity fail", r.json())
assert r.status_code == 200
b = r.json()
assert b["decision"] == "fail"
assert abs(b["probability"] - (_Q(1.5 / _UC) - _Q(2.5 / _UC))) < 1e-15

# 53) 仅上限 -> 半无限接受区间 (-inf, U - uc*z_t]
r = client.post(
    "/api/v1/conformity",
    json={**_CONF_BASE, "upper_limit": {"value": 21.5, "unit": "W"}},
)
show("53) upper limit only", r.json())
assert r.status_code == 200
b = r.json()
assert b["decision"] == "pass" and b["lower_limit"] is None
assert abs(b["probability"] - _Phi(1.5 / _UC)) < 1e-15
_zt = 1.6448536269514715  # Phi^{-1}(0.95)
ai = b["acceptance_interval"]
assert ai["lower"] is None and not ai["is_empty"]
assert abs(ai["upper"] - (21.5 - _UC * _zt)) < 1e-12

# 54) 仅下限 -> 半无限接受区间 [L + uc*z_t, +inf)
r = client.post(
    "/api/v1/conformity",
    json={**_CONF_BASE, "lower_limit": {"value": 18.5, "unit": "W"}},
)
show("54) lower limit only", r.json())
assert r.status_code == 200
b = r.json()
assert b["decision"] == "pass" and b["upper_limit"] is None
assert abs(b["probability"] - _Q(-1.5 / _UC)) < 1e-15
ai = b["acceptance_interval"]
assert ai["upper"] is None
assert abs(ai["lower"] - (18.5 + _UC * _zt)) < 1e-12

# 55) u_c = 0 按确定值处理：区间内 p=1 合格，区间外 p=0 不合格；接受区间即规格区间
_DET = {
    "inputs": [{"name": "x", "value": 5.0, "unit": "m", "standard_uncertainty": 0.0}],
    "formula": "x*2",
    "confidence_threshold": 0.9,
}
r = client.post(
    "/api/v1/conformity",
    json={**_DET, "lower_limit": {"value": 9, "unit": "m"}, "upper_limit": {"value": 11, "unit": "m"}},
)
show("55) uc=0 deterministic inside", r.json())
assert r.status_code == 200
b = r.json()
assert b["decision"] == "pass" and b["probability"] == 1.0
assert b["result"]["standard_uncertainty"] == 0.0
assert b["acceptance_interval"] == {"lower": 9.0, "upper": 11.0, "unit": "m", "is_empty": False}
r = client.post(
    "/api/v1/conformity",
    json={**_DET, "lower_limit": {"value": 10.5, "unit": "m"}, "upper_limit": {"value": 11, "unit": "m"}},
)
assert r.json()["decision"] == "fail" and r.json()["probability"] == 0.0
# 单侧 + uc=0：接受区间端点即界限本身
r = client.post("/api/v1/conformity", json={**_DET, "upper_limit": {"value": 10, "unit": "m"}})
assert r.json()["acceptance_interval"]["upper"] == 10.0
r = client.post("/api/v1/conformity", json={**_DET, "lower_limit": {"value": 10, "unit": "m"}})
assert r.json()["acceptance_interval"]["lower"] == 10.0

# 56) 界限单位换算：18500 mW = 18.5 W，0.0215 kW = 21.5 W
r = client.post(
    "/api/v1/conformity",
    json={
        **_CONF_BASE,
        "lower_limit": {"value": 18500, "unit": "mW"},
        "upper_limit": {"value": 0.0215, "unit": "kW"},
    },
)
show("56) limit unit conversion mW/kW -> W", r.json())
assert r.status_code == 200
b = r.json()
assert b["decision"] == "pass"
assert b["lower_limit"] == {"value": 18.5, "unit": "W"}
assert b["upper_limit"] == {"value": 21.5, "unit": "W"}

# 57) 远尾非零概率：区间 [30,31] m、y=0、uc=1 -> p ≈ 4.9e-198（朴素 1-Phi 会饱和为 0）
r = client.post(
    "/api/v1/conformity",
    json={
        "inputs": [{"name": "x", "value": 0.0, "unit": "m", "standard_uncertainty": 1.0}],
        "formula": "x",
        "lower_limit": {"value": 30, "unit": "m"},
        "upper_limit": {"value": 31, "unit": "m"},
        "confidence_threshold": 0.95,
    },
)
show("57) far-tail non-zero probability", r.json())
assert r.status_code == 200
b = r.json()
assert b["decision"] == "fail"
assert 0.0 < b["probability"] < 1e-190
assert abs(b["probability"] - (_Q(30.0) - _Q(31.0))) / b["probability"] < 1e-10

# 58) 界限量纲不一致 -> 422 定位到界限单位
r = client.post("/api/v1/conformity", json={**_CONF_BASE, "lower_limit": {"value": 1, "unit": "A"}})
show("58) limit unit A vs W -> 422", r.json())
assert r.status_code == 422
assert r.json()["error"]["code"] == "incompatible_unit"
assert r.json()["error"]["loc"] == ["lower_limit", "unit"]
r = client.post("/api/v1/conformity", json={**_CONF_BASE, "upper_limit": {"value": 1, "unit": "s"}})
assert r.status_code == 422
assert r.json()["error"]["loc"] == ["upper_limit", "unit"]

# 59) 双侧下限不小于上限 -> 422（含相等情形）
r = client.post(
    "/api/v1/conformity",
    json={**_CONF_BASE, "lower_limit": {"value": 21, "unit": "W"}, "upper_limit": {"value": 18.5, "unit": "W"}},
)
show("59) lower >= upper -> 422", r.json())
assert r.status_code == 422
assert r.json()["error"]["code"] == "invalid_spec_limits"
assert r.json()["error"]["loc"] == ["lower_limit", "value"]
r = client.post(
    "/api/v1/conformity",
    json={**_CONF_BASE, "lower_limit": {"value": 20, "unit": "W"}, "upper_limit": {"value": 20, "unit": "W"}},
)
assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_spec_limits"

# 60) 未给任何界限 -> 422 missing_spec_limits
r = client.post("/api/v1/conformity", json=_CONF_BASE)
show("60) no limits -> 422", r.json())
assert r.status_code == 422
assert r.json()["error"]["code"] == "missing_spec_limits"

# 61) 置信阈值越界（须严格满足 0.5 < t < 1）-> 422 定位到字段
for bad_t in (0.5, 1.0, 0.3, 1.5):
    r = client.post(
        "/api/v1/conformity",
        json={**_CONF_BASE, "upper_limit": {"value": 21, "unit": "W"}, "confidence_threshold": bad_t},
    )
    assert r.status_code == 422 and r.json()["error"]["loc"] == ["confidence_threshold"], (bad_t, r.json())

# 62) 复用传播校验：相关矩阵越界 / 量纲不一致 / 未声明符号
r = client.post(
    "/api/v1/conformity",
    json={**_CONF_BASE, "upper_limit": {"value": 21, "unit": "W"}, "correlation_matrix": [[1, 1.2], [1.2, 1]]},
)
assert r.status_code == 422 and r.json()["error"]["code"] == "correlation_out_of_range"
r = client.post(
    "/api/v1/conformity",
    json={**_CONF_BASE, "upper_limit": {"value": 21, "unit": "W"}, "formula": "U+I"},
)
assert r.status_code == 422 and r.json()["error"]["code"] == "dimension_mismatch"
r = client.post(
    "/api/v1/conformity",
    json={**_CONF_BASE, "upper_limit": {"value": 21, "unit": "W"}, "formula": "U+V"},
)
assert r.status_code == 422 and r.json()["error"]["code"] == "unknown_symbol"

# --- POST /api/v1/repeatability -------------------------------------------

_REP_READINGS = [10.00, 10.01, 10.03, 10.02, 10.00, 9.98, 9.99, 10.01]

# 63) 基本示例（n=8, L=2，正自相关）：精确有理数结果
#   mean=10.005; gamma0=0.000225, gamma1=0.000103125, gamma2=-0.000075
#   u_A^2 = [gamma0 + 2*(2/3*gamma1 + 1/3*gamma2)]/8 = 1/25600 -> u_A=0.00625
#   n_eff = 8*gamma0 / 括号 = 5.76
r = client.post(
    "/api/v1/repeatability",
    json={"readings": _REP_READINGS, "unit": "V", "max_lag": 2},
)
show("63) repeatability n=8 L=2 (u_A=0.00625 V, n_eff=5.76)", r.json())
assert r.status_code == 200, r.text
b = r.json()
assert b["number_of_readings"] == 8 and b["max_lag"] == 2
assert b["result"]["unit"] == "V"
assert abs(b["result"]["value"] - 10.005) < 1e-12
assert abs(b["result"]["standard_uncertainty"] - 0.00625) < 1e-12
assert [g["lag"] for g in b["autocovariances"]] == [0, 1, 2]
assert all(g["unit"] == "V ** 2" for g in b["autocovariances"])
assert abs(b["autocovariances"][0]["value"] - 0.000225) < 1e-12
assert abs(b["autocovariances"][1]["value"] - 0.000103125) < 1e-12
assert abs(b["autocovariances"][2]["value"] - (-0.000075)) < 1e-12
assert abs(b["variance"] - 0.0000390625) < 1e-12
assert b["variance_unit"] == "V ** 2"
assert abs(b["effective_sample_size"] - 5.76) < 1e-10

# 63b) u_A 可直接填入 /api/v1/propagate 的 standard_uncertainty
r2 = client.post(
    "/api/v1/propagate",
    json={
        "inputs": [
            {"name": "U", "value": 10.0, "unit": "V",
             "standard_uncertainty": b["result"]["standard_uncertainty"]}
        ],
        "formula": "U",
    },
)
assert r2.status_code == 200, r2.text
assert abs(r2.json()["result"]["standard_uncertainty"] - 0.00625) < 1e-12

# 64) L=0：u_A^2 = gamma0/n，仅一个滞后；n_eff 恒为 n
r = client.post(
    "/api/v1/repeatability",
    json={"readings": _REP_READINGS, "unit": "V", "max_lag": 0},
)
show("64) repeatability L=0 (u_A^2=gamma0/n, n_eff=n)", r.json())
assert r.status_code == 200
b = r.json()
assert len(b["autocovariances"]) == 1 and b["autocovariances"][0]["lag"] == 0
assert abs(b["autocovariances"][0]["value"] - 0.000225) < 1e-12
assert abs(b["variance"] - 0.000225 / 8) < 1e-12
assert abs(b["result"]["standard_uncertainty"] - (0.000225 / 8) ** 0.5) < 1e-12
assert b["effective_sample_size"] == 8.0

# 65) 最少 3 个读数、L=n-1 上限
r = client.post(
    "/api/v1/repeatability",
    json={"readings": [1.0, 2.0, 3.0], "unit": "m", "max_lag": 2},
)
show("65) n=3 L=2", r.json())
assert r.status_code == 200, r.text
b = r.json()
assert abs(b["result"]["value"] - 2.0) < 1e-12
assert b["result"]["unit"] == "m"
# mean=2, d=[-1,0,1]: gamma0=(1+0+1)/3=2/3, gamma1=0, gamma2=-1/3
assert abs(b["autocovariances"][0]["value"] - 2 / 3) < 1e-12
assert abs(b["autocovariances"][1]["value"] - 0.0) < 1e-12
assert abs(b["autocovariances"][2]["value"] - (-1 / 3)) < 1e-12
assert all(g["unit"] == "m ** 2" for g in b["autocovariances"])
assert abs(b["effective_sample_size"] - 4.5) < 1e-10

# 66) 全部读数相同：u_A=0，长程方差分母为 0 -> n_eff=null
r = client.post(
    "/api/v1/repeatability",
    json={"readings": [5.0, 5.0, 5.0, 5.0], "unit": "Pa", "max_lag": 2},
)
show("66) constant readings -> u_A=0, n_eff=null", r.json())
assert r.status_code == 200
b = r.json()
assert b["result"] == {"value": 5.0, "unit": "Pa", "standard_uncertainty": 0.0}
assert all(g["value"] == 0.0 for g in b["autocovariances"])
assert b["variance"] == 0.0
assert b["variance_unit"] == "Pa ** 2"
assert b["effective_sample_size"] is None

# 67) 负自相关（交替序列）：n_eff 可大于 n；Bartlett 保证长程方差非负
r = client.post(
    "/api/v1/repeatability",
    json={"readings": [-1.0, 1.0, -1.0, 1.0], "unit": "dimensionless", "max_lag": 1},
)
show("67) alternating L=1 (gamma0=1, gamma1=-0.75, n_eff=16)", r.json())
assert r.status_code == 200
b = r.json()
assert abs(b["autocovariances"][0]["value"] - 1.0) < 1e-12
assert abs(b["autocovariances"][1]["value"] - (-0.75)) < 1e-12
assert b["autocovariances"][0]["unit"] == "dimensionless"
assert abs(b["result"]["standard_uncertainty"] - 0.25) < 1e-12
assert abs(b["effective_sample_size"] - 16.0) < 1e-10

# 68) 单位保持：mV 回显；degC 偏移单位均值仍按声明数值、平方单位可展示
r = client.post(
    "/api/v1/repeatability",
    json={"readings": [100.0, 102.0, 101.0], "unit": "mV", "max_lag": 1},
)
assert r.status_code == 200
b = r.json()
assert b["result"]["unit"] == "mV"
assert b["autocovariances"][0]["unit"] == "mV ** 2"
r = client.post(
    "/api/v1/repeatability",
    json={"readings": [20.0, 20.5, 20.2, 20.3], "unit": "degC", "max_lag": 2},
)
show("68) degC offset unit", r.json())
assert r.status_code == 200
b = r.json()
assert abs(b["result"]["value"] - 20.25) < 1e-12 and b["result"]["unit"] == "°C"
assert b["autocovariances"][0]["unit"] == "°C ** 2"
assert b["variance_unit"] == "°C ** 2"

# 69) 读数少于 3 个 -> 422 loc readings
r = client.post(
    "/api/v1/repeatability",
    json={"readings": [1.0, 2.0], "unit": "V", "max_lag": 0},
)
show("69) fewer than 3 readings -> 422", r.json())
assert r.status_code == 422 and r.json()["error"]["loc"] == ["readings"]

# 70) NaN/Infinity -> 422（原始 JSON 文本发送）
for token in ("NaN", "Infinity", "-Infinity"):
    r = client.post(
        "/api/v1/repeatability",
        content='{"readings": [1.0, %s, 3.0], "unit": "V", "max_lag": 0}' % token,
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422 and r.json()["error"]["loc"][0] == "readings", token

# 71) 单位非法 -> 422 loc unit（unknown_unit / empty_unit）
r = client.post(
    "/api/v1/repeatability",
    json={"readings": [1.0, 2.0, 3.0], "unit": "frobnicate", "max_lag": 0},
)
show("71) unknown unit -> 422", r.json())
assert r.status_code == 422
assert r.json()["error"]["code"] == "unknown_unit"
assert r.json()["error"]["loc"] == ["unit"]
r = client.post(
    "/api/v1/repeatability",
    json={"readings": [1.0, 2.0, 3.0], "unit": "   ", "max_lag": 0},
)
assert r.status_code == 422 and r.json()["error"]["loc"] == ["unit"]

# 72) L >= n -> 422 lag_too_large loc max_lag
for bad_l in (3, 99):
    r = client.post(
        "/api/v1/repeatability",
        json={"readings": [1.0, 2.0, 3.0], "unit": "V", "max_lag": bad_l},
    )
    show(f"72) L={bad_l} >= n -> 422", r.json())
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "lag_too_large"
    assert r.json()["error"]["loc"] == ["max_lag"]

# 73) L 为负数或非整数 -> 422 loc max_lag（pydantic）
for bad_l in (-1, 1.5):
    r = client.post(
        "/api/v1/repeatability",
        json={"readings": [1.0, 2.0, 3.0], "unit": "V", "max_lag": bad_l},
    )
    assert r.status_code == 422 and r.json()["error"]["loc"] == ["max_lag"], bad_l

# 74) 缺字段 -> 422 定位到对应字段
for body, field in [
    ({"unit": "V", "max_lag": 0}, "readings"),
    ({"readings": [1.0, 2.0, 3.0], "max_lag": 0}, "unit"),
    ({"readings": [1.0, 2.0, 3.0], "unit": "V"}, "max_lag"),
]:
    r = client.post("/api/v1/repeatability", json=body)
    assert r.status_code == 422 and r.json()["error"]["loc"] == [field], body

# --- POST /api/v1/apply-calibration -----------------------------------------

_CAL_NODES = [
    {
        "indication": {"value": 0.0, "unit": "mV"},
        "correction": {"value": 0.002, "unit": "mV"},
        "correction_standard_uncertainty": {"value": 0.001, "unit": "mV"},
    },
    {
        "indication": {"value": 10.0, "unit": "mV"},
        "correction": {"value": -0.004, "unit": "mV"},
        "correction_standard_uncertainty": {"value": 0.0015, "unit": "mV"},
    },
    {
        "indication": {"value": 20.0, "unit": "mV"},
        "correction": {"value": 0.006, "unit": "mV"},
        "correction_standard_uncertainty": {"value": 0.002, "unit": "mV"},
    },
]
_CAL_CORR = [[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]]

# 75) 基本示例：x=5 mV 落在 [0,10)，t=0.5，weights=[0.5,0.5,0]
#   c = -0.001 mV，slope = -0.0006，y = 4.999 mV
#   u²(c) = 0.25*1e-6 + 0.25*2.25e-6 + 2*0.5*0.5*0.5*0.001*0.0015 = 1.1875e-6
#   u²_x = (1-0.0006)² * 1e-4
r = client.post(
    "/api/v1/apply-calibration",
    json={
        "nodes": _CAL_NODES,
        "correlation_matrix": _CAL_CORR,
        "indication": {"value": 5.0, "unit": "mV"},
        "indication_standard_uncertainty": {"value": 0.01, "unit": "mV"},
    },
)
show("75) apply-calibration x=5 mV (y=4.999 mV)", r.json())
assert r.status_code == 200, r.text
b = r.json()
assert b["number_of_nodes"] == 3
assert b["interval"] == {
    "index": 0,
    "lower": {"value": 0.0, "unit": "mV"},
    "upper": {"value": 10.0, "unit": "mV"},
}
assert b["weights"] == [0.5, 0.5, 0.0]
assert b["correction"] == {"value": -0.001, "unit": "mV"}
assert abs(b["slope"] - (-0.0006)) < 1e-15
assert abs(b["result"]["value"] - 4.999) < 1e-12
assert b["result"]["unit"] == "mV"
_cv = 0.25 * 1e-6 + 0.25 * 2.25e-6 + 2 * 0.5 * 0.5 * 0.5 * 0.001 * 0.0015
_iv = (1.0 - 0.0006) ** 2 * 1e-4
assert abs(_cv - 1.1875e-6) < 1e-20
assert abs(b["correction_variance_contribution"]["value"] - _cv) < 1e-20
assert abs(b["indication_variance_contribution"]["value"] - _iv) < 1e-20
assert b["correction_variance_contribution"]["unit"] == "mV ** 2"
assert b["indication_variance_contribution"]["unit"] == "mV ** 2"
assert abs(b["result"]["standard_uncertainty"] - math.sqrt(_cv + _iv)) < 1e-15
assert b["correlation_matrix"] == _CAL_CORR

# 76) 节点处选右侧区间：x=10 mV 恰为中间节点 -> 区间 [10,20)，t=0
r = client.post(
    "/api/v1/apply-calibration",
    json={
        "nodes": _CAL_NODES,
        "correlation_matrix": _CAL_CORR,
        "indication": {"value": 10.0, "unit": "mV"},
        "indication_standard_uncertainty": {"value": 0.01, "unit": "mV"},
    },
)
show("76) x=10 mV at node -> right interval", r.json())
assert r.status_code == 200, r.text
b = r.json()
assert b["interval"]["index"] == 1
assert b["interval"]["lower"]["value"] == 10.0
assert b["weights"] == [0.0, 1.0, 0.0]
assert b["correction"]["value"] == -0.004
assert abs(b["slope"] - 0.001) < 1e-15
assert abs(b["result"]["value"] - 9.996) < 1e-12
# 单点权重：u²(c) = u_1² = 2.25e-6
assert abs(b["correction_variance_contribution"]["value"] - 2.25e-6) < 1e-20

# 77) 末节点选左侧区间：x=20 mV -> 区间 [10,20]，t=1
r = client.post(
    "/api/v1/apply-calibration",
    json={
        "nodes": _CAL_NODES,
        "correlation_matrix": _CAL_CORR,
        "indication": {"value": 20.0, "unit": "mV"},
        "indication_standard_uncertainty": {"value": 0.01, "unit": "mV"},
    },
)
show("77) x=20 mV last node -> left interval", r.json())
assert r.status_code == 200, r.text
b = r.json()
assert b["interval"]["index"] == 1
assert b["interval"]["upper"]["value"] == 20.0
assert b["weights"] == [0.0, 0.0, 1.0]
assert b["correction"]["value"] == 0.006
assert abs(b["result"]["value"] - 20.006) < 1e-12
assert abs(b["correction_variance_contribution"]["value"] - 4.0e-6) < 1e-20

# 77b) 首节点：x=0 -> 区间 [0,10)，t=0
r = client.post(
    "/api/v1/apply-calibration",
    json={
        "nodes": _CAL_NODES,
        "correlation_matrix": _CAL_CORR,
        "indication": {"value": 0.0, "unit": "mV"},
        "indication_standard_uncertainty": {"value": 0.0, "unit": "mV"},
    },
)
assert r.status_code == 200, r.text
b = r.json()
assert b["interval"]["index"] == 0 and b["weights"] == [1.0, 0.0, 0.0]
assert b["correction"]["value"] == 0.002
assert abs(b["result"]["value"] - 0.002) < 1e-15
assert b["indication_variance_contribution"]["value"] == 0.0

# 78) 越界示值 -> 422 indication_out_of_range，loc 定位到 indication.value 并附范围
for bad_x in (-0.5, 20.0001, 1e6):
    r = client.post(
        "/api/v1/apply-calibration",
        json={
            "nodes": _CAL_NODES,
            "correlation_matrix": _CAL_CORR,
            "indication": {"value": bad_x, "unit": "mV"},
            "indication_standard_uncertainty": {"value": 0.01, "unit": "mV"},
        },
    )
    show(f"78) x={bad_x} out of range -> 422", r.json())
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "indication_out_of_range"
    assert err["loc"] == ["indication", "value"]
    assert err["details"]["min_indication"] == 0.0
    assert err["details"]["max_indication"] == 20.0
    assert err["details"]["unit"] == "mV"

# 79) 节点示值非严格递增（相等或递减）-> 422 定位到后一节点
for bad_indications in ([0.0, 10.0, 10.0], [0.0, 10.0, 9.0]):
    nodes = [
        {**node, "indication": {"value": v, "unit": "mV"}}
        for node, v in zip(_CAL_NODES, bad_indications)
    ]
    r = client.post(
        "/api/v1/apply-calibration",
        json={
            "nodes": nodes,
            "indication": {"value": 5.0, "unit": "mV"},
            "indication_standard_uncertainty": {"value": 0.01, "unit": "mV"},
        },
    )
    show(f"79) indications {bad_indications} -> 422", r.json())
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "indication_not_strictly_increasing"
    assert r.json()["error"]["loc"] == ["nodes", 2, "indication", "value"]

# 80) 单位换算：节点 V、待修正示值 mV、不确定度 µV 混用，结果与 75) 一致
r = client.post(
    "/api/v1/apply-calibration",
    json={
        "nodes": [
            {
                "indication": {"value": 0.0, "unit": "V"},
                "correction": {"value": 2e-6, "unit": "V"},
                "correction_standard_uncertainty": {"value": 1.0, "unit": "uV"},
            },
            {
                "indication": {"value": 0.01, "unit": "V"},
                "correction": {"value": -4e-6, "unit": "V"},
                "correction_standard_uncertainty": {"value": 1.5e-6, "unit": "V"},
            },
            {
                "indication": {"value": 0.02, "unit": "V"},
                "correction": {"value": 6e-6, "unit": "V"},
                "correction_standard_uncertainty": {"value": 0.002, "unit": "mV"},
            },
        ],
        "correlation_matrix": _CAL_CORR,
        "indication": {"value": 5.0, "unit": "mV"},
        "indication_standard_uncertainty": {"value": 1e-5, "unit": "V"},
    },
)
show("80) mixed units V/mV/uV -> display mV", r.json())
assert r.status_code == 200, r.text
b = r.json()
assert b["result"]["unit"] == "mV"
assert abs(b["result"]["value"] - 4.999) < 1e-12
assert abs(b["result"]["standard_uncertainty"] - math.sqrt(_cv + _iv)) < 1e-12
assert b["interval"]["upper"]["value"] == 10.0

# 81) 量纲不一致 -> 422 incompatible_unit，loc 定位到对应单位字段
_bad_unit_cases = [
    ("correction", ["nodes", 0, "correction", "unit"]),
    ("correction_standard_uncertainty", ["nodes", 1, "correction_standard_uncertainty", "unit"]),
    ("indication", ["nodes", 2, "indication", "unit"]),
]
for field, expected_loc in _bad_unit_cases:
    nodes = [dict(node) for node in _CAL_NODES]
    idx = expected_loc[1]
    nodes[idx] = {**nodes[idx], field: {"value": 1.0, "unit": "A"}}
    r = client.post(
        "/api/v1/apply-calibration",
        json={
            "nodes": nodes,
            "indication": {"value": 5.0, "unit": "mV"},
            "indication_standard_uncertainty": {"value": 0.01, "unit": "mV"},
        },
    )
    assert r.status_code == 422, field
    assert r.json()["error"]["code"] == "incompatible_unit", field
    assert r.json()["error"]["loc"] == expected_loc, field
r = client.post(
    "/api/v1/apply-calibration",
    json={
        "nodes": _CAL_NODES,
        "indication": {"value": 5.0, "unit": "mV"},
        "indication_standard_uncertainty": {"value": 0.01, "unit": "s"},
    },
)
show("81) u(x) unit s vs mV -> 422", r.json())
assert r.status_code == 422
assert r.json()["error"]["code"] == "incompatible_unit"
assert r.json()["error"]["loc"] == ["indication_standard_uncertainty", "unit"]

# 82) 相关矩阵校验沿用同一套：形状 / 越界 / 非半正定
r = client.post(
    "/api/v1/apply-calibration",
    json={
        "nodes": _CAL_NODES,
        "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
        "indication": {"value": 5.0, "unit": "mV"},
        "indication_standard_uncertainty": {"value": 0.01, "unit": "mV"},
    },
)
show("82) 2x2 matrix for 3 nodes -> 422", r.json())
assert r.status_code == 422 and r.json()["error"]["code"] == "bad_matrix_shape"
r = client.post(
    "/api/v1/apply-calibration",
    json={
        "nodes": _CAL_NODES,
        "correlation_matrix": [[1.0, 1.2, 0.0], [1.2, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "indication": {"value": 5.0, "unit": "mV"},
        "indication_standard_uncertainty": {"value": 0.01, "unit": "mV"},
    },
)
assert r.status_code == 422
assert r.json()["error"]["code"] == "correlation_out_of_range"
assert r.json()["error"]["loc"] == ["correlation_matrix", 0, 1]
r = client.post(
    "/api/v1/apply-calibration",
    json={
        "nodes": _CAL_NODES,
        "correlation_matrix": [[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]],
        "indication": {"value": 5.0, "unit": "mV"},
        "indication_standard_uncertainty": {"value": 0.01, "unit": "mV"},
    },
)
assert r.status_code == 422 and r.json()["error"]["code"] == "not_positive_semidefinite"

# 83) 缺省相关矩阵 = 单位阵：u²(c) 仅剩对角项
r = client.post(
    "/api/v1/apply-calibration",
    json={
        "nodes": _CAL_NODES,
        "indication": {"value": 5.0, "unit": "mV"},
        "indication_standard_uncertainty": {"value": 0.01, "unit": "mV"},
    },
)
show("83) default identity correlation", r.json())
assert r.status_code == 200, r.text
b = r.json()
assert b["correlation_matrix"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
assert abs(b["correction_variance_contribution"]["value"] - (0.25 * 1e-6 + 0.25 * 2.25e-6)) < 1e-20

# 84) 完全相关/完全负相关的两节点：验证协方差非对角项进入 wᵀΣw
_TWO_NODES = [
    {
        "indication": {"value": 0.0, "unit": "m"},
        "correction": {"value": 0.0, "unit": "m"},
        "correction_standard_uncertainty": {"value": 0.1, "unit": "m"},
    },
    {
        "indication": {"value": 10.0, "unit": "m"},
        "correction": {"value": 0.0, "unit": "m"},
        "correction_standard_uncertainty": {"value": 0.2, "unit": "m"},
    },
]
for r12, expected_u in ((1.0, 0.15), (-1.0, 0.05), (0.0, (0.05 ** 2 + 0.1 ** 2) ** 0.5)):
    r = client.post(
        "/api/v1/apply-calibration",
        json={
            "nodes": _TWO_NODES,
            "correlation_matrix": [[1.0, r12], [r12, 1.0]],
            "indication": {"value": 5.0, "unit": "m"},
            "indication_standard_uncertainty": {"value": 0.0, "unit": "m"},
        },
    )
    assert r.status_code == 200, r12
    b = r.json()
    # u(c) = |0.5*0.1 ± 0.5*0.2|（完全相关时线性组合）
    assert abs(b["result"]["standard_uncertainty"] - expected_u) < 1e-12, r12
    assert b["weights"] == [0.5, 0.5]

# 85) 单节点 -> 422 loc nodes（Pydantic too_short）
r = client.post(
    "/api/v1/apply-calibration",
    json={
        "nodes": _CAL_NODES[:1],
        "indication": {"value": 5.0, "unit": "mV"},
        "indication_standard_uncertainty": {"value": 0.01, "unit": "mV"},
    },
)
show("85) single node -> 422", r.json())
assert r.status_code == 422 and r.json()["error"]["loc"] == ["nodes"]

# 86) 修正值标准不确定度为负 -> 422 定位到字段
nodes = [dict(node) for node in _CAL_NODES]
nodes[0] = {**nodes[0], "correction_standard_uncertainty": {"value": -0.1, "unit": "mV"}}
r = client.post(
    "/api/v1/apply-calibration",
    json={
        "nodes": nodes,
        "indication": {"value": 5.0, "unit": "mV"},
        "indication_standard_uncertainty": {"value": 0.01, "unit": "mV"},
    },
)
show("86) negative correction u -> 422", r.json())
assert r.status_code == 422
assert r.json()["error"]["loc"] == ["nodes", 0, "correction_standard_uncertainty", "value"]

# 87) 未知单位 -> 422 unknown_unit 定位到节点单位字段
nodes = [dict(node) for node in _CAL_NODES]
nodes[1] = {**nodes[1], "indication": {"value": 10.0, "unit": "frobnicate"}}
r = client.post(
    "/api/v1/apply-calibration",
    json={
        "nodes": nodes,
        "indication": {"value": 5.0, "unit": "mV"},
        "indication_standard_uncertainty": {"value": 0.01, "unit": "mV"},
    },
)
show("87) unknown unit -> 422", r.json())
assert r.status_code == 422
assert r.json()["error"]["code"] == "unknown_unit"
assert r.json()["error"]["loc"] == ["nodes", 1, "indication", "unit"]

# 88) 斜率进入示值方差：slope=0.2 时 u²_x = 1.44 u²
r = client.post(
    "/api/v1/apply-calibration",
    json={
        "nodes": [
            {
                "indication": {"value": 0.0, "unit": "m"},
                "correction": {"value": 0.0, "unit": "m"},
                "correction_standard_uncertainty": {"value": 0.0, "unit": "m"},
            },
            {
                "indication": {"value": 10.0, "unit": "m"},
                "correction": {"value": 2.0, "unit": "m"},
                "correction_standard_uncertainty": {"value": 0.0, "unit": "m"},
            },
        ],
        "indication": {"value": 5.0, "unit": "m"},
        "indication_standard_uncertainty": {"value": 0.1, "unit": "m"},
    },
)
show("88) slope=0.2 -> u_x contribution (1.2*0.1)^2", r.json())
assert r.status_code == 200, r.text
b = r.json()
assert abs(b["slope"] - 0.2) < 1e-15
assert b["correction_variance_contribution"]["value"] == 0.0
assert abs(b["indication_variance_contribution"]["value"] - (1.2 * 0.1) ** 2) < 1e-15
assert abs(b["result"]["standard_uncertainty"] - 0.12) < 1e-12
assert abs(b["result"]["value"] - 6.0) < 1e-12

print("\nALL TESTS PASSED")

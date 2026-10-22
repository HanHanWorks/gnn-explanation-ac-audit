#!/usr/bin/env python3
"""Phase 1 审计核心模块：网络构造、边映射、注入推导、AC 潮流可行性审计。

审计语义（评审报告 2.3 操作化协议）：
  给定一个解释器给出的边解释（edge mask top-k = 候选断开集），
  在保持运行点（节点注入不变）的前提下断开这些支路，用 pandapower AC-NR
  检查潮流收敛与运行约束，判定"物理可行性"。

注入推导：节点注入 = 该样本保留边潮流的平衡量（Σ P_ij / Σ Q_ij），
与 PowerGraph 级联数据（AC 口径）同源，无需依赖 Bf 特征。
"""
import os
import re
import copy
import numpy as np
import pandapower as pp
from pandapower.converter.matpower import from_mpc

SYSTEM_M = {
    "ieee24": "datasets/PowerGraph-Node/13_Power_system/IEEE24/System.m",
    "ieee39": "datasets/PowerGraph-Node/13_Power_system/IEEE39/System.m",
    "ieee118": "datasets/PowerGraph-Node/13_Power_system/IEEE118/System.m",
    "uk": "datasets/PowerGraph-Node/13_Power_system/UK/System.m",
}

VOLT_MIN, VOLT_MAX = 0.85, 1.10   # 电压限（p.u.），N-1 应急标准（case118 基准 vmin≈0.87）
LOAD_MAX_PCT = 1e9                # 负载不作为可行判据（PowerGraph System.m 原始工况含设计性过载线路，如 ieee118 544%/uk 181%，见 README）；仅记录


def _parse_mpc_matrices(m_file):
    """解析 System.m 的各矩阵（容忍行内注释/分号/附加列），返回 (baseMVA, bus, gen, branch, gencost)。"""
    t = open(m_file).read()
    def parse_matrix(name):
        m = re.search(r"mpc\.%s\s*=\s*\[(.*?)\];" % name, t, re.S)
        if not m:
            return None
        rows = []
        for ln in m.group(1).split("\n"):
            s = ln.split("%")[0].strip().rstrip(";").strip()
            if not s:
                continue
            rows.append([float(x) for x in s.replace(";", " ").split()])
        return np.array(rows)
    baseMVA = float(re.search(r"mpc\.baseMVA\s*=\s*([\d.]+)", t).group(1))
    return baseMVA, parse_matrix("bus"), parse_matrix("gen"), parse_matrix("branch"), parse_matrix("gencost")


def load_net_fixed(system, root="."):
    """构建 pandapower net（ieee118 的 System.m 有附加列，需截标准列后用 from_ppc）。

    from_mpc（matpowercaseframes）bus 索引 0-based；from_ppc 保留 bus 编号（1-based）。
    """
    m_file = os.path.join(root, SYSTEM_M[system])
    try:
        return from_mpc(m_file)
    except Exception:
        baseMVA, bus, gen, branch, gencost = _parse_mpc_matrices(m_file)
        mpc = {"version": "2", "baseMVA": baseMVA,
               "bus": bus[:, :13], "gen": gen[:, :21], "branch": branch[:, :21]}
        if gencost is not None:
            mpc["gencost"] = gencost
        from pandapower.converter.matpower import from_ppc
        return from_ppc(mpc)


def build_net(system, root="."):
    """构建 pandapower net 并建立 blist 行 -> (kind, idx) 映射。"""
    net = load_net_fixed(system, root)
    # blist（1-based bus 对）-> element
    blist_path = os.path.join(root, "datasets", "PowerGraph", "data_cascades", system, "raw", "blist.mat")
    import mat73
    blist = np.asarray(mat73.loadmat(blist_path)["bList"], dtype=int)  # (E,2) 1-based

    # 探测 bus 偏移：line[0].from_bus 与 blist[0][0]（1-based）的关系
    first_from = net.line["from_bus"].iloc[0] if len(net.line) else net.trafo["hv_bus"].iloc[0]
    offset = int(first_from) - (int(blist[0][0]) - 1)   # from_mpc: 0；from_ppc: 1
    def bus_idx(b): return int(b) - 1 + offset          # blist 1-based -> net bus index

    mapping = []
    line_used = {}
    for (fr, to) in blist:
        f, t = bus_idx(fr), bus_idx(to)
        found = None
        for kind, df in (("line", net.line), ("trafo", net.trafo), ("impedance", net.impedance)):
            if kind == "line":
                frs, tos = df["from_bus"].values, df["to_bus"].values
            elif kind == "trafo":
                frs, tos = df["hv_bus"].values, df["lv_bus"].values
            else:
                frs, tos = df["from_bus"].values, df["to_bus"].values
            for i in range(len(df)):
                if found is None and not line_used.get((kind, i), False):
                    if (frs[i] == f and tos[i] == t) or (frs[i] == t and tos[i] == f):
                        found = (kind, i)
                        line_used[(kind, i)] = True
        assert found is not None, f"blist 行 {(fr,to)} 未匹配到 pandapower 元件"
        mapping.append(found)
    return net, mapping


def derive_injections(ef_sample, blist, nbus, base_mva=100.0, bus_labels=None):
    """从单样本 Ef 潮流 (E,4)=P,Q,X,rating 推导节点注入 (MW/MVar)。

    Ef 为 p.u. 潮流（PowerGraph 用 MATPOWER baseMVA=100），× base_mva 转有名值。
    ``bus_labels`` maps the 1-based MATPOWER bus numbers to positions in the
    returned vectors; pass ``net.bus.index`` so the function works whether
    pandapower keeps 0-based or 1-based bus labels.
    """
    f = np.asarray(ef_sample, dtype=float).reshape(-1, 4)
    if bus_labels is None:
        pos = {b + 1: b for b in range(nbus)}
    else:
        pos = {int(label): k for k, label in enumerate(bus_labels)}
    P = np.zeros(nbus); Q = np.zeros(nbus)
    for j, (fr, to) in enumerate(blist):
        fr, to = int(fr), int(to)
        P[pos[fr]] += f[j, 0]; P[pos[to]] -= f[j, 0]
        Q[pos[fr]] += f[j, 1]; Q[pos[to]] -= f[j, 1]
    return P * base_mva, Q * base_mva


def set_injections(net, P, Q, slack_bus=None):
    """把节点注入写入 net：P>0 -> gen(PV, Q 可调)，P<0 -> load(PQ)；ext_grid 作 slack。

    用 PV 模型为系统提供无功/电压支撑（与 MATPOWER 原 case 的发电机行为一致）。
    """
    nbus = len(net.bus)
    # 清空原有发电/负荷/slack，替换为样本注入
    net.gen.drop(net.gen.index, inplace=True)
    net.load.drop(net.load.index, inplace=True)
    net.sgen.drop(net.sgen.index, inplace=True)
    net.ext_grid.drop(net.ext_grid.index, inplace=True)
    bus_labels = [int(x) for x in net.bus.index]
    if slack_bus is None:
        slack_bus = int(np.argmin(np.abs(P)))   # 选净注入最小的节点，尽量少干扰运行点
    pp.create_ext_grid(net, bus=bus_labels[slack_bus], vm_pu=1.0, name="slack")
    for b in range(nbus):
        if abs(P[b]) > 1e-6:
            label = bus_labels[b]
            if P[b] > 0:
                pp.create_gen(net, bus=label, p_mw=float(P[b]), vm_pu=1.0,
                              min_q_mvar=-1000.0, max_q_mvar=1000.0)
            else:
                pp.create_load(net, bus=label, p_mw=float(-P[b]), q_mvar=float(-Q[b]))
    return net


def audit(net, injections, disconnect, cont_edges=(), volt_min=VOLT_MIN, volt_max=VOLT_MAX, load_max=LOAD_MAX_PCT):
    """断开 disconnect 支路后跑 AC 潮流，返回 (feasible, status, detail)。

    语义：样本的当前状态 = 原始拓扑已断开 contingency 边(cont_edges)；
    候选修改 = 在 cont_edges 基础上再断开解释/随机边(disconnect)。
    空集对照(disconnect=[]) = 仅恢复样本自身状态，应为可行（仿真器状态）。

    status: 'feasible' | 'non_converged' | 'voltage' | 'overload'
    """
    n = copy.deepcopy(net)
    set_injections(n, injections[0], injections[1])
    for kind, idx in list(cont_edges) + list(disconnect):
        if kind == "line":
            n.line.loc[idx, "in_service"] = False
        elif kind == "trafo":
            n.trafo.loc[idx, "in_service"] = False
        else:
            n.impedance.loc[idx, "in_service"] = False
    try:
        pp.runpp(n, algorithm="nr", numba=False)
    except Exception:
        return False, "non_converged", {}
    if not n.converged:
        return False, "non_converged", {}
    v = n.res_bus.vm_pu.values
    if np.isnan(v).any():
        return False, "island", {}
    if v.min() < volt_min or v.max() > volt_max:
        return False, "voltage", {"vmin": float(v.min()), "vmax": float(v.max())}
    lmax = max(n.res_line.loading_percent.max(), n.res_trafo.loading_percent.max())
    if lmax > load_max:
        return False, "overload", {"load_max": float(lmax)}
    return True, "feasible", {"vmin": float(v.min()), "vmax": float(v.max()), "load_max": float(lmax)}


def audit_baseline(net, disconnect, volt_min=VOLT_MIN, volt_max=VOLT_MAX, load_max=LOAD_MAX_PCT):
    """公共工况审计：用 System.m 原始运行点（精确可复现），仅断开候选支路。

    语义（评审报告 2.3 操作化）：解释器建议的拓扑修改（断开候选边）在标准运行点下
    是否物理可行。空集对照 = 原始 case（必然可行，sanity）。

    返回 (feasible, status, detail)；status: feasible|non_converged|island|voltage|overload
    """
    n = copy.deepcopy(net)
    for kind, idx in disconnect:
        if kind == "line":
            n.line.loc[idx, "in_service"] = False
        elif kind == "trafo":
            n.trafo.loc[idx, "in_service"] = False
        else:
            n.impedance.loc[idx, "in_service"] = False
    try:
        pp.runpp(n, algorithm="nr", numba=False)
    except Exception:
        return False, "non_converged", {}
    if not n.converged:
        return False, "non_converged", {}
    v = n.res_bus.vm_pu.values
    if np.isnan(v).any():
        return False, "island", {}          # 孤立节点（拓扑被切断）
    if v.min() < volt_min or v.max() > volt_max:
        return False, "voltage", {"vmin": float(v.min()), "vmax": float(v.max())}
    l = float(np.nanmax(np.concatenate([
        n.res_line.loading_percent.values, n.res_trafo.loading_percent.values])))
    if l > load_max:
        return False, "overload", {"load_max": l}
    return True, "feasible", {"vmin": float(v.min()), "vmax": float(v.max()), "load_max": l}


def pfr_summary(results):
    """统计 PFR 指标族：可行率 + 不可行类型分布。"""
    n = len(results)
    if n == 0:
        return {}
    from collections import Counter
    statuses = Counter(r[1] for r in results)
    return {
        "n": n,
        "PFR": statuses.get("feasible", 0) / n,
        "types": dict(statuses),
    }

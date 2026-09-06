"""Core AC power-flow feasibility audit.

Given an explainer's edge mask (top-k = candidate disconnect set), opens those
branches while keeping the operating point fixed, then runs Newton--Raphson AC
power flow and classifies feasibility.

Injection derivation: nodal injection equals the imbalance of retained-branch
flows for the sample, consistent with the simulation data.
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

VOLT_MIN, VOLT_MAX = 0.85, 1.10
LOAD_MAX_PCT = 1e9

def _parse_mpc_matrices(m_file):
 """ System.m / / ， (baseMVA, bus, gen, branch, gencost) """
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
 """ pandapower net ieee118 System.m ， from_ppc

 from_mpc matpowercaseframes bus 0-based；from_ppc bus 1-based
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
 """ pandapower net blist -> (kind, idx) """
    net = load_net_fixed(system, root)

    blist_path = os.path.join(root, "datasets", "PowerGraph", "data_cascades", system, "raw", "blist.mat")
    import mat73
    blist = np.asarray(mat73.loadmat(blist_path)["bList"], dtype=int)  # (E,2) 1-based

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
 assert found is not None, f"blist {(fr,to)} pandapower "
        mapping.append(found)
    return net, mapping

def derive_injections(ef_sample, blist, nbus, base_mva=100.0):
 """ Ef (E,4)=P,Q,X,rating (MW/MVar)

 Ef p.u. PowerGraph MATPOWER baseMVA=100 ，× base_mva
    """
    f = np.asarray(ef_sample, dtype=float).reshape(-1, 4)
    P = np.zeros(nbus); Q = np.zeros(nbus)
    for j, (fr, to) in enumerate(blist):
        P[fr - 1] += f[j, 0]; P[to - 1] -= f[j, 0]
        Q[fr - 1] += f[j, 1]; Q[to - 1] -= f[j, 1]
    return P * base_mva, Q * base_mva

def set_injections(net, P, Q, slack_bus=None):
 """ net：P>0 -> gen(PV, Q )，P<0 -> load(PQ)；ext_grid slack

    """
    nbus = len(net.bus)

    net.gen.drop(net.gen.index, inplace=True)
    net.load.drop(net.load.index, inplace=True)
    net.sgen.drop(net.sgen.index, inplace=True)
    net.ext_grid.drop(net.ext_grid.index, inplace=True)
    if slack_bus is None:
        slack_bus = int(np.argmin(np.abs(P)))
    pp.create_ext_grid(net, bus=slack_bus, vm_pu=1.0, name="slack")
    for b in range(nbus):
        if abs(P[b]) > 1e-6:
            if P[b] > 0:
                pp.create_gen(net, bus=b, p_mw=float(P[b]), vm_pu=1.0,
                              min_q_mvar=-1000.0, max_q_mvar=1000.0)
            else:
                pp.create_load(net, bus=b, p_mw=float(-P[b]), q_mvar=float(-Q[b]))
    return net

def audit(net, injections, disconnect, cont_edges=(), volt_min=VOLT_MIN, volt_max=VOLT_MAX, load_max=LOAD_MAX_PCT):
 """ disconnect AC ， (feasible, status, detail)

    status: 'feasible' | 'non_converged' | 'voltage' | 'overload'
    """
    n = copy.deepcopy(net)
    set_injections(n, injections[0], injections[1])
    for kind, idx in list(cont_edges) + list(disconnect):
        if kind == "line":
            n.line.loc[idx, "in_service"] = False
        else:
            n.trafo.loc[idx, "in_service"] = False
    try:
        pp.runpp(n, algorithm="nr", numba=False)
    except Exception:
        return False, "non_converged", {}
    if not n.converged:
        return False, "non_converged", {}
    v = n.res_bus.vm_pu.values
    if v.min() < volt_min or v.max() > volt_max:
        return False, "voltage", {"vmin": float(v.min()), "vmax": float(v.max())}
    lmax = max(n.res_line.loading_percent.max(), n.res_trafo.loading_percent.max())
    if lmax > load_max:
        return False, "overload", {"load_max": float(lmax)}
    return True, "feasible", {"vmin": float(v.min()), "vmax": float(v.max()), "load_max": float(lmax)}

def audit_baseline(net, disconnect, volt_min=VOLT_MIN, volt_max=VOLT_MAX, load_max=LOAD_MAX_PCT):
 """ ： System.m ，

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
        return False, "island", {}
    if v.min() < volt_min or v.max() > volt_max:
        return False, "voltage", {"vmin": float(v.min()), "vmax": float(v.max())}
    l = float(np.nanmax(np.concatenate([
        n.res_line.loading_percent.values, n.res_trafo.loading_percent.values])))
    if l > load_max:
        return False, "overload", {"load_max": l}
    return True, "feasible", {"vmin": float(v.min()), "vmax": float(v.max()), "load_max": l}

def pfr_summary(results):
 """ PFR ： + """
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

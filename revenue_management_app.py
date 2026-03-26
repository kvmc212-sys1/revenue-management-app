import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize_scalar, minimize

st.set_page_config(page_title="Revenue Management & Pricing Optimizer", layout="wide")
st.title("Revenue Management & Pricing Optimizer")

# ---------------------------------------------------------------------------
# Demand model definitions
# ---------------------------------------------------------------------------

def linear_demand(p, D, m):
    """Q(p) = D - m*p"""
    return np.maximum(D - m * p, 0)

def constant_elasticity_demand(p, a, e):
    """Q(p) = a * p^(-e)"""
    with np.errstate(divide='ignore', invalid='ignore'):
        q = a * np.power(p, -e)
    return np.where(np.isfinite(q), q, 0)

def exponential_demand(p, a, b):
    """Q(p) = a * exp(-b*p)"""
    return a * np.exp(-b * p)

def logit_demand(p, market_size, beta_0, beta_p):
    """Q(p) = M * exp(beta_0 + beta_p*p) / (1 + exp(beta_0 + beta_p*p))"""
    v = beta_0 + beta_p * p
    prob = np.exp(v) / (1 + np.exp(v))
    return market_size * prob

# ---------------------------------------------------------------------------
# Number formatting helpers
# ---------------------------------------------------------------------------

def fmt_dollar(x):
    """Format as dollar with commas: $1,234.56"""
    return f"${x:,.2f}"

def fmt_qty(x):
    """Format quantity with commas: 1,234.1"""
    return f"{x:,.1f}"

def fmt_qty2(x):
    """Format quantity with 2 decimals: 1,234.56"""
    return f"{x:,.2f}"

def fmt_int(x):
    """Format integer-like with commas: 1,234"""
    return f"{x:,.0f}"

def python_output(title, code_lines, result_lines):
    """Display a Python-style printout in an expander.
    code_lines: list of strings (Python code)
    result_lines: list of strings (printed output)
    """
    with st.expander(f"Python Output — {title}"):
        code = ""
        for cl in code_lines:
            code += f">>> {cl}\n"
        for rl in result_lines:
            code += f"{rl}\n"
        st.code(code, language="python")

# ---------------------------------------------------------------------------
# Sidebar – mode selection
# ---------------------------------------------------------------------------

mode = st.sidebar.radio(
    "Mode",
    [
        "Single Segment Optimization",
        "Multi-Segment Optimization",
        "Dynamic Pricing with Inventory",
        "Marginal Revenue Allocation",
        "Marginal Value of Capacity",
        "Demand Estimation from Data",
        "Newsvendor / Quantity",
        "Incentive Compatible Pricing",
        "Loan Pricing Optimization",
    ],
    index=0,
)

# =====================================================================
# MODE 1 – Single segment (solve for any variable)
# =====================================================================
if mode == "Single Segment Optimization":
    st.header("Single Segment Optimization")

    solve_for = st.radio(
        "Solve for",
        ["Optimal Price (given demand model)",
         "Optimal Quantity (inverse demand)",
         "Solve for Price (given Q)",
         "Solve for Quantity (given P)",
         "Solve for Revenue (given P or Q)",
         "Solve for Elasticity (given P)"],
        horizontal=True,
    )

    # =================================================================
    # SUB-MODE: Optimal Price
    # =================================================================
    if solve_for == "Optimal Price (given demand model)":
        demand_type = st.selectbox(
            "Demand Model",
            ["Linear", "Constant Elasticity", "Exponential", "Logit"],
        )

        col1, col2 = st.columns(2)

        if demand_type == "Linear":
            with col1:
                st.subheader("Parameters")
                st.latex(r"Q(p) = D - m \cdot p")
                D = st.number_input("D (max demand / intercept)", value=100.0, min_value=1.0, step=10.0)
                m = st.number_input("m (slope)", value=2.0, min_value=0.01, step=0.5)
            demand_fn = lambda p: linear_demand(p, D, m)
            p_max = D / m
            analytical_price = D / (2 * m)
            analytical_label = f"Analytical optimum: p* = D/(2m) = {analytical_price:,.2f}"

        elif demand_type == "Constant Elasticity":
            with col1:
                st.subheader("Parameters")
                st.latex(r"Q(p) = a \cdot p^{-\varepsilon}")
                a = st.number_input("a (scale)", value=1000.0, min_value=1.0, step=100.0)
                e = st.number_input("ε (elasticity, must be > 1 for finite optimum)", value=2.0, min_value=1.01, step=0.1)
            demand_fn = lambda p: constant_elasticity_demand(p, a, e)
            p_max = 500.0
            analytical_price = None
            analytical_label = "No finite unconstrained optimum without marginal cost"

        elif demand_type == "Exponential":
            with col1:
                st.subheader("Parameters")
                st.latex(r"Q(p) = a \cdot e^{-b \cdot p}")
                a = st.number_input("a (scale)", value=100.0, min_value=1.0, step=10.0)
                b = st.number_input("b (decay rate)", value=0.05, min_value=0.001, step=0.01, format="%.3f")
            demand_fn = lambda p: exponential_demand(p, a, b)
            p_max = 10.0 / b
            analytical_price = 1.0 / b
            analytical_label = f"Analytical optimum: p* = 1/b = {analytical_price:,.2f}"

        else:  # Logit
            with col1:
                st.subheader("Parameters")
                st.latex(r"Q(p) = M \cdot \frac{e^{\beta_0 + \beta_p p}}{1 + e^{\beta_0 + \beta_p p}}")
                M = st.number_input("M (market size)", value=1000.0, min_value=1.0, step=100.0)
                beta_0 = st.number_input("β₀ (intercept utility)", value=3.0, step=0.5)
                beta_p = st.number_input("βₚ (price sensitivity, negative)", value=-0.1, max_value=-0.001, step=0.01, format="%.3f")
            demand_fn = lambda p: logit_demand(p, M, beta_0, beta_p)
            p_max = -2 * beta_0 / beta_p
            analytical_price = None
            analytical_label = "Optimum found numerically"

        with col1:
            st.subheader("Constraints (optional)")
            use_cost = st.checkbox("Include marginal cost")
            cost = st.number_input("Marginal cost per unit", value=0.0, min_value=0.0, step=1.0) if use_cost else 0.0
            use_capacity = st.checkbox("Capacity constraint")
            capacity = st.number_input("Max inventory / capacity", value=30.0, min_value=1.0, step=5.0) if use_capacity else None

        def neg_profit(p):
            if p <= 0:
                return 0
            q = float(demand_fn(np.array([p]))[0]) if hasattr(demand_fn(np.array([p])), '__len__') else float(demand_fn(p))
            return -(p - cost) * q

        best_p = None
        if use_capacity:
            from scipy.optimize import minimize as sp_minimize
            def neg_profit_arr(x):
                return neg_profit(x[0])
            def capacity_constraint(x):
                p = x[0]
                q = float(demand_fn(np.array([p]))[0]) if hasattr(demand_fn(np.array([p])), '__len__') else float(demand_fn(p))
                return capacity - q
            res = sp_minimize(neg_profit_arr, [p_max * 0.3],
                              method='SLSQP',
                              bounds=[(0.01, p_max)],
                              constraints={'type': 'ineq', 'fun': capacity_constraint})
            best_p = res.x[0]
        else:
            res = minimize_scalar(neg_profit, bounds=(0.01, p_max), method='bounded')
            best_p = res.x

        best_q_val = demand_fn(np.array([best_p]))[0] if hasattr(demand_fn(np.array([best_p])), '__len__') else demand_fn(best_p)
        best_q = float(best_q_val)
        best_revenue = best_p * best_q
        best_profit = (best_p - cost) * best_q

        dp = best_p * 1e-5
        q_plus = float(demand_fn(np.array([best_p + dp]))[0]) if hasattr(demand_fn(np.array([best_p + dp])), '__len__') else float(demand_fn(best_p + dp))
        q_minus = float(demand_fn(np.array([best_p - dp]))[0]) if hasattr(demand_fn(np.array([best_p - dp])), '__len__') else float(demand_fn(best_p - dp))
        dQdp = (q_plus - q_minus) / (2 * dp)
        elasticity_at_opt = dQdp * best_p / best_q if best_q > 0 else 0

        with col2:
            st.subheader("Optimal Solution")
            r1, r2, r3 = st.columns(3)
            r1.metric("Optimal Price", fmt_dollar(best_p))
            r1.caption("The price that maximizes revenue (or profit). Set where marginal revenue equals marginal cost (or zero).")
            r2.metric("Quantity Sold", fmt_qty(best_q))
            r2.caption("Expected units sold at the optimal price, determined by the demand function Q(p*).")
            r3.metric("Revenue", fmt_dollar(best_revenue))
            r3.caption("Total revenue = Price × Quantity. This is the maximum achievable given the demand model.")
            if use_cost:
                st.metric("Profit (Revenue − Cost)", fmt_dollar(best_profit))
                st.caption("Revenue minus total variable cost. Profit = (p − c) × Q.")
            st.metric("Price Elasticity at p*", f"{elasticity_at_opt:.3f}")
            if use_cost and cost > 0:
                st.caption("Percentage change in quantity demanded for a 1% change in price. At the profit-maximizing price with MC > 0, |ε| > 1 (Lerner index: markup = 1/|ε|).")
            else:
                st.caption("Percentage change in quantity demanded for a 1% change in price. At the revenue-maximizing price, elasticity = −1.")
            if abs(elasticity_at_opt) > 1:
                st.caption("Demand is **elastic** (|ε| > 1): a price increase reduces revenue.")
            elif abs(elasticity_at_opt) < 1:
                st.caption("Demand is **inelastic** (|ε| < 1): a price increase raises revenue.")
            else:
                st.caption("Demand is **unit elastic** (|ε| = 1): revenue is maximized.")
            if use_capacity and best_q >= capacity - 0.01:
                st.info(f"Capacity constraint is **binding** at {fmt_int(capacity)} units.")
            if analytical_price and not use_capacity and cost == 0:
                st.caption(analytical_label)

        prices = np.linspace(0.01, p_max * 0.95, 500)
        quantities = demand_fn(prices)
        revenues = prices * quantities
        profits = (prices - cost) * quantities

        dp_arr = prices * 1e-5
        q_plus_arr = demand_fn(prices + dp_arr)
        q_minus_arr = demand_fn(prices - dp_arr)
        dQdp_arr = (q_plus_arr - q_minus_arr) / (2 * dp_arr)
        with np.errstate(divide='ignore', invalid='ignore'):
            elasticity_arr = dQdp_arr * prices / quantities
        elasticity_arr = np.where(np.isfinite(elasticity_arr), elasticity_arr, 0)

        # Python printout
        _code = [
            f"from scipy.optimize import minimize_scalar",
            f"import numpy as np",
            f"",
            f"# Demand model: {demand_type}",
        ]
        if demand_type == "Linear":
            _code += [f"D, m = {D}, {m}", f"demand = lambda p: max(D - m * p, 0)"]
        elif demand_type == "Constant Elasticity":
            _code += [f"a, e = {a}, {e}", f"demand = lambda p: a * p**(-e)"]
        elif demand_type == "Exponential":
            _code += [f"a, b = {a}, {b}", f"demand = lambda p: a * np.exp(-b * p)"]
        else:
            _code += [f"M, beta_0, beta_p = {M}, {beta_0}, {beta_p}",
                      f"demand = lambda p: M * np.exp(beta_0+beta_p*p) / (1+np.exp(beta_0+beta_p*p))"]
        if use_cost:
            _code.append(f"cost = {cost}")
        _code += [
            f"",
            f"neg_profit = lambda p: -((p - {cost}) * demand(p))",
            f"result = minimize_scalar(neg_profit, bounds=(0.01, {p_max:.2f}), method='bounded')",
            f"p_star = result.x",
            f"q_star = demand(p_star)",
            f"revenue = p_star * q_star",
            f"print(f'Optimal Price:  ${{p_star:.2f}}')",
            f"print(f'Quantity Sold:  {{q_star:.1f}}')",
            f"print(f'Revenue:        ${{revenue:.2f}}')",
        ]
        if use_cost:
            _code.append(f"print(f'Profit:         ${{(p_star - {cost}) * q_star:.2f}}')")
        _code.append(f"print(f'Elasticity:     {{elasticity:.3f}}')")
        _results = [
            f"Optimal Price:  ${best_p:.2f}",
            f"Quantity Sold:  {best_q:.1f}",
            f"Revenue:        ${best_revenue:.2f}",
        ]
        if use_cost:
            _results.append(f"Profit:         ${best_profit:.2f}")
        _results.append(f"Elasticity:     {elasticity_at_opt:.3f}")
        if use_capacity and best_q >= capacity - 0.01:
            _results.append(f"Capacity constraint is BINDING at {capacity:.0f} units.")
        python_output("Optimal Price", _code, _results)

        fig = make_subplots(rows=1, cols=3, subplot_titles=("Demand Curve", "Revenue / Profit Curve", "Price Elasticity ε(p)"))
        fig.add_trace(go.Scatter(x=prices, y=quantities, name="Q(p)", line=dict(color="dodgerblue")), row=1, col=1)
        fig.add_trace(go.Scatter(x=[best_p], y=[best_q], mode='markers', name='Optimum',
                                 marker=dict(size=12, color='red', symbol='star')), row=1, col=1)
        fig.add_trace(go.Scatter(x=prices, y=revenues, name="Revenue", line=dict(color="green")), row=1, col=2)
        if use_cost:
            fig.add_trace(go.Scatter(x=prices, y=profits, name="Profit", line=dict(color="orange", dash="dash")), row=1, col=2)
        fig.add_trace(go.Scatter(x=[best_p], y=[best_revenue], mode='markers', name='Opt Revenue',
                                 marker=dict(size=12, color='red', symbol='star')), row=1, col=2)
        fig.add_trace(go.Scatter(x=prices, y=elasticity_arr, name="ε(p)", line=dict(color="purple")), row=1, col=3)
        fig.add_trace(go.Scatter(x=[best_p], y=[elasticity_at_opt], mode='markers', name='ε at p*',
                                 marker=dict(size=12, color='red', symbol='star')), row=1, col=3)
        fig.add_hline(y=-1, line_dash="dot", line_color="gray", annotation_text="Unit elastic (ε = −1)", row=1, col=3)
        if use_capacity:
            fig.add_hline(y=capacity, line_dash="dot", line_color="red", annotation_text="Capacity", row=1, col=1)
        fig.update_xaxes(title_text="Price", row=1, col=1)
        fig.update_xaxes(title_text="Price", row=1, col=2)
        fig.update_xaxes(title_text="Price", row=1, col=3)
        fig.update_yaxes(title_text="Quantity", row=1, col=1)
        fig.update_yaxes(title_text="$", row=1, col=2)
        fig.update_yaxes(title_text="Elasticity", row=1, col=3)
        fig.update_layout(height=450, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Price Sensitivity Table"):
            sample_prices = np.linspace(max(0.01, best_p * 0.5), min(p_max * 0.95, best_p * 1.5), 15)
            rows = []
            for sp in sample_prices:
                sp_arr = np.array([sp])
                sq = float(demand_fn(sp_arr)[0]) if hasattr(demand_fn(sp_arr), '__len__') else float(demand_fn(sp))
                dp_s = sp * 1e-5
                sq_p = float(demand_fn(np.array([sp + dp_s]))[0]) if hasattr(demand_fn(np.array([sp + dp_s])), '__len__') else float(demand_fn(sp + dp_s))
                sq_m = float(demand_fn(np.array([sp - dp_s]))[0]) if hasattr(demand_fn(np.array([sp - dp_s])), '__len__') else float(demand_fn(sp - dp_s))
                el = (sq_p - sq_m) / (2 * dp_s) * sp / sq if sq > 0 else 0
                rows.append({"Price": fmt_dollar(sp), "Quantity": fmt_qty(sq),
                             "Revenue": fmt_dollar(sp * sq), "Profit": fmt_dollar((sp - cost) * sq),
                             "Elasticity ε": f"{el:.3f}"})
            st.table(rows)

    # =================================================================
    # SUB-MODE: Optimal Quantity (inverse demand)
    # =================================================================
    elif solve_for == "Optimal Quantity (inverse demand)":
        st.markdown("""
        **Inverse demand**: choose **quantity** as the decision variable.
        Price is determined by the inverse demand function P(Q).
        """)

        inv_type = st.selectbox("Inverse Demand Model", ["Linear P(Q) = a − b·Q", "Constant Elasticity", "Exponential"])

        col1, col2 = st.columns(2)

        if inv_type == "Linear P(Q) = a − b·Q":
            with col1:
                st.subheader("Parameters")
                st.latex(r"P(Q) = a - b \cdot Q")
                st.latex(r"R(Q) = (a - bQ) \cdot Q")
                st.latex(r"MR(Q) = a - 2bQ")
                a_inv = st.number_input("a (price intercept)", value=50.0, min_value=0.01, step=5.0, key="inv_a")
                b_inv = st.number_input("b (slope)", value=0.5, min_value=0.001, step=0.1, key="inv_b", format="%.3f")
            price_fn = lambda Q: np.maximum(a_inv - b_inv * Q, 0)
            Q_max = a_inv / b_inv
            analytical_Q = a_inv / (2 * b_inv)
            analytical_Q_label = f"Analytical optimum: Q* = a/(2b) = {analytical_Q:,.2f}"

        elif inv_type == "Constant Elasticity":
            with col1:
                st.subheader("Parameters")
                st.latex(r"Q(p) = a \cdot p^{-\varepsilon} \;\Rightarrow\; P(Q) = \left(\frac{Q}{a}\right)^{-1/\varepsilon}")
                a_inv = st.number_input("a (scale)", value=1000.0, min_value=1.0, step=100.0, key="inv_a_ce")
                e_inv = st.number_input("ε (elasticity)", value=2.0, min_value=1.01, step=0.1, key="inv_e_ce")
            def price_fn(Q):
                with np.errstate(divide='ignore', invalid='ignore'):
                    p = np.power(Q / a_inv, -1.0 / e_inv)
                return np.where(np.isfinite(p), p, 0)
            Q_max = a_inv  # practical upper bound
            analytical_Q = None
            analytical_Q_label = ""

        else:  # Exponential
            with col1:
                st.subheader("Parameters")
                st.latex(r"Q(p) = a \cdot e^{-bp} \;\Rightarrow\; P(Q) = -\frac{1}{b}\ln\left(\frac{Q}{a}\right)")
                a_inv = st.number_input("a (scale)", value=100.0, min_value=1.0, step=10.0, key="inv_a_exp")
                b_inv = st.number_input("b (decay rate)", value=0.05, min_value=0.001, step=0.01, key="inv_b_exp", format="%.3f")
            def price_fn(Q):
                with np.errstate(divide='ignore', invalid='ignore'):
                    p = -(1.0 / b_inv) * np.log(Q / a_inv)
                return np.where(np.isfinite(p) & (p > 0), p, 0)
            Q_max = a_inv * 0.99
            analytical_Q = None
            analytical_Q_label = ""

        with col1:
            st.subheader("Constraints (optional)")
            use_cost_q = st.checkbox("Include marginal cost", key="mc_q")
            cost_q = st.number_input("Marginal cost per unit", value=0.0, min_value=0.0, step=1.0, key="c_q") if use_cost_q else 0.0
            use_cap_q = st.checkbox("Capacity constraint", key="cap_q")
            cap_q = st.number_input("Max capacity", value=30.0, min_value=1.0, step=5.0, key="capval_q") if use_cap_q else None

        # Optimize over Q
        def neg_profit_q(Q):
            if Q <= 0:
                return 0
            Q_arr = np.array([Q])
            P = float(price_fn(Q_arr)[0]) if hasattr(price_fn(Q_arr), '__len__') else float(price_fn(Q))
            return -(P - cost_q) * Q

        upper_q = min(Q_max * 0.99, cap_q) if use_cap_q else Q_max * 0.99
        res_q = minimize_scalar(neg_profit_q, bounds=(0.01, upper_q), method='bounded')
        best_Q = res_q.x

        best_P_arr = price_fn(np.array([best_Q]))
        best_P = float(best_P_arr[0]) if hasattr(best_P_arr, '__len__') else float(best_P_arr)
        best_rev_q = best_P * best_Q
        best_profit_q = (best_P - cost_q) * best_Q

        # Marginal revenue at optimum
        dq = best_Q * 1e-5
        rev_plus = float(price_fn(np.array([best_Q + dq]))[0]) * (best_Q + dq)
        rev_minus = float(price_fn(np.array([best_Q - dq]))[0]) * (best_Q - dq)
        mr_at_opt = (rev_plus - rev_minus) / (2 * dq)

        with col2:
            st.subheader("Optimal Solution")
            r1, r2, r3 = st.columns(3)
            r1.metric("Optimal Quantity", fmt_qty2(best_Q))
            r1.caption("The quantity that maximizes revenue. Found where MR = MC (or MR = 0 without costs).")
            r2.metric("Price at Q*", fmt_dollar(best_P))
            r2.caption("The market-clearing price at the optimal quantity, from the inverse demand curve P(Q*).")
            r3.metric("Revenue", fmt_dollar(best_rev_q))
            r3.caption("Total revenue = P(Q*) × Q*. The maximum achievable under the given demand and constraints.")
            if use_cost_q:
                st.metric("Profit", fmt_dollar(best_profit_q))
                st.caption("Revenue minus total variable cost: (P − c) × Q.")
            st.metric("Marginal Revenue at Q*", fmt_dollar(mr_at_opt))
            st.caption("The additional revenue from selling one more unit. At the optimum, MR = MC (or ≈ 0 if no cost).")
            if use_cap_q and best_Q >= cap_q - 0.01:
                st.info(f"Capacity constraint is **binding** at {fmt_int(cap_q)} units.")
            if analytical_Q is not None and not use_cap_q and cost_q == 0:
                st.caption(analytical_Q_label)

        # Python printout
        _code = [
            f"from scipy.optimize import minimize_scalar",
            f"",
            f"# Inverse demand model: {inv_type}",
            f"# Optimize over quantity Q",
            f"neg_revenue = lambda Q: -(price_fn(Q) * Q)",
            f"result = minimize_scalar(neg_revenue, bounds=(0.01, {upper_q:.2f}), method='bounded')",
            f"Q_star = result.x",
            f"P_star = price_fn(Q_star)",
            f"revenue = P_star * Q_star",
            f"print(f'Optimal Quantity: {{Q_star:.2f}}')",
            f"print(f'Price at Q*:     ${{P_star:.2f}}')",
            f"print(f'Revenue:         ${{revenue:.2f}}')",
            f"print(f'Marginal Rev:    ${{MR:.2f}}')",
        ]
        _results = [
            f"Optimal Quantity: {best_Q:.2f}",
            f"Price at Q*:     ${best_P:.2f}",
            f"Revenue:         ${best_rev_q:.2f}",
            f"Marginal Rev:    ${mr_at_opt:.2f}",
        ]
        if use_cost_q:
            _results.append(f"Profit:          ${best_profit_q:.2f}")
        python_output("Optimal Quantity", _code, _results)

        # Charts
        q_arr = np.linspace(0.01, upper_q * 0.99, 500)
        p_arr = price_fn(q_arr)
        rev_arr = p_arr * q_arr
        profit_arr = (p_arr - cost_q) * q_arr

        # MR curve
        dq_arr = q_arr * 1e-5
        rev_p = price_fn(q_arr + dq_arr) * (q_arr + dq_arr)
        rev_m = price_fn(q_arr - dq_arr) * (q_arr - dq_arr)
        mr_arr = (rev_p - rev_m) / (2 * dq_arr)

        fig_q = make_subplots(rows=1, cols=3,
                               subplot_titles=("Inverse Demand P(Q)", "Revenue / Profit vs Q", "Marginal Revenue MR(Q)"))

        fig_q.add_trace(go.Scatter(x=q_arr, y=p_arr, name="P(Q)", line=dict(color="dodgerblue")), row=1, col=1)
        fig_q.add_trace(go.Scatter(x=[best_Q], y=[best_P], mode='markers', name='Optimum',
                                    marker=dict(size=12, color='red', symbol='star')), row=1, col=1)
        if use_cost_q:
            fig_q.add_hline(y=cost_q, line_dash="dot", line_color="orange",
                             annotation_text=f"MC = {fmt_dollar(cost_q)}", row=1, col=1)

        fig_q.add_trace(go.Scatter(x=q_arr, y=rev_arr, name="Revenue", line=dict(color="green")), row=1, col=2)
        if use_cost_q:
            fig_q.add_trace(go.Scatter(x=q_arr, y=profit_arr, name="Profit",
                                        line=dict(color="orange", dash="dash")), row=1, col=2)
        fig_q.add_trace(go.Scatter(x=[best_Q], y=[best_rev_q], mode='markers', name='Opt Revenue',
                                    marker=dict(size=12, color='red', symbol='star')), row=1, col=2)

        fig_q.add_trace(go.Scatter(x=q_arr, y=mr_arr, name="MR(Q)", line=dict(color="purple")), row=1, col=3)
        fig_q.add_trace(go.Scatter(x=[best_Q], y=[mr_at_opt], mode='markers', name='MR at Q*',
                                    marker=dict(size=12, color='red', symbol='star')), row=1, col=3)
        if use_cost_q:
            fig_q.add_hline(y=cost_q, line_dash="dot", line_color="orange",
                             annotation_text=f"MC = {fmt_dollar(cost_q)}", row=1, col=3)
        else:
            fig_q.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="MR = 0", row=1, col=3)

        if use_cap_q:
            fig_q.add_vline(x=cap_q, line_dash="dot", line_color="red",
                             annotation_text="Capacity", row=1, col=2)

        fig_q.update_xaxes(title_text="Quantity", row=1, col=1)
        fig_q.update_xaxes(title_text="Quantity", row=1, col=2)
        fig_q.update_xaxes(title_text="Quantity", row=1, col=3)
        fig_q.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig_q.update_yaxes(title_text="$", row=1, col=2)
        fig_q.update_yaxes(title_text="MR ($)", row=1, col=3)
        fig_q.update_layout(height=450, showlegend=True)
        st.plotly_chart(fig_q, use_container_width=True)

        with st.expander("Quantity Sensitivity Table"):
            sample_q = np.linspace(max(0.1, best_Q * 0.3), min(upper_q * 0.99, best_Q * 1.7), 15)
            rows = []
            for sq in sample_q:
                sp = float(price_fn(np.array([sq]))[0])
                dq_s = sq * 1e-5
                rp = float(price_fn(np.array([sq + dq_s]))[0]) * (sq + dq_s)
                rm = float(price_fn(np.array([sq - dq_s]))[0]) * (sq - dq_s)
                mr_s = (rp - rm) / (2 * dq_s)
                rows.append({"Quantity": fmt_qty(sq), "Price": fmt_dollar(sp),
                             "Revenue": fmt_dollar(sp * sq), "MR": fmt_dollar(mr_s)})
            st.table(rows)

    # =================================================================
    # SUB-MODE: Solve for Price given Q
    # =================================================================
    elif solve_for == "Solve for Price (given Q)":
        st.markdown("**Given a quantity, compute the corresponding price** using the inverse demand function.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Linear Inverse Demand")
            st.latex(r"P(Q) = a - b \cdot Q")
            a_val = st.number_input("a (price intercept)", value=50.0, min_value=0.01, step=5.0, key="sfp_a")
            b_val = st.number_input("b (slope)", value=0.5, min_value=0.001, step=0.1, key="sfp_b", format="%.3f")
            Q_input = st.number_input("Quantity (Q)", value=25.0, min_value=0.0, step=5.0, key="sfp_Q")

        P_result = max(a_val - b_val * Q_input, 0)
        R_result = P_result * Q_input
        MR_result = a_val - 2 * b_val * Q_input

        with col2:
            st.subheader("Results")
            st.metric("Price P(Q)", fmt_dollar(P_result))
            st.caption("The price the market will bear at this quantity, from the inverse demand function P(Q) = a − b·Q.")
            st.metric("Revenue R = P·Q", fmt_dollar(R_result))
            st.caption("Total revenue earned by selling Q units at the market-clearing price.")
            st.metric("Marginal Revenue MR(Q) = a − 2bQ", fmt_dollar(MR_result))
            st.caption("Additional revenue from one more unit. MR declines twice as fast as price because selling more also lowers the price on all units.")
            if P_result <= 0:
                st.warning("Price is zero — quantity exceeds maximum demand.")

        # Python printout
        python_output("Solve for Price", [
            f"a, b = {a_val}, {b_val}",
            f"Q = {Q_input}",
            f"P = max(a - b * Q, 0)",
            f"R = P * Q",
            f"MR = a - 2 * b * Q",
            f"print(f'Price P(Q):        ${{P:.2f}}')",
            f"print(f'Revenue R = P*Q:   ${{R:.2f}}')",
            f"print(f'Marginal Revenue:  ${{MR:.2f}}')",
        ], [
            f"Price P(Q):        ${P_result:.2f}",
            f"Revenue R = P*Q:   ${R_result:.2f}",
            f"Marginal Revenue:  ${MR_result:.2f}",
        ])

        # Quick chart
        q_arr = np.linspace(0, a_val / b_val, 300)
        p_arr = np.maximum(a_val - b_val * q_arr, 0)
        mr_arr_c = a_val - 2 * b_val * q_arr

        fig_sfp = make_subplots(rows=1, cols=2, subplot_titles=("Inverse Demand & MR", "Revenue Curve"))
        fig_sfp.add_trace(go.Scatter(x=q_arr, y=p_arr, name="P(Q)", line=dict(color="dodgerblue")), row=1, col=1)
        fig_sfp.add_trace(go.Scatter(x=q_arr, y=mr_arr_c, name="MR(Q)", line=dict(color="purple", dash="dash")), row=1, col=1)
        fig_sfp.add_trace(go.Scatter(x=[Q_input], y=[P_result], mode='markers', name='Your Q',
                                      marker=dict(size=14, color='red', symbol='star')), row=1, col=1)
        fig_sfp.add_trace(go.Scatter(x=q_arr, y=p_arr * q_arr, name="Revenue", line=dict(color="green")), row=1, col=2)
        fig_sfp.add_trace(go.Scatter(x=[Q_input], y=[R_result], mode='markers', name='Your Q',
                                      marker=dict(size=14, color='red', symbol='star')), row=1, col=2)
        fig_sfp.update_xaxes(title_text="Quantity")
        fig_sfp.update_yaxes(title_text="$", row=1, col=1)
        fig_sfp.update_yaxes(title_text="Revenue ($)", row=1, col=2)
        fig_sfp.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_sfp, use_container_width=True)

    # =================================================================
    # SUB-MODE: Solve for Quantity given P
    # =================================================================
    elif solve_for == "Solve for Quantity (given P)":
        st.markdown("**Given a price, compute the corresponding quantity** using the demand function.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Demand Model")
            dm = st.selectbox("Model", ["Linear", "Exponential", "Constant Elasticity"], key="sfq_model")

            if dm == "Linear":
                st.latex(r"Q(p) = D - m \cdot p")
                D_val = st.number_input("D (intercept)", value=100.0, min_value=1.0, step=10.0, key="sfq_D")
                m_val = st.number_input("m (slope)", value=2.0, min_value=0.01, step=0.5, key="sfq_m")
                P_input = st.number_input("Price (P)", value=25.0, min_value=0.0, step=5.0, key="sfq_P")
                Q_result = max(D_val - m_val * P_input, 0)
                elast = -m_val * P_input / Q_result if Q_result > 0 else 0

            elif dm == "Exponential":
                st.latex(r"Q(p) = a \cdot e^{-bp}")
                a_val = st.number_input("a (scale)", value=100.0, min_value=1.0, step=10.0, key="sfq_a")
                b_val = st.number_input("b (decay)", value=0.05, min_value=0.001, step=0.01, key="sfq_b", format="%.3f")
                P_input = st.number_input("Price (P)", value=20.0, min_value=0.0, step=5.0, key="sfq_P")
                Q_result = a_val * np.exp(-b_val * P_input)
                elast = -b_val * P_input

            else:  # Constant Elasticity
                st.latex(r"Q(p) = a \cdot p^{-\varepsilon}")
                a_val = st.number_input("a (scale)", value=1000.0, min_value=1.0, step=100.0, key="sfq_a")
                e_val = st.number_input("ε (elasticity)", value=2.0, min_value=0.01, step=0.1, key="sfq_e")
                P_input = st.number_input("Price (P)", value=10.0, min_value=0.01, step=5.0, key="sfq_P")
                Q_result = a_val * P_input ** (-e_val)
                elast = -e_val

        R_result = P_input * Q_result

        with col2:
            st.subheader("Results")
            st.metric("Quantity Q(P)", fmt_qty2(Q_result))
            st.caption("The number of units demanded at the given price, from the demand function Q(P).")
            st.metric("Revenue R = P·Q", fmt_dollar(R_result))
            st.caption("Total revenue at this price point: Price × Quantity demanded.")
            st.metric("Elasticity ε at P", f"{elast:.3f}")
            st.caption("Price elasticity of demand. |ε| > 1 means elastic (price ↑ → revenue ↓); |ε| < 1 means inelastic (price ↑ → revenue ↑).")

        python_output("Solve for Quantity", [
            f"# Demand model: {dm}",
            f"P = {P_input}",
            f"Q = demand(P)  # = {Q_result:.2f}",
            f"R = P * Q",
            f"elasticity = {elast:.3f}",
            f"print(f'Quantity Q(P):  {{Q:.2f}}')",
            f"print(f'Revenue R=P*Q: ${{R:.2f}}')",
            f"print(f'Elasticity:    {{elasticity:.3f}}')",
        ], [
            f"Quantity Q(P):  {Q_result:.2f}",
            f"Revenue R=P*Q: ${R_result:.2f}",
            f"Elasticity:    {elast:.3f}",
        ])

    # =================================================================
    # SUB-MODE: Solve for Revenue
    # =================================================================
    elif solve_for == "Solve for Revenue (given P or Q)":
        st.markdown("**Compute revenue** given either price or quantity with a linear demand model.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Linear Demand: Q = D − m·p")
            D_val = st.number_input("D (intercept)", value=100.0, min_value=1.0, step=10.0, key="sfr_D")
            m_val = st.number_input("m (slope)", value=2.0, min_value=0.01, step=0.5, key="sfr_m")

            input_mode = st.radio("Input", ["Given Price", "Given Quantity"], key="sfr_mode", horizontal=True)
            if input_mode == "Given Price":
                P_in = st.number_input("Price", value=25.0, min_value=0.0, step=5.0, key="sfr_P")
                Q_calc = max(D_val - m_val * P_in, 0)
                P_calc = P_in
            else:
                Q_in = st.number_input("Quantity", value=50.0, min_value=0.0, step=5.0, key="sfr_Q")
                P_calc = max((D_val - Q_in) / m_val, 0)
                Q_calc = Q_in

        R_calc = P_calc * Q_calc
        MR_calc = (D_val - 2 * Q_calc) / m_val if m_val > 0 else 0

        with col2:
            st.subheader("Results")
            st.metric("Price", fmt_dollar(P_calc))
            st.caption("The price corresponding to the given input (entered directly, or derived from quantity via the inverse demand).")
            st.metric("Quantity", fmt_qty2(Q_calc))
            st.caption("The quantity corresponding to the given input (entered directly, or derived from price via the demand function).")
            st.metric("Revenue (P × Q)", fmt_dollar(R_calc))
            st.caption("Total revenue at this price–quantity combination.")
            st.metric("Marginal Revenue at Q", fmt_dollar(MR_calc))
            st.caption("The revenue gained from selling one additional unit at this quantity level.")

        python_output("Revenue Calculation", [
            f"D, m = {D_val}, {m_val}",
            f"P = {P_calc:.2f}",
            f"Q = {Q_calc:.2f}",
            f"R = P * Q",
            f"MR = (D - 2*Q) / m",
            f"print(f'Price:            ${{P:.2f}}')",
            f"print(f'Quantity:          {{Q:.2f}}')",
            f"print(f'Revenue (P x Q):  ${{R:.2f}}')",
            f"print(f'Marginal Revenue: ${{MR:.2f}}')",
        ], [
            f"Price:            ${P_calc:.2f}",
            f"Quantity:          {Q_calc:.2f}",
            f"Revenue (P x Q):  ${R_calc:.2f}",
            f"Marginal Revenue: ${MR_calc:.2f}",
        ])

        # Revenue as function of both P and Q for reference
        p_range = np.linspace(0, D_val / m_val, 300)
        q_range = np.maximum(D_val - m_val * p_range, 0)
        r_range = p_range * q_range

        fig_sfr = go.Figure()
        fig_sfr.add_trace(go.Scatter(x=p_range, y=r_range, name="R(p)", line=dict(color="green")))
        fig_sfr.add_trace(go.Scatter(x=[P_calc], y=[R_calc], mode='markers', name='Your Point',
                                      marker=dict(size=14, color='red', symbol='star')))
        fig_sfr.update_layout(title="Revenue Curve R(p)", xaxis_title="Price", yaxis_title="Revenue ($)", height=350)
        st.plotly_chart(fig_sfr, use_container_width=True)

    # =================================================================
    # SUB-MODE: Solve for Elasticity
    # =================================================================
    elif solve_for == "Solve for Elasticity (given P)":
        st.markdown("**Compute price elasticity of demand** at a given price point.")
        st.latex(r"\varepsilon(p) = \frac{dQ}{dp} \cdot \frac{p}{Q(p)}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Demand Model")
            dm = st.selectbox("Model", ["Linear", "Exponential", "Constant Elasticity", "Logit"], key="sfe_model")

            if dm == "Linear":
                st.latex(r"Q = D - mp \;\;\Rightarrow\;\; \varepsilon = \frac{-mp}{D - mp}")
                D_val = st.number_input("D", value=100.0, min_value=1.0, step=10.0, key="sfe_D")
                m_val = st.number_input("m", value=2.0, min_value=0.01, step=0.5, key="sfe_m")
                P_in = st.number_input("Price", value=25.0, min_value=0.01, step=5.0, key="sfe_P")
                Q_at_p = max(D_val - m_val * P_in, 0)
                elast = -m_val * P_in / Q_at_p if Q_at_p > 0 else float('-inf')
                elast_formula = f"ε = −m·p / (D − m·p) = −{m_val}·{P_in} / {Q_at_p:,.1f} = {elast:.4f}"

            elif dm == "Exponential":
                st.latex(r"Q = ae^{-bp} \;\;\Rightarrow\;\; \varepsilon = -bp")
                a_val = st.number_input("a", value=100.0, min_value=1.0, step=10.0, key="sfe_a")
                b_val = st.number_input("b", value=0.05, min_value=0.001, step=0.01, key="sfe_b", format="%.3f")
                P_in = st.number_input("Price", value=20.0, min_value=0.01, step=5.0, key="sfe_P")
                Q_at_p = a_val * np.exp(-b_val * P_in)
                elast = -b_val * P_in
                elast_formula = f"ε = −b·p = −{b_val}·{P_in} = {elast:.4f}"

            elif dm == "Constant Elasticity":
                st.latex(r"Q = ap^{-\varepsilon} \;\;\Rightarrow\;\; \varepsilon = -\varepsilon \text{ (constant!)}")
                a_val = st.number_input("a", value=1000.0, min_value=1.0, step=100.0, key="sfe_a")
                e_val = st.number_input("ε parameter", value=2.0, min_value=0.01, step=0.1, key="sfe_e")
                P_in = st.number_input("Price", value=10.0, min_value=0.01, step=5.0, key="sfe_P")
                Q_at_p = a_val * P_in ** (-e_val)
                elast = -e_val
                elast_formula = f"ε = −{e_val} (constant for all prices)"

            else:  # Logit
                st.latex(r"\varepsilon = \beta_p \cdot p \cdot (1 - s(p))")
                M_val = st.number_input("M (market size)", value=1000.0, min_value=1.0, step=100.0, key="sfe_M")
                b0_val = st.number_input("β₀", value=3.0, step=0.5, key="sfe_b0")
                bp_val = st.number_input("βₚ (negative)", value=-0.1, max_value=-0.001, step=0.01, key="sfe_bp", format="%.3f")
                P_in = st.number_input("Price", value=20.0, min_value=0.01, step=5.0, key="sfe_P")
                v = b0_val + bp_val * P_in
                share = np.exp(v) / (1 + np.exp(v))
                Q_at_p = M_val * share
                elast = bp_val * P_in * (1 - share)
                elast_formula = f"ε = βₚ·p·(1−s) = {bp_val}·{P_in}·{1 - share:.4f} = {elast:.4f}"

        R_at_p = P_in * Q_at_p

        with col2:
            st.subheader("Results")
            st.metric("Quantity at P", fmt_qty2(Q_at_p))
            st.caption("Units demanded at the specified price.")
            st.metric("Revenue at P", fmt_dollar(R_at_p))
            st.caption("Total revenue if you charge this price: P × Q(P).")
            st.metric("Price Elasticity ε", f"{elast:.4f}")
            st.caption("The % change in quantity for a 1% price increase. Negative means demand falls as price rises (the normal case).")
            st.code(elast_formula)
            if abs(elast) > 1:
                st.caption("Demand is **elastic** — raising price will decrease revenue.")
            elif abs(elast) < 1:
                st.caption("Demand is **inelastic** — raising price will increase revenue.")
            else:
                st.caption("Demand is **unit elastic** — revenue is at its maximum.")

        python_output("Elasticity Calculation", [
            f"# Demand model: {dm}",
            f"P = {P_in}",
            f"Q = demand(P)  # = {Q_at_p:.2f}",
            f"R = P * Q",
            f"# {elast_formula}",
            f"print(f'Quantity at P:  {{Q:.2f}}')",
            f"print(f'Revenue at P:  ${{R:.2f}}')",
            f"print(f'Elasticity e:  {{e:.4f}}')",
        ], [
            f"Quantity at P:  {Q_at_p:.2f}",
            f"Revenue at P:  ${R_at_p:.2f}",
            f"Elasticity e:  {elast:.4f}",
        ])

        # Elasticity over price range
        p_range = np.linspace(0.01, P_in * 3, 500)
        if dm == "Linear":
            q_range = np.maximum(D_val - m_val * p_range, 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                e_range = -m_val * p_range / q_range
            e_range = np.where(np.isfinite(e_range), e_range, 0)
        elif dm == "Exponential":
            e_range = -b_val * p_range
        elif dm == "Constant Elasticity":
            e_range = np.full_like(p_range, -e_val)
        else:
            v_range = b0_val + bp_val * p_range
            s_range = np.exp(v_range) / (1 + np.exp(v_range))
            e_range = bp_val * p_range * (1 - s_range)

        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(x=p_range, y=e_range, name="ε(p)", line=dict(color="purple")))
        fig_e.add_trace(go.Scatter(x=[P_in], y=[elast], mode='markers', name='Your Price',
                                    marker=dict(size=14, color='red', symbol='star')))
        fig_e.add_hline(y=-1, line_dash="dot", line_color="gray", annotation_text="Unit elastic (ε = −1)")
        fig_e.update_layout(title="Elasticity ε(p)", xaxis_title="Price", yaxis_title="Elasticity", height=350)
        st.plotly_chart(fig_e, use_container_width=True)


# =====================================================================
# MODE 2 – Multi-segment
# =====================================================================
elif mode == "Multi-Segment Optimization":
    st.header("Multi-Segment Price Optimization")
    st.markdown("Optimize prices across multiple customer segments sharing a common capacity pool.")

    preset_ms = st.selectbox("Load example", [
        "Custom",
        "Vertigo Nightclub (Members vs General)"
    ], key="preset_ms")

    if preset_ms == "Vertigo Nightclub (Members vs General)":
        _num_seg_default = 2
        _seg_defaults = [
            {"name": "Members", "D": 10000.0, "m": 100.0},
            {"name": "General", "D": 40000.0, "m": 200.0},
        ]
        _cap_default = 20000.0
        _ic_default = True
    else:
        _seg_defaults = None
        _ic_default = False
        _cap_default = 30.0

    num_segments = st.slider("Number of segments", 2, 5,
                             2 if _seg_defaults else 2)

    segments = []
    seg_names = []
    cols = st.columns(num_segments)
    for i, c in enumerate(cols):
        with c:
            if _seg_defaults and i < len(_seg_defaults):
                _sd = _seg_defaults[i]
                st.subheader(_sd["name"])
                D = st.number_input(f"D{i+1} (intercept)", value=_sd["D"], min_value=1.0, key=f"D{i}")
                m = st.number_input(f"m{i+1} (slope)", value=_sd["m"], min_value=0.01, key=f"m{i}", format="%.2f")
                seg_names.append(_sd["name"])
            else:
                st.subheader(f"Segment {i+1}")
                D = st.number_input(f"D{i+1} (intercept)", value=max(60.0 - i * 20, 10.0), min_value=1.0, key=f"D{i}")
                m = st.number_input(f"m{i+1} (slope)", value=max(1.6 - i * 0.4, 0.2), min_value=0.01, key=f"m{i}", format="%.2f")
                seg_names.append(f"Segment {i+1}")
            segments.append((D, m))

    st.subheader("Shared Constraint")
    use_shared_cap = st.checkbox("Shared capacity constraint", value=True)
    shared_cap = st.number_input("Total capacity", value=_cap_default, min_value=1.0, step=5.0) if use_shared_cap else None
    use_ic_price = st.checkbox("Incentive-compatible price constraint (P₁ ≤ P₂ ≤ ... ≤ Pₙ)",
                               value=_ic_default,
                               help="Ensures prices are ordered so lower-tier customers don't switch to higher-tier products.")
    use_cost_multi = st.checkbox("Include marginal cost", key="mc_multi")
    cost_multi = st.number_input("Marginal cost", value=0.0, min_value=0.0, step=1.0, key="c_multi") if use_cost_multi else 0.0

    # Optimize
    n = len(segments)
    def neg_total_profit(prices):
        total = 0
        for idx, (D, m) in enumerate(segments):
            q = max(D - m * prices[idx], 0)
            total += (prices[idx] - cost_multi) * q
        return -total

    bounds = [(0.01, seg[0] / seg[1]) for seg in segments]
    x0 = [seg[0] / (2 * seg[1]) for seg in segments]

    constraints = []
    if use_shared_cap:
        def cap_con(prices):
            return shared_cap - sum(max(seg[0] - seg[1] * prices[idx], 0) for idx, seg in enumerate(segments))
        constraints.append({'type': 'ineq', 'fun': cap_con})

    if use_ic_price and n >= 2:
        # IC: P_0 <= P_1 <= ... <= P_{n-1}
        for j in range(n - 1):
            def ic_con(prices, j=j):
                return prices[j + 1] - prices[j]
            constraints.append({'type': 'ineq', 'fun': ic_con})

    res = minimize(neg_total_profit, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_prices = res.x

    # Display results
    st.subheader("Optimal Solution")
    result_cols = st.columns(num_segments + 1)
    total_rev = 0
    total_q = 0
    for i, (D, m) in enumerate(segments):
        q = max(D - m * opt_prices[i], 0)
        rev = opt_prices[i] * q
        total_rev += rev
        total_q += q
        sname = seg_names[i] if i < len(seg_names) else f"Segment {i+1}"
        with result_cols[i]:
            st.metric(f"{sname} Price", fmt_dollar(opt_prices[i]))
            st.caption("Optimal price for this segment — higher willingness-to-pay segments get higher prices.")
            st.metric(f"{sname} Qty", fmt_qty(q))
            st.caption("Units allocated to this segment at its optimal price.")
            st.metric(f"{sname} Revenue", fmt_dollar(rev))
            st.caption("Revenue earned from this segment: Price × Quantity.")

    with result_cols[-1]:
        st.metric("Total Revenue", fmt_dollar(total_rev))
        st.caption("Combined revenue across all segments — the objective being maximized.")
        st.metric("Total Quantity", fmt_qty(total_q))
        st.caption("Total units sold across all segments.")
        if use_cost_multi:
            st.metric("Total Profit", fmt_dollar(-neg_total_profit(opt_prices)))
            st.caption("Total revenue minus total variable cost across all segments.")
        if use_shared_cap and total_q >= shared_cap - 0.01:
            st.info("Capacity constraint is **binding**.")
        if use_ic_price and n >= 2:
            ic_binding = []
            for j in range(n - 1):
                if abs(opt_prices[j+1] - opt_prices[j]) < 0.01:
                    ic_binding.append(f"{seg_names[j]}/{seg_names[j+1]}")
            if ic_binding:
                st.warning(f"IC price constraint is **binding** between: {', '.join(ic_binding)}")
            else:
                st.success("IC price constraint satisfied with slack (prices naturally ordered).")

    # Shadow price (Lagrange multiplier approximation)
    if use_shared_cap and total_q >= shared_cap - 0.5:
        eps = 0.1
        def solve_with_cap(cap):
            def cap_con2(prices):
                return cap - sum(max(seg[0] - seg[1] * prices[idx], 0) for idx, seg in enumerate(segments))
            r = minimize(neg_total_profit, x0, method='SLSQP', bounds=bounds,
                         constraints=[{'type': 'ineq', 'fun': cap_con2}])
            return -r.fun
        shadow = (solve_with_cap(shared_cap + eps) - solve_with_cap(shared_cap)) / eps
        st.metric("Shadow Price of Capacity (≈ Lagrange multiplier)", fmt_dollar(shadow))
        st.caption("The extra revenue you'd earn from one more unit of capacity. This is the Lagrange multiplier on the capacity constraint — it tells you the maximum you should pay to add capacity.")

    # Python printout
    _code_ms = [
        f"from scipy.optimize import minimize",
        f"import numpy as np",
        f"",
        f"segments = {[(D, m) for D, m in segments]}",
        f"cost = {cost_multi}",
    ]
    if use_shared_cap:
        _code_ms.append(f"capacity = {shared_cap}")
    _code_ms += [
        f"",
        f"def neg_profit(prices):",
        f"    return -sum((p - cost) * max(D - m*p, 0) for p, (D, m) in zip(prices, segments))",
        f"",
        f"result = minimize(neg_profit, x0, method='SLSQP', bounds=bounds, constraints=constraints)",
        f"opt_prices = result.x",
    ]
    _results_ms = [f"{'='*50}", f"Multi-Segment Optimal Solution", f"{'='*50}"]
    for i, (D, m) in enumerate(segments):
        q = max(D - m * opt_prices[i], 0)
        rev = opt_prices[i] * q
        sname = seg_names[i] if i < len(seg_names) else f"Segment {i+1}"
        _results_ms.append(f"{sname}: Price = ${opt_prices[i]:.2f}, Qty = {q:.1f}, Revenue = ${rev:.2f}")
    _results_ms.append(f"{'─'*50}")
    _results_ms.append(f"Total Revenue:  ${total_rev:.2f}")
    _results_ms.append(f"Total Quantity: {total_q:.1f}")
    if use_ic_price and n >= 2:
        _results_ms.append(f"IC Price Constraint: {'ACTIVE' if any(abs(opt_prices[j+1] - opt_prices[j]) < 0.01 for j in range(n-1)) else 'Satisfied with slack'}")
    if use_shared_cap and total_q >= shared_cap - 0.5:
        _results_ms.append(f"Shadow Price:   ${shadow:.2f}")
    python_output("Multi-Segment Optimization", _code_ms, _results_ms)

    # Chart
    fig = go.Figure()
    for i, (D, m) in enumerate(segments):
        sname = seg_names[i] if i < len(seg_names) else f"Seg {i+1}"
        pp = np.linspace(0.01, D / m * 0.95, 300)
        qq = np.maximum(D - m * pp, 0)
        rr = pp * qq
        fig.add_trace(go.Scatter(x=pp, y=rr, name=f"{sname} Revenue"))
        qi = max(D - m * opt_prices[i], 0)
        fig.add_trace(go.Scatter(x=[opt_prices[i]], y=[opt_prices[i] * qi],
                                 mode='markers', name=f"{sname} Opt",
                                 marker=dict(size=12, symbol='star')))
    fig.update_layout(title="Revenue Curves by Segment", xaxis_title="Price", yaxis_title="Revenue", height=400)
    st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# MODE 3 – Dynamic Pricing with Inventory (Multi-Segment per Period)
# =====================================================================
elif mode == "Dynamic Pricing with Inventory":
    st.header("Dynamic Pricing with Inventory")
    st.markdown("""
    Set prices across **multiple time periods** and **multiple customer segments** with a shared
    inventory pool. Total sales across all period–segment combinations cannot exceed inventory.
    """)

    col_set1, col_set2 = st.columns(2)
    with col_set1:
        num_periods = st.slider("Number of periods", 2, 4, 2)
    with col_set2:
        num_seg_dp = st.slider("Segments per period", 1, 4, 2)

    st.subheader("Demand Parameters (Linear: Q = D − m·p)")

    # Defaults inspired by course East/West example
    default_D = [[60, 40, 50, 30], [80, 50, 60, 40], [70, 45, 55, 35], [90, 60, 70, 50]]
    default_m = [[1.6, 0.4, 1.0, 0.5], [2.0, 0.8, 1.2, 0.6], [1.8, 0.6, 1.1, 0.5], [2.2, 1.0, 1.4, 0.7]]
    seg_names = ["A", "B", "C", "D"]

    # period_seg_params[t][s] = (D, m)
    period_seg_params = []
    for t in range(num_periods):
        st.markdown(f"**Period {t+1}**")
        seg_cols = st.columns(num_seg_dp)
        seg_list = []
        for s, sc in enumerate(seg_cols):
            with sc:
                st.markdown(f"*Seg {seg_names[s]}*")
                Dts = st.number_input(f"D (P{t+1},S{seg_names[s]})",
                                       value=float(default_D[t % 4][s % 4]),
                                       min_value=1.0, step=5.0, key=f"dp_D_{t}_{s}")
                mts = st.number_input(f"m (P{t+1},S{seg_names[s]})",
                                       value=float(default_m[t % 4][s % 4]),
                                       min_value=0.01, step=0.1, key=f"dp_m_{t}_{s}", format="%.2f")
                seg_list.append((Dts, mts))
        period_seg_params.append(seg_list)

    total_inv = st.number_input("Total inventory (shared across all periods & segments)",
                                 value=30.0, min_value=1.0, step=5.0, key="dp_inv")

    # Build flat list of decision variables: price for each (period, segment)
    n_vars = num_periods * num_seg_dp

    def flat_idx(t, s):
        return t * num_seg_dp + s

    def neg_total_rev_dp(prices):
        total = 0
        for t in range(num_periods):
            for s in range(num_seg_dp):
                D_ts, m_ts = period_seg_params[t][s]
                p = prices[flat_idx(t, s)]
                q = max(D_ts - m_ts * p, 0)
                total += p * q
        return -total

    bounds_dp = []
    x0_dp = []
    for t in range(num_periods):
        for s in range(num_seg_dp):
            D_ts, m_ts = period_seg_params[t][s]
            bounds_dp.append((0.01, D_ts / m_ts))
            x0_dp.append(D_ts / (2 * m_ts))

    # Total inventory constraint
    def total_inv_con(prices):
        total_q = 0
        for t in range(num_periods):
            for s in range(num_seg_dp):
                D_ts, m_ts = period_seg_params[t][s]
                total_q += max(D_ts - m_ts * prices[flat_idx(t, s)], 0)
        return total_inv - total_q
    constraints_dp = [{'type': 'ineq', 'fun': total_inv_con}]

    res_dp = minimize(neg_total_rev_dp, x0_dp, method='SLSQP', bounds=bounds_dp, constraints=constraints_dp)
    opt_dp = res_dp.x

    # Results table
    st.subheader("Optimal Solution")
    total_rev_dp = 0
    total_q_dp = 0

    for t in range(num_periods):
        st.markdown(f"**Period {t+1}**")
        rcols = st.columns(num_seg_dp + 1)
        period_rev = 0
        period_q = 0
        for s in range(num_seg_dp):
            D_ts, m_ts = period_seg_params[t][s]
            p = opt_dp[flat_idx(t, s)]
            q = max(D_ts - m_ts * p, 0)
            rev = p * q
            period_rev += rev
            period_q += q
            total_rev_dp += rev
            total_q_dp += q
            with rcols[s]:
                st.metric(f"Seg {seg_names[s]} Price", fmt_dollar(p))
                st.metric(f"Seg {seg_names[s]} Qty", fmt_qty(q))
                st.metric(f"Seg {seg_names[s]} Rev", fmt_dollar(rev))
        with rcols[-1]:
            st.metric(f"Period {t+1} Total Rev", fmt_dollar(period_rev))
            st.caption("Sum of revenue across all segments in this period.")
            st.metric(f"Period {t+1} Total Qty", fmt_qty(period_q))
            st.caption("Total units sold in this period, drawn from the shared inventory pool.")

    st.divider()
    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Grand Total Revenue", fmt_dollar(total_rev_dp))
    tc1.caption("Total revenue across all periods and segments — the objective being maximized.")
    tc2.metric("Total Qty Sold", fmt_qty(total_q_dp))
    tc2.caption("Total units sold across all periods and segments.")
    tc3.metric("Remaining Inventory", fmt_qty(max(total_inv - total_q_dp, 0)))
    tc3.caption("Unsold inventory after all periods. Zero means the constraint is binding.")
    if total_q_dp >= total_inv - 0.01:
        st.info("Inventory constraint is **binding**.")

    # Shadow price of inventory
    if total_q_dp >= total_inv - 0.5:
        eps_dp = 0.1
        def solve_dp_cap(cap):
            def tc(prices):
                tq = 0
                for t in range(num_periods):
                    for s in range(num_seg_dp):
                        D_ts, m_ts = period_seg_params[t][s]
                        tq += max(D_ts - m_ts * prices[flat_idx(t, s)], 0)
                return cap - tq
            r = minimize(neg_total_rev_dp, x0_dp, method='SLSQP', bounds=bounds_dp,
                         constraints=[{'type': 'ineq', 'fun': tc}])
            return -r.fun
        shadow_dp = (solve_dp_cap(total_inv + eps_dp) - solve_dp_cap(total_inv)) / eps_dp
        st.metric("Shadow Price of Inventory", fmt_dollar(shadow_dp))
        st.caption("The extra revenue from one more unit of inventory. This is what you should be willing to pay (at most) to acquire an additional unit.")

    # Python printout
    _results_dp = [f"{'='*55}", f"Dynamic Pricing — Optimal Solution", f"{'='*55}"]
    for t in range(num_periods):
        _results_dp.append(f"")
        _results_dp.append(f"Period {t+1}:")
        for s in range(num_seg_dp):
            D_ts, m_ts = period_seg_params[t][s]
            p = opt_dp[flat_idx(t, s)]
            q = max(D_ts - m_ts * p, 0)
            rev = p * q
            _results_dp.append(f"  Seg {seg_names[s]}: Price = ${p:.2f}, Qty = {q:.1f}, Rev = ${rev:.2f}")
    _results_dp.append(f"{'─'*55}")
    _results_dp.append(f"Grand Total Revenue:  ${total_rev_dp:.2f}")
    _results_dp.append(f"Total Qty Sold:       {total_q_dp:.1f}")
    _results_dp.append(f"Remaining Inventory:  {max(total_inv - total_q_dp, 0):.1f}")
    python_output("Dynamic Pricing", [
        f"from scipy.optimize import minimize",
        f"",
        f"inventory = {total_inv}",
        f"# {num_periods} periods x {num_seg_dp} segments",
        f"# Linear demand: Q = D - m*p for each (period, segment)",
        f"",
        f"result = minimize(neg_total_rev, x0, method='SLSQP',",
        f"                  bounds=bounds, constraints=inventory_constraint)",
        f"opt_prices = result.x",
    ], _results_dp)

    # Charts
    fig_dp = make_subplots(rows=1, cols=2,
                            subplot_titles=("Revenue Curves (by Period–Segment)", "Inventory Waterfall"))

    colors_dp = ["dodgerblue", "orange", "green", "red", "purple", "brown", "pink", "gray",
                  "olive", "cyan", "magenta", "lime", "teal", "navy", "coral", "gold"]
    ci = 0
    for t in range(num_periods):
        for s in range(num_seg_dp):
            D_ts, m_ts = period_seg_params[t][s]
            pp = np.linspace(0.01, D_ts / m_ts * 0.95, 200)
            rr = pp * np.maximum(D_ts - m_ts * pp, 0)
            label = f"P{t+1}-S{seg_names[s]}"
            fig_dp.add_trace(go.Scatter(x=pp, y=rr, name=label,
                                         line=dict(color=colors_dp[ci % len(colors_dp)])), row=1, col=1)
            p_opt = opt_dp[flat_idx(t, s)]
            q_opt = max(D_ts - m_ts * p_opt, 0)
            fig_dp.add_trace(go.Scatter(x=[p_opt], y=[p_opt * q_opt], mode='markers',
                                         name=f"{label} Opt", showlegend=False,
                                         marker=dict(size=10, color=colors_dp[ci % len(colors_dp)],
                                                     symbol='star')), row=1, col=1)
            ci += 1

    # Inventory waterfall by period
    cum_q = 0
    wf_x = ["Start"]
    wf_y = [total_inv]
    for t in range(num_periods):
        pq = 0
        for s in range(num_seg_dp):
            D_ts, m_ts = period_seg_params[t][s]
            pq += max(D_ts - m_ts * opt_dp[flat_idx(t, s)], 0)
        cum_q += pq
        wf_x.append(f"After P{t+1}")
        wf_y.append(max(total_inv - cum_q, 0))
    fig_dp.add_trace(go.Bar(x=wf_x, y=wf_y, name="Remaining Inv", marker_color="steelblue"), row=1, col=2)

    fig_dp.update_xaxes(title_text="Price", row=1, col=1)
    fig_dp.update_yaxes(title_text="Revenue", row=1, col=1)
    fig_dp.update_yaxes(title_text="Units", row=1, col=2)
    fig_dp.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig_dp, use_container_width=True)

    # --- Two-Period Sequential Analysis (single-segment only for clarity) ---
    if num_periods == 2 and num_seg_dp == 1:
        with st.expander("Two-Period Sequential Analysis"):
            st.markdown("""
            **How does the Period 2 optimal price change as Period 1 price varies?**
            Shows the sequential decision structure when there is one segment per period.
            """)

            D1, m1 = period_seg_params[0][0]
            D2, m2 = period_seg_params[1][0]

            p1_range = np.linspace(0.01, D1 / m1 * 0.95, 200)
            opt_p2_seq = []
            rev1_seq = []
            rev2_seq = []
            total_rev_seq = []

            for p1 in p1_range:
                q1 = max(D1 - m1 * p1, 0)
                remaining_inv = max(total_inv - q1, 0)
                r1 = p1 * q1

                if remaining_inv <= 0:
                    opt_p2_seq.append(D2 / m2)
                    rev2_seq.append(0)
                else:
                    p2_unc = D2 / (2 * m2)
                    q2_unc = max(D2 - m2 * p2_unc, 0)
                    if q2_unc <= remaining_inv:
                        best_p2 = p2_unc
                    else:
                        best_p2 = (D2 - remaining_inv) / m2
                    q2_opt = min(max(D2 - m2 * best_p2, 0), remaining_inv)
                    opt_p2_seq.append(best_p2)
                    rev2_seq.append(best_p2 * q2_opt)

                rev1_seq.append(r1)
                total_rev_seq.append(r1 + rev2_seq[-1])

            opt_p2_seq = np.array(opt_p2_seq)
            total_rev_seq = np.array(total_rev_seq)

            best_idx = np.argmax(total_rev_seq)
            best_p1_seq = p1_range[best_idx]
            best_p2_val = opt_p2_seq[best_idx]

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Optimal P1", fmt_dollar(best_p1_seq))
            sc1.caption("Best Period 1 price considering its impact on inventory available for Period 2.")
            sc2.metric("Optimal P2", fmt_dollar(best_p2_val))
            sc2.caption("Best Period 2 price given the remaining inventory after Period 1 sales.")
            sc3.metric("Total Revenue", fmt_dollar(total_rev_seq[best_idx]))
            sc3.caption("Combined revenue from both periods at the optimal sequential pricing strategy.")

            fig_seq = make_subplots(rows=1, cols=2,
                                     subplot_titles=("Optimal P2 vs P1 Price", "Revenue Breakdown"))
            fig_seq.add_trace(go.Scatter(x=p1_range, y=opt_p2_seq, name="Optimal P2",
                                          line=dict(color="orange")), row=1, col=1)
            fig_seq.add_trace(go.Scatter(x=[best_p1_seq], y=[best_p2_val], mode='markers',
                                          name='Optimum', marker=dict(size=12, color='red', symbol='star')),
                               row=1, col=1)
            fig_seq.add_trace(go.Scatter(x=p1_range, y=rev1_seq, name="P1 Revenue",
                                          line=dict(color="dodgerblue")), row=1, col=2)
            fig_seq.add_trace(go.Scatter(x=p1_range, y=rev2_seq, name="P2 Revenue",
                                          line=dict(color="orange")), row=1, col=2)
            fig_seq.add_trace(go.Scatter(x=p1_range, y=total_rev_seq, name="Total Revenue",
                                          line=dict(color="green", width=3)), row=1, col=2)
            fig_seq.add_trace(go.Scatter(x=[best_p1_seq], y=[total_rev_seq[best_idx]], mode='markers',
                                          name='Optimum', marker=dict(size=12, color='red', symbol='star')),
                               row=1, col=2)
            fig_seq.update_xaxes(title_text="Period 1 Price")
            fig_seq.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_seq, use_container_width=True)

            st.markdown("**Key insight**: Low P1 price → more P1 sales → less inventory for P2 → P2 price rises.")


# =====================================================================
# MODE 4 – Marginal Revenue Allocation
# =====================================================================
elif mode == "Marginal Revenue Allocation":
    st.header("Marginal Revenue Allocation (Quantity-Based)")
    st.markdown("""
    Allocate a **fixed capacity** across multiple products by equalizing **weighted marginal revenues**.
    Each product has linear demand Q(p) = D − m·p, solved in terms of **quantity** as the decision variable.
    Products can consume **different amounts of capacity** per unit (e.g., seats take more space than standing).
    """)
    st.latex(r"\text{Revenue}_i(Q_i) = P_i(Q_i) \cdot Q_i = \frac{D_i - Q_i}{m_i} \cdot Q_i")
    st.latex(r"\text{Constraint: } \sum_i w_i \cdot Q_i \leq C \quad \text{(weighted capacity)}")
    st.latex(r"\text{At optimum: } \frac{MR_i}{w_i} = \lambda \;\;\forall i \quad \text{(equalized MR per unit capacity)}")

    # Preset examples
    preset = st.selectbox("Load example", ["Custom", "Vertigo Nightclub (Seats vs Non-Seats)"], key="mr_preset")

    if preset == "Vertigo Nightclub (Seats vs Non-Seats)":
        default_n = 2
        default_D = [9000.0, 40000.0]
        default_m = [100.0, 200.0]
        default_w = [1.5, 1.0]
        default_cap = 20000.0
        default_labels = ["Seats", "Non-Seats"]
        use_weights_default = True
    else:
        default_n = 3
        default_D = [40.0, 30.0, 20.0, 50.0, 35.0, 25.0]
        default_m = [5.0, 2.0, 1.0, 3.0, 2.5, 1.5]
        default_w = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        default_cap = 40.0
        default_labels = None
        use_weights_default = False

    num_products = st.slider("Number of products", 2, 6, default_n, key="mr_n")

    use_weights = st.checkbox("Use capacity weights (products use different amounts of space/capacity per unit)",
                               value=use_weights_default, key="mr_use_w")

    products = []
    weights = []
    product_labels = []
    mcols = st.columns(num_products)
    for i, mc in enumerate(mcols):
        with mc:
            if default_labels and i < len(default_labels):
                lbl = default_labels[i]
            else:
                lbl = f"Product {i+1}"
            st.markdown(f"**{lbl}**")
            product_labels.append(lbl)
            Di = st.number_input(f"D ({lbl})", value=default_D[i] if i < len(default_D) else 30.0,
                                 min_value=1.0, step=5.0, key=f"mr_D{i}")
            mi = st.number_input(f"m ({lbl})", value=default_m[i] if i < len(default_m) else 2.0,
                                 min_value=0.01, step=0.5, key=f"mr_m{i}", format="%.2f")
            if use_weights:
                wi = st.number_input(f"w ({lbl}) — capacity per unit", value=default_w[i] if i < len(default_w) else 1.0,
                                     min_value=0.01, step=0.5, key=f"mr_w{i}", format="%.2f")
            else:
                wi = 1.0
            products.append((Di, mi))
            weights.append(wi)

    total_cap_mr = st.number_input("Total capacity", value=default_cap, min_value=1.0, step=5.0, key="mr_cap")

    if use_weights:
        weight_str = ", ".join(f"{product_labels[i]}: {weights[i]}" for i in range(num_products))
        st.caption(f"Capacity weights: {weight_str}. Constraint: {' + '.join(f'{weights[i]}·Q_{product_labels[i]}' for i in range(num_products))} ≤ {fmt_int(total_cap_mr)}")

    # Optimize: maximize total revenue = sum of P_i(Q_i)*Q_i subject to sum w_i*Q_i <= cap
    def neg_total_rev_mr(quantities):
        total = 0
        for idx, (Di, mi) in enumerate(products):
            Qi = quantities[idx]
            Pi = max((Di - Qi) / mi, 0)
            total += Pi * Qi
        return -total

    bounds_mr = [(0, Di) for Di, mi in products]  # Q can't exceed D
    x0_mr = [min(Di / 2, total_cap_mr / (num_products * weights[i])) for i, (Di, mi) in enumerate(products)]

    def cap_con_mr(quantities):
        return total_cap_mr - sum(weights[i] * quantities[i] for i in range(num_products))
    constraints_mr = [{'type': 'ineq', 'fun': cap_con_mr}]

    res_mr = minimize(neg_total_rev_mr, x0_mr, method='SLSQP', bounds=bounds_mr, constraints=constraints_mr)
    opt_q_mr = res_mr.x

    # Compute shadow price (Lagrange multiplier on capacity)
    eps_mr = 0.1
    def solve_mr_cap(cap):
        def cc(q):
            return cap - sum(weights[i] * q[i] for i in range(num_products))
        r = minimize(neg_total_rev_mr, x0_mr, method='SLSQP', bounds=bounds_mr,
                     constraints=[{'type': 'ineq', 'fun': cc}])
        return -r.fun
    weighted_usage = sum(weights[i] * opt_q_mr[i] for i in range(num_products))
    if weighted_usage >= total_cap_mr - 0.5:
        shadow_mr = (solve_mr_cap(total_cap_mr + eps_mr) - solve_mr_cap(total_cap_mr)) / eps_mr
    else:
        shadow_mr = 0.0

    # Results
    st.subheader("Optimal Allocation")
    rcols_mr = st.columns(num_products + 1)
    total_rev_mr = 0
    mr_values = []
    mr_per_cap = []  # MR / weight (should be equalized)
    for i, (Di, mi) in enumerate(products):
        Qi = opt_q_mr[i]
        Pi = max((Di - Qi) / mi, 0)
        rev = Pi * Qi
        mr_i = (Di - 2 * Qi) / mi  # marginal revenue
        total_rev_mr += rev
        mr_values.append(mr_i)
        mr_per_cap.append(mr_i / weights[i])
        with rcols_mr[i]:
            st.metric(f"{product_labels[i]} Qty", fmt_qty(Qi))
            st.caption("Units allocated to this product.")
            st.metric(f"{product_labels[i]} Price", fmt_dollar(Pi))
            st.caption("Market-clearing price: P = (D − Q) / m.")
            st.metric(f"{product_labels[i]} Revenue", fmt_dollar(rev))
            st.caption("Revenue from this product: Price × Quantity.")
            st.metric(f"{product_labels[i]} MR", fmt_dollar(mr_i))
            st.caption("Marginal revenue — extra revenue from one more unit.")
            if use_weights and weights[i] != 1.0:
                st.metric(f"{product_labels[i]} MR / w", fmt_dollar(mr_per_cap[i]))
                st.caption(f"MR per unit of capacity used (w={weights[i]}). At optimum, this is equalized across products and equals the shadow price.")
            if use_weights:
                st.metric(f"{product_labels[i]} Space Used", fmt_qty(weights[i] * Qi))
                st.caption(f"Capacity consumed: {weights[i]} × {Qi:.1f} = {weights[i]*Qi:.1f}")

    with rcols_mr[-1]:
        st.metric("Total Revenue", fmt_dollar(total_rev_mr))
        st.caption("Combined revenue across all products.")
        st.metric("Total Qty Allocated", fmt_qty(sum(opt_q_mr)))
        st.caption("Total units allocated across all products.")
        if use_weights:
            st.metric("Total Capacity Used", fmt_qty(weighted_usage))
            st.caption(f"Weighted sum of quantities: Σ wᵢ·Qᵢ out of {fmt_int(total_cap_mr)} available.")
        if weighted_usage >= total_cap_mr - 0.01:
            st.info("Capacity is **fully allocated**.")

    # Optimality checks
    if use_weights:
        if len(set(round(mr_per_cap[i], 2) for i in range(len(mr_per_cap)) if opt_q_mr[i] > 0.01)) <= 1:
            st.success("MR/weight is equalized across products — optimality confirmed. Each unit of capacity earns the same marginal revenue regardless of which product it's allocated to.")
    else:
        if len(set(round(mr, 2) for mr in mr_values)) <= 1:
            st.success("Marginal revenues are equalized across all products — optimality confirmed.")

    # Shadow price
    if weighted_usage >= total_cap_mr - 0.5:
        st.metric("Shadow Price / Lagrange Multiplier (λ)", fmt_dollar(shadow_mr))
        st.caption("The marginal value of one additional unit of capacity. At optimum, λ = MRᵢ / wᵢ for all active products. This is the maximum you should pay to expand capacity.")

    # Python printout
    _results_mr = [f"{'='*65}", f"Marginal Revenue Allocation — Optimal Solution", f"{'='*65}"]
    for i, (Di, mi) in enumerate(products):
        Qi = opt_q_mr[i]
        Pi = max((Di - Qi) / mi, 0)
        rev = Pi * Qi
        w_str = f", w = {weights[i]}, MR/w = ${mr_per_cap[i]:.2f}" if use_weights and weights[i] != 1.0 else ""
        space_str = f", Space = {weights[i]*Qi:.1f}" if use_weights and weights[i] != 1.0 else ""
        _results_mr.append(f"{product_labels[i]}: Qty = {Qi:.1f}, Price = ${Pi:.2f}, Rev = ${rev:.2f}, MR = ${mr_values[i]:.2f}{w_str}{space_str}")
    _results_mr.append(f"{'─'*65}")
    _results_mr.append(f"Total Revenue:       ${total_rev_mr:.2f}")
    _results_mr.append(f"Total Qty:           {sum(opt_q_mr):.1f}")
    if use_weights:
        _results_mr.append(f"Total Capacity Used: {weighted_usage:.1f} / {total_cap_mr:.0f}")
    if weighted_usage >= total_cap_mr - 0.5:
        _results_mr.append(f"Lagrange Multiplier: ${shadow_mr:.2f}")
    _code_mr = [
        f"from scipy.optimize import minimize",
        f"",
        f"# Products: (D, m) demand parameters",
        f"products = {[(Di, mi) for Di, mi in products]}",
    ]
    if use_weights:
        _code_mr.append(f"weights  = {weights}  # capacity per unit")
    _code_mr += [
        f"capacity = {total_cap_mr}",
        f"",
        f"def neg_total_rev(quantities):",
        f"    return -sum(((D-Q)/m)*Q for Q, (D,m) in zip(quantities, products))",
        f"",
        f"# Constraint: {'sum(w_i * Q_i)' if use_weights else 'sum(Q_i)'} <= capacity",
        f"result = minimize(neg_total_rev, x0, method='SLSQP',",
        f"                  bounds=bounds, constraints=cap_constraint)",
        f"opt_quantities = result.x",
        f"",
        f"# At optimum: MR_i / w_i = lambda (shadow price) for all active products",
    ]
    python_output("MR Allocation", _code_mr, _results_mr)

    # Charts
    fig_mr = make_subplots(rows=1, cols=2, subplot_titles=("Revenue Curves (by Quantity)",
                           "Marginal Revenue" + (" per Unit Capacity (MR/w)" if use_weights else " Curves")))
    for i, (Di, mi) in enumerate(products):
        q_arr = np.linspace(0, Di * 0.95, 300)
        p_arr = np.maximum((Di - q_arr) / mi, 0)
        rev_arr = p_arr * q_arr
        mr_arr = (Di - 2 * q_arr) / mi
        if use_weights:
            mr_arr_plot = mr_arr / weights[i]
        else:
            mr_arr_plot = mr_arr

        fig_mr.add_trace(go.Scatter(x=q_arr, y=rev_arr, name=f"{product_labels[i]} Rev"), row=1, col=1)
        Qi = opt_q_mr[i]
        Pi = max((Di - Qi) / mi, 0)
        fig_mr.add_trace(go.Scatter(x=[Qi], y=[Pi * Qi], mode='markers',
                                     name=f"{product_labels[i]} Opt", marker=dict(size=12, symbol='star')), row=1, col=1)
        fig_mr.add_trace(go.Scatter(x=q_arr, y=mr_arr_plot, name=f"{product_labels[i]} {'MR/w' if use_weights else 'MR'}"), row=1, col=2)

    # Show equalized MR line (= shadow price)
    if use_weights and weighted_usage >= total_cap_mr - 0.5:
        fig_mr.add_hline(y=shadow_mr, line_dash="dot", line_color="red",
                          annotation_text=f"λ = {fmt_dollar(shadow_mr)}", row=1, col=2)
    elif mr_values:
        avg_mr = np.mean(mr_values)
        fig_mr.add_hline(y=avg_mr, line_dash="dot", line_color="red",
                          annotation_text=f"Equalized MR ≈ {fmt_dollar(avg_mr)}", row=1, col=2)

    fig_mr.update_xaxes(title_text="Quantity", row=1, col=1)
    fig_mr.update_xaxes(title_text="Quantity", row=1, col=2)
    fig_mr.update_yaxes(title_text="Revenue ($)", row=1, col=1)
    fig_mr.update_yaxes(title_text="MR/w ($)" if use_weights else "Marginal Revenue ($)", row=1, col=2)
    fig_mr.update_layout(height=450, showlegend=True)
    st.plotly_chart(fig_mr, use_container_width=True)


# =====================================================================
# MODE 5 – Marginal Value of Capacity
# =====================================================================
elif mode == "Marginal Value of Capacity":
    st.header("Marginal Value of Capacity")
    st.markdown("""
    Analyze how **optimal revenue changes as capacity increases**. The marginal value of capacity
    (shadow price) tells you the additional revenue from one more unit of capacity — critical for
    investment, overbooking, and resource allocation decisions.
    """)
    st.latex(r"\text{Marginal Value of Capacity} = \frac{\partial R^*(C)}{\partial C} \approx R^*(C+1) - R^*(C)")

    st.subheader("Demand Setup")
    num_seg_mvc = st.slider("Number of segments / products", 1, 5, 2, key="mvc_n")

    segments_mvc = []
    seg_cols = st.columns(num_seg_mvc)
    for i, sc in enumerate(seg_cols):
        with sc:
            st.markdown(f"**Segment {i+1}**")
            Di = st.number_input(f"D{i+1} (intercept)", value=[60.0, 40.0, 50.0, 30.0, 45.0][i] if i < 5 else 40.0,
                                 min_value=1.0, step=5.0, key=f"mvc_D{i}")
            mi = st.number_input(f"m{i+1} (slope)", value=[1.6, 0.4, 1.0, 0.5, 0.8][i] if i < 5 else 1.0,
                                 min_value=0.01, step=0.1, key=f"mvc_m{i}", format="%.2f")
            segments_mvc.append((Di, mi))

    col_cap1, col_cap2 = st.columns(2)
    with col_cap1:
        use_cost_mvc = st.checkbox("Include marginal cost", key="mvc_mc")
        cost_mvc = st.number_input("Marginal cost per unit", value=0.0, min_value=0.0, step=1.0, key="mvc_c") if use_cost_mvc else 0.0
    with col_cap2:
        cap_min = st.number_input("Capacity range — min", value=5.0, min_value=1.0, step=5.0, key="mvc_cmin")
        cap_max_val = st.number_input("Capacity range — max", value=100.0, min_value=2.0, step=10.0, key="mvc_cmax")
        current_cap = st.number_input("Current capacity (highlighted)", value=30.0, min_value=1.0, step=5.0, key="mvc_ccur")

    # Solve optimal revenue for a given capacity
    def solve_opt_revenue_mvc(cap):
        def neg_total_profit_mvc(prices):
            total = 0
            for idx, (Di, mi) in enumerate(segments_mvc):
                q = max(Di - mi * prices[idx], 0)
                total += (prices[idx] - cost_mvc) * q
            return -total

        bounds_mvc = [(0.01, seg[0] / seg[1]) for seg in segments_mvc]
        x0_mvc = [seg[0] / (2 * seg[1]) for seg in segments_mvc]

        def cap_con_mvc(prices):
            return cap - sum(max(seg[0] - seg[1] * prices[idx], 0) for idx, seg in enumerate(segments_mvc))

        res_mvc = minimize(neg_total_profit_mvc, x0_mvc, method='SLSQP', bounds=bounds_mvc,
                           constraints=[{'type': 'ineq', 'fun': cap_con_mvc}])
        opt_rev = -res_mvc.fun
        opt_prices = res_mvc.x
        total_q = sum(max(seg[0] - seg[1] * opt_prices[idx], 0) for idx, seg in enumerate(segments_mvc))
        return opt_rev, total_q, opt_prices

    # Compute over capacity range
    cap_range = np.linspace(cap_min, cap_max_val, 200)
    rev_curve = []
    qty_curve = []
    for c in cap_range:
        r, q, _ = solve_opt_revenue_mvc(c)
        rev_curve.append(r)
        qty_curve.append(q)
    rev_curve = np.array(rev_curve)
    qty_curve = np.array(qty_curve)

    # Marginal value (numerical derivative)
    dc = cap_range[1] - cap_range[0]
    marginal_value = np.gradient(rev_curve, dc)

    # Current capacity solution
    cur_rev, cur_q, cur_prices = solve_opt_revenue_mvc(current_cap)
    # Marginal value at current capacity
    eps_mv = 0.1
    rev_plus_mv, _, _ = solve_opt_revenue_mvc(current_cap + eps_mv)
    mv_at_current = (rev_plus_mv - cur_rev) / eps_mv

    # Unconstrained revenue (very large capacity)
    unc_rev, unc_q, _ = solve_opt_revenue_mvc(sum(D for D, m in segments_mvc))

    # Results
    st.subheader("Results at Current Capacity")
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("Current Capacity", fmt_int(current_cap))
    col_r1.caption("The total units available to sell across all segments.")
    col_r2.metric("Optimal Revenue" if not use_cost_mvc else "Optimal Profit", fmt_dollar(cur_rev))
    col_r2.caption("The best achievable revenue (or profit) given the current capacity constraint.")
    col_r3.metric("Total Qty Sold", fmt_qty(cur_q))
    col_r3.caption("Units actually sold at the optimum — equals capacity when the constraint is binding.")
    col_r4.metric("Marginal Value of Capacity", fmt_dollar(mv_at_current))
    col_r4.caption("The additional revenue from one more unit of capacity. This is the maximum you should pay to expand capacity.")

    if cur_q >= current_cap - 0.01:
        st.info(f"Capacity is **binding** at {fmt_int(current_cap)} units. Each additional unit is worth approximately **{fmt_dollar(mv_at_current)}**.")
    else:
        st.info(f"Capacity is **not binding** — only {fmt_qty(cur_q)} of {fmt_int(current_cap)} units used. Marginal value of additional capacity is near $0.")

    # Segment-level breakdown at current capacity
    st.subheader("Segment Breakdown at Current Capacity")
    seg_result_cols = st.columns(num_seg_mvc)
    for i, (Di, mi) in enumerate(segments_mvc):
        qi = max(Di - mi * cur_prices[i], 0)
        ri = cur_prices[i] * qi
        with seg_result_cols[i]:
            st.metric(f"Seg {i+1} Price", fmt_dollar(cur_prices[i]))
            st.metric(f"Seg {i+1} Qty", fmt_qty(qi))
            st.metric(f"Seg {i+1} Revenue", fmt_dollar(ri))

    # Revenue gap
    rev_gap = unc_rev - cur_rev
    if rev_gap > 0.01:
        st.metric("Revenue Lost Due to Capacity Constraint", fmt_dollar(rev_gap),
                   delta=f"-{rev_gap / unc_rev * 100:.1f}% vs unconstrained", delta_color="inverse")
        st.caption("The gap between your constrained revenue and what you'd earn with unlimited capacity. This quantifies the cost of scarcity.")

    # Python printout
    _results_mvc = [f"{'='*55}", f"Marginal Value of Capacity Analysis", f"{'='*55}"]
    _results_mvc.append(f"Current Capacity:        {current_cap:.0f}")
    _results_mvc.append(f"Optimal Revenue:         ${cur_rev:.2f}")
    _results_mvc.append(f"Total Qty Sold:          {cur_q:.1f}")
    _results_mvc.append(f"Utilization:             {min(cur_q/current_cap, 1.0)*100:.1f}%")
    _results_mvc.append(f"Marginal Value (dR/dC):  ${mv_at_current:.2f}")
    _results_mvc.append(f"{'─'*55}")
    for i, (Di, mi) in enumerate(segments_mvc):
        qi = max(Di - mi * cur_prices[i], 0)
        _results_mvc.append(f"Seg {i+1}: Price = ${cur_prices[i]:.2f}, Qty = {qi:.1f}, Rev = ${cur_prices[i]*qi:.2f}")
    if rev_gap > 0.01:
        _results_mvc.append(f"{'─'*55}")
        _results_mvc.append(f"Unconstrained Revenue:   ${unc_rev:.2f}")
        _results_mvc.append(f"Revenue Lost (scarcity): ${rev_gap:.2f} ({rev_gap/unc_rev*100:.1f}%)")
    python_output("Marginal Value of Capacity", [
        f"from scipy.optimize import minimize",
        f"",
        f"segments = {[(Di, mi) for Di, mi in segments_mvc]}",
        f"capacity = {current_cap}",
        f"",
        f"# Solve optimal revenue at capacity C",
        f"R_star = solve_optimal_revenue(capacity)  # = ${cur_rev:.2f}",
        f"",
        f"# Marginal value = dR*/dC ≈ R*(C+1) - R*(C)",
        f"R_plus = solve_optimal_revenue(capacity + 0.1)",
        f"marginal_value = (R_plus - R_star) / 0.1",
        f"print(f'Marginal Value of Capacity: ${{marginal_value:.2f}}')",
    ], _results_mvc)

    # Charts
    fig_mvc = make_subplots(rows=1, cols=3,
                             subplot_titles=("Optimal Revenue vs Capacity",
                                              "Marginal Value of Capacity",
                                              "Utilization vs Capacity"))

    # Revenue vs capacity
    fig_mvc.add_trace(go.Scatter(x=cap_range, y=rev_curve, name="R*(C)",
                                  line=dict(color="green", width=3)), row=1, col=1)
    fig_mvc.add_trace(go.Scatter(x=[current_cap], y=[cur_rev], mode='markers', name='Current',
                                  marker=dict(size=14, color='red', symbol='star')), row=1, col=1)
    fig_mvc.add_hline(y=unc_rev, line_dash="dot", line_color="gray",
                       annotation_text=f"Unconstrained = {fmt_dollar(unc_rev)}", row=1, col=1)

    # Marginal value vs capacity
    fig_mvc.add_trace(go.Scatter(x=cap_range, y=marginal_value, name="MV(C)",
                                  line=dict(color="purple", width=3)), row=1, col=2)
    fig_mvc.add_trace(go.Scatter(x=[current_cap], y=[mv_at_current], mode='markers', name='Current MV',
                                  marker=dict(size=14, color='red', symbol='star')), row=1, col=2)
    fig_mvc.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=2)

    # Utilization
    utilization = np.minimum(qty_curve / cap_range, 1.0) * 100
    fig_mvc.add_trace(go.Scatter(x=cap_range, y=utilization, name="Utilization %",
                                  line=dict(color="dodgerblue", width=3)), row=1, col=3)
    cur_util = min(cur_q / current_cap, 1.0) * 100
    fig_mvc.add_trace(go.Scatter(x=[current_cap], y=[cur_util], mode='markers', name='Current Util',
                                  marker=dict(size=14, color='red', symbol='star')), row=1, col=3)
    fig_mvc.add_hline(y=100, line_dash="dot", line_color="gray", annotation_text="100%", row=1, col=3)

    fig_mvc.update_xaxes(title_text="Capacity", row=1, col=1)
    fig_mvc.update_xaxes(title_text="Capacity", row=1, col=2)
    fig_mvc.update_xaxes(title_text="Capacity", row=1, col=3)
    fig_mvc.update_yaxes(title_text="Revenue ($)", row=1, col=1)
    fig_mvc.update_yaxes(title_text="Marginal Value ($)", row=1, col=2)
    fig_mvc.update_yaxes(title_text="Utilization (%)", row=1, col=3)
    fig_mvc.update_layout(height=450, showlegend=True)
    st.plotly_chart(fig_mvc, use_container_width=True)

    # Capacity sensitivity table
    with st.expander("Capacity Sensitivity Table"):
        table_caps = np.linspace(cap_min, cap_max_val, 20)
        table_rows = []
        prev_rev = None
        for c in table_caps:
            r, q, _ = solve_opt_revenue_mvc(c)
            mv = (r - prev_rev) / (table_caps[1] - table_caps[0]) if prev_rev is not None else mv_at_current
            prev_rev = r
            util = min(q / c, 1.0) * 100
            binding = "Yes" if q >= c - 0.01 else "No"
            table_rows.append({
                "Capacity": fmt_int(c),
                "Opt Revenue": fmt_dollar(r),
                "Qty Sold": fmt_qty(q),
                "Utilization": f"{util:.0f}%",
                "Marginal Value": fmt_dollar(mv),
                "Binding": binding,
            })
        st.table(table_rows)

    st.markdown("""
    **Key insights:**
    - The marginal value of capacity is **decreasing** — each additional unit is worth less than the previous one.
    - When capacity is **not binding** (excess capacity), the marginal value drops to **$0**.
    - The marginal value equals the **shadow price** (Lagrange multiplier) of the capacity constraint.
    - Use this to evaluate whether investing in additional capacity (at a given cost) is worthwhile.
    """)


# =====================================================================
# MODE 6 – Demand Estimation from Data
# =====================================================================
elif mode == "Demand Estimation from Data":
    st.header("Demand Estimation from Data")
    st.markdown("""
    Fit a demand curve from observed **price–quantity data**, then find the revenue-maximizing price.
    Enter data manually or use the preloaded example from the course template.
    """)

    data_source = st.radio("Data source", ["Use course example", "Enter manually", "Upload CSV"])

    import pandas as pd
    from scipy.optimize import curve_fit

    if data_source == "Use course example":
        # From the "Demand Curve Example 2" sheet
        df = pd.DataFrame({
            "Price": [1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200],
            "Purchases": [1, 2, 1, 5, 6, 6, 13, 11, 12, 19, 17, 20, 22, 25, 29, 28, 30],
        })
        df["Cumulative Demand"] = df["Purchases"].cumsum()
        st.dataframe(df, use_container_width=True)
        price_col, qty_col = "Price", "Cumulative Demand"

    elif data_source == "Enter manually":
        st.markdown("Enter price and quantity data (comma-separated, one pair per line):")
        raw = st.text_area("Price, Quantity (one per line)",
                           "1000, 1\n800, 15\n600, 57\n400, 135\n200, 247", height=200)
        rows_data = []
        for line in raw.strip().split("\n"):
            parts = line.split(",")
            if len(parts) == 2:
                try:
                    rows_data.append({"Price": float(parts[0].strip()), "Quantity": float(parts[1].strip())})
                except ValueError:
                    st.warning(f"Skipping invalid line: {line.strip()}")
        df = pd.DataFrame(rows_data)
        price_col, qty_col = "Price", "Quantity"
        st.dataframe(df, use_container_width=True)

    else:  # Upload CSV
        uploaded = st.file_uploader("Upload CSV with Price and Quantity columns", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.dataframe(df, use_container_width=True)
            price_col = st.selectbox("Price column", df.columns)
            qty_col = st.selectbox("Quantity column", df.columns)
        else:
            st.info("Upload a CSV file to proceed.")
            st.stop()

    prices_data = df[price_col].values.astype(float)
    qty_data = df[qty_col].values.astype(float)

    # Fit models
    st.subheader("Fitted Demand Models")

    fit_results = {}

    # Linear: Q = D - m*p
    try:
        def lin_fn(p, D, m):
            return D - m * p
        popt_lin, _ = curve_fit(lin_fn, prices_data, qty_data, p0=[300, 0.3])
        pred_lin = lin_fn(prices_data, *popt_lin)
        ss_res = np.sum((qty_data - pred_lin) ** 2)
        ss_tot = np.sum((qty_data - np.mean(qty_data)) ** 2)
        r2_lin = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        fit_results["Linear"] = {"params": popt_lin, "r2": r2_lin,
                                  "label": f"Q = {popt_lin[0]:,.1f} − {popt_lin[1]:.4f}·p  (R² = {r2_lin:.4f})"}
    except Exception:
        pass

    # Exponential: Q = a * exp(-b*p)
    try:
        def exp_fn(p, a, b):
            return a * np.exp(-b * p)
        popt_exp, _ = curve_fit(exp_fn, prices_data, qty_data, p0=[300, 0.005], maxfev=5000)
        pred_exp = exp_fn(prices_data, *popt_exp)
        ss_res = np.sum((qty_data - pred_exp) ** 2)
        ss_tot = np.sum((qty_data - np.mean(qty_data)) ** 2)
        r2_exp = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        fit_results["Exponential"] = {"params": popt_exp, "r2": r2_exp,
                                       "label": f"Q = {popt_exp[0]:,.1f}·exp(−{popt_exp[1]:.6f}·p)  (R² = {r2_exp:.4f})"}
    except Exception:
        pass

    # Power/Constant elasticity: Q = a * p^(-e)
    try:
        def power_fn(p, a, e):
            return a * np.power(p, -e)
        popt_pow, _ = curve_fit(power_fn, prices_data, qty_data, p0=[1e6, 1.5], maxfev=5000)
        pred_pow = power_fn(prices_data, *popt_pow)
        ss_res = np.sum((qty_data - pred_pow) ** 2)
        ss_tot = np.sum((qty_data - np.mean(qty_data)) ** 2)
        r2_pow = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        fit_results["Constant Elasticity"] = {"params": popt_pow, "r2": r2_pow,
                                               "label": f"Q = {popt_pow[0]:,.1f}·p^(−{popt_pow[1]:.3f})  (R² = {r2_pow:.4f})"}
    except Exception:
        pass

    # Display fit comparison
    if fit_results:
        best_model = max(fit_results, key=lambda k: fit_results[k]["r2"])
        for name, info in fit_results.items():
            marker = " **<-- Best fit**" if name == best_model else ""
            st.markdown(f"- **{name}**: {info['label']}{marker}")

        # Plot fits
        p_plot = np.linspace(max(prices_data.min() * 0.5, 1), prices_data.max() * 1.1, 500)
        fig_de = go.Figure()
        fig_de.add_trace(go.Scatter(x=prices_data, y=qty_data, mode='markers', name='Data',
                                     marker=dict(size=10, color='black')))
        colors = {"Linear": "dodgerblue", "Exponential": "orange", "Constant Elasticity": "green"}
        for name, info in fit_results.items():
            if name == "Linear":
                y_fit = lin_fn(p_plot, *info["params"])
            elif name == "Exponential":
                y_fit = exp_fn(p_plot, *info["params"])
            else:
                y_fit = power_fn(p_plot, *info["params"])
            y_fit = np.maximum(y_fit, 0)
            fig_de.add_trace(go.Scatter(x=p_plot, y=y_fit, name=f"{name} (R²={info['r2']:.3f})",
                                         line=dict(color=colors.get(name, "gray"))))
        fig_de.update_layout(title="Fitted Demand Curves", xaxis_title="Price", yaxis_title="Quantity", height=400)
        st.plotly_chart(fig_de, use_container_width=True)

        # Optimize using best-fit model
        st.subheader(f"Revenue Optimization (using {best_model} fit)")
        bp = fit_results[best_model]["params"]
        if best_model == "Linear":
            opt_p_de = bp[0] / (2 * bp[1])
            opt_q_de = max(lin_fn(opt_p_de, *bp), 0)
        elif best_model == "Exponential":
            res_de = minimize_scalar(lambda p: -(p * exp_fn(p, *bp)), bounds=(1, prices_data.max() * 2), method='bounded')
            opt_p_de = res_de.x
            opt_q_de = exp_fn(opt_p_de, *bp)
        else:
            res_de = minimize_scalar(lambda p: -(p * power_fn(p, *bp)), bounds=(1, prices_data.max() * 2), method='bounded')
            opt_p_de = res_de.x
            opt_q_de = power_fn(opt_p_de, *bp)

        opt_rev_de = opt_p_de * opt_q_de
        de1, de2, de3 = st.columns(3)
        de1.metric("Optimal Price", fmt_dollar(opt_p_de))
        de1.caption("The revenue-maximizing price derived from the best-fit demand model.")
        de2.metric("Expected Quantity", fmt_qty(opt_q_de))
        de2.caption("Predicted demand at the optimal price, based on the fitted curve.")
        de3.metric("Expected Revenue", fmt_dollar(opt_rev_de))
        de3.caption("Estimated maximum revenue: Optimal Price × Expected Quantity.")

        # Python printout
        _code_de = [
            f"from scipy.optimize import curve_fit, minimize_scalar",
            f"import numpy as np",
            f"",
            f"# Observed data: {len(prices_data)} price-quantity pairs",
            f"prices = [{', '.join(f'{p:.0f}' for p in prices_data[:5])}{'...' if len(prices_data)>5 else ''}]",
            f"quantities = [{', '.join(f'{q:.0f}' for q in qty_data[:5])}{'...' if len(qty_data)>5 else ''}]",
            f"",
        ]
        for name, info in fit_results.items():
            _code_de.append(f"# {name}: {info['label']}")
        _code_de += [
            f"",
            f"# Best fit: {best_model} (R² = {fit_results[best_model]['r2']:.4f})",
            f"opt_price = {opt_p_de:.2f}",
            f"opt_qty = demand_fitted(opt_price)  # = {opt_q_de:.1f}",
            f"opt_revenue = opt_price * opt_qty",
        ]
        _results_de = [
            f"{'='*50}",
            f"Demand Estimation Results",
            f"{'='*50}",
        ]
        for name, info in fit_results.items():
            tag = " <-- BEST" if name == best_model else ""
            _results_de.append(f"{name}: R² = {info['r2']:.4f}{tag}")
        _results_de += [
            f"{'─'*50}",
            f"Optimal Price:      ${opt_p_de:.2f}",
            f"Expected Quantity:  {opt_q_de:.1f}",
            f"Expected Revenue:   ${opt_rev_de:.2f}",
        ]
        python_output("Demand Estimation", _code_de, _results_de)

        # Revenue curve
        rev_plot = p_plot.copy()
        if best_model == "Linear":
            rev_y = rev_plot * np.maximum(lin_fn(rev_plot, *bp), 0)
        elif best_model == "Exponential":
            rev_y = rev_plot * exp_fn(rev_plot, *bp)
        else:
            rev_y = rev_plot * power_fn(rev_plot, *bp)

        fig_rev_de = go.Figure()
        fig_rev_de.add_trace(go.Scatter(x=rev_plot, y=rev_y, name="Revenue", line=dict(color="green")))
        fig_rev_de.add_trace(go.Scatter(x=[opt_p_de], y=[opt_rev_de], mode='markers', name='Optimum',
                                         marker=dict(size=14, color='red', symbol='star')))
        # Also show actual revenue from data
        fig_rev_de.add_trace(go.Scatter(x=prices_data, y=prices_data * qty_data, mode='markers',
                                         name='Actual Revenue', marker=dict(size=8, color='black', symbol='diamond')))
        fig_rev_de.update_layout(title="Revenue Curve (fitted)", xaxis_title="Price", yaxis_title="Revenue", height=400)
        st.plotly_chart(fig_rev_de, use_container_width=True)
    else:
        st.error("Could not fit any demand model to the data.")


# =====================================================================
# MODE 7 – Newsvendor / Quantity
# =====================================================================
elif mode == "Newsvendor / Quantity":
    st.header("Newsvendor Model / Quantity-Based RM")
    st.markdown("""
    Determine the **optimal quantity** (booking limit / order quantity) under demand uncertainty.
    Balances the cost of **spoilage** (unsold units) vs. **dilution/spill** (unmet demand).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Parameters")
        selling_price = st.number_input("Selling price (p)", value=100.0, min_value=0.01, step=10.0)
        unit_cost = st.number_input("Cost per unit (c)", value=60.0, min_value=0.0, step=5.0)
        salvage = st.number_input("Salvage value per unsold unit (s)", value=10.0, min_value=0.0, step=5.0)

        st.subheader("Demand Distribution")
        dist_type = st.selectbox("Distribution", ["Normal", "Uniform", "Poisson"])

        if dist_type == "Normal":
            mu = st.number_input("Mean demand (μ)", value=100.0, min_value=1.0, step=10.0)
            sigma = st.number_input("Std dev (σ)", value=20.0, min_value=1.0, step=5.0)
        elif dist_type == "Uniform":
            lo = st.number_input("Min demand", value=50.0, min_value=0.0, step=10.0)
            hi = st.number_input("Max demand", value=150.0, min_value=0.0, step=10.0)
            if hi <= lo:
                st.error("Max demand must be greater than Min demand.")
                st.stop()
        else:
            lam = st.number_input("λ (mean demand)", value=100.0, min_value=1.0, step=10.0)

    # Critical ratio
    Cu = selling_price - unit_cost  # underage cost
    Co = unit_cost - salvage        # overage cost
    if Cu + Co > 0:
        critical_ratio = Cu / (Cu + Co)
    else:
        critical_ratio = 0.5

    # Optimal quantity
    from scipy import stats

    if dist_type == "Normal":
        Q_star = stats.norm.ppf(critical_ratio, loc=mu, scale=sigma)
        dist = stats.norm(loc=mu, scale=sigma)
    elif dist_type == "Uniform":
        Q_star = stats.uniform.ppf(critical_ratio, loc=lo, scale=hi - lo)
        dist = stats.uniform(loc=lo, scale=hi - lo)
    else:
        Q_star = stats.poisson.ppf(critical_ratio, mu=lam)
        dist = stats.poisson(mu=lam)

    Q_star = max(Q_star, 0)

    # Expected profit calculation
    def expected_profit(Q):
        if dist_type == "Poisson":
            demands = np.arange(0, int(lam * 3) + 1)
            probs = dist.pmf(demands)
            sold = np.minimum(demands, Q)
            unsold = np.maximum(Q - demands, 0)
            profit = selling_price * sold + salvage * unsold - unit_cost * Q
            return np.sum(profit * probs)
        else:
            from scipy.integrate import quad
            def integrand(d):
                sold = min(d, Q)
                unsold = max(Q - d, 0)
                profit = selling_price * sold + salvage * unsold - unit_cost * Q
                return profit * dist.pdf(d)
            result, _ = quad(integrand, 0, dist.ppf(0.999))
            return result

    exp_profit = expected_profit(Q_star)

    with col2:
        st.subheader("Newsvendor Solution")
        st.latex(r"Q^* = F^{-1}\left(\frac{C_u}{C_u + C_o}\right)")
        m1, m2 = st.columns(2)
        m1.metric("Underage cost (Cu = p − c)", fmt_dollar(Cu))
        m1.caption("The opportunity cost of being short one unit — the profit margin you miss per unit of unmet demand.")
        m2.metric("Overage cost (Co = c − s)", fmt_dollar(Co))
        m2.caption("The cost of having one unit too many — the loss per unsold unit (cost minus salvage value).")
        st.metric("Critical Ratio", f"{critical_ratio:.4f}")
        st.caption("Cu / (Cu + Co). This is the target service level — the probability that demand won't exceed your order quantity.")
        st.metric("Optimal Order Quantity (Q*)", fmt_qty(Q_star))
        st.caption("The order quantity where the expected marginal cost of overage equals the expected marginal cost of underage.")
        st.metric("Expected Profit at Q*", fmt_dollar(exp_profit))
        st.caption("Average profit across all possible demand scenarios when ordering Q* units.")

    # Python printout
    python_output("Newsvendor Model", [
        f"from scipy import stats",
        f"",
        f"p = {selling_price}    # selling price",
        f"c = {unit_cost}    # unit cost",
        f"s = {salvage}    # salvage value",
        f"",
        f"Cu = p - c           # underage cost = {Cu:.2f}",
        f"Co = c - s           # overage cost = {Co:.2f}",
        f"CR = Cu / (Cu + Co)  # critical ratio = {critical_ratio:.4f}",
        f"",
        f"# Demand ~ {dist_type}",
        f"Q_star = dist.ppf(CR)  # optimal order quantity",
        f"E_profit = expected_profit(Q_star)",
        f"",
        f"print(f'Critical Ratio:  {{CR:.4f}}')",
        f"print(f'Optimal Q*:      {{Q_star:.1f}}')",
        f"print(f'E[Profit]:       ${{E_profit:.2f}}')",
    ], [
        f"{'='*45}",
        f"Newsvendor Solution",
        f"{'='*45}",
        f"Underage Cost (Cu): ${Cu:.2f}",
        f"Overage Cost (Co):  ${Co:.2f}",
        f"Critical Ratio:     {critical_ratio:.4f}",
        f"Optimal Q*:         {Q_star:.1f}",
        f"Expected Profit:    ${exp_profit:.2f}",
    ])

    # Profit vs Quantity chart
    if dist_type == "Normal":
        q_range = np.arange(max(1, int(mu - 3 * sigma)), int(mu + 3 * sigma) + 1)
    elif dist_type == "Uniform":
        q_range = np.arange(max(1, int(lo * 0.5)), int(hi * 1.2) + 1)
    else:
        q_range = np.arange(max(1, int(lam * 0.3)), int(lam * 2) + 1)

    profits = [expected_profit(q) for q in q_range]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q_range, y=profits, name="E[Profit]", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=[Q_star], y=[exp_profit], mode='markers', name='Q*',
                             marker=dict(size=14, color='red', symbol='star')))
    fig.add_vline(x=Q_star, line_dash="dot", line_color="red", annotation_text=f"Q*={fmt_int(Q_star)}")
    fig.update_layout(title="Expected Profit vs. Order Quantity", xaxis_title="Quantity (Q)",
                      yaxis_title="E[Profit]", height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Protection levels (EMSR-like)
    with st.expander("Protection Levels (Two Fare Classes)"):
        st.markdown("**EMSR-b style**: How many seats to protect for the higher-fare class?")
        f_high = st.number_input("High fare", value=200.0, min_value=0.01, step=10.0)
        f_low = st.number_input("Low fare", value=80.0, min_value=0.01, step=10.0)
        mu_h = st.number_input("Mean high-fare demand", value=40.0, min_value=1.0, step=5.0)
        sigma_h = st.number_input("Std dev high-fare demand", value=15.0, min_value=1.0, step=2.0)
        total_cap = st.number_input("Total capacity", value=100.0, min_value=1.0, step=10.0)

        if f_high > f_low:
            protection_ratio = f_low / f_high
            protection_level = stats.norm.ppf(1 - protection_ratio, loc=mu_h, scale=sigma_h)
            protection_level = max(0, min(protection_level, total_cap))
            booking_limit_low = total_cap - protection_level

            st.metric("Protection Level (high fare)", f"{fmt_qty(protection_level)} seats")
            st.caption("The number of seats to reserve (protect) for high-fare passengers. Do not sell these to low-fare customers.")
            st.metric("Booking Limit (low fare)", f"{fmt_qty(booking_limit_low)} seats")
            st.caption("The maximum number of seats to sell at the low fare. Once this limit is reached, remaining seats are held for high-fare demand.")
            st.caption(f"Protect seats for high fare until P(Demand_high > y) = f_low/f_high = {protection_ratio:.3f}")
        else:
            st.warning("High fare must exceed low fare.")


# =====================================================================
# MODE 8 – Incentive Compatible Pricing
# =====================================================================
elif mode == "Incentive Compatible Pricing":
    st.header("Incentive Compatible Pricing")
    st.markdown("""
    Design **self-selecting menus** (bundles, fare classes, versioning) where each customer type
    **voluntarily chooses** the option intended for them. The two key constraints are:

    - **Incentive Compatibility (IC)**: each type prefers its own bundle to every other type's bundle.
    - **Individual Rationality (IR)**: each type gets non-negative surplus from its own bundle.

    This is the foundation of **second-degree price discrimination**, **versioned goods**, and **fare-class design**.
    """)

    ic_mode = st.radio(
        "Scenario",
        ["Two-Segment Menu Design",
         "Multi-Segment Menu Design",
         "Versioning / Quality Differentiation",
         "Fare-Class Seat Allocation"],
        horizontal=True,
    )

    # =================================================================
    # SUB-MODE: Two-Segment Menu Design
    # =================================================================
    if ic_mode == "Two-Segment Menu Design":
        st.subheader("Two-Segment Menu Design")
        st.markdown("""
        A monopolist sells to **High** (H) and **Low** (L) type customers who differ in willingness-to-pay.
        Each type has a valuation per unit of quality. The firm designs two bundles: (q_H, p_H) and (q_L, p_L).
        """)
        st.latex(r"\text{Surplus}_i = v_i \cdot q_j - p_j \quad \text{(type } i \text{ choosing bundle } j\text{)}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Customer Types")
            v_H = st.number_input("v_H (High type valuation per unit quality)", value=10.0, min_value=0.01, step=1.0, key="ic2_vH")
            v_L = st.number_input("v_L (Low type valuation per unit quality)", value=5.0, min_value=0.01, step=1.0, key="ic2_vL")
            if v_L >= v_H:
                st.warning("v_L must be less than v_H for meaningful segmentation.")

            st.subheader("Market Composition")
            n_H = st.number_input("n_H (number of High types)", value=40.0, min_value=1.0, step=10.0, key="ic2_nH")
            n_L = st.number_input("n_L (number of Low types)", value=60.0, min_value=1.0, step=10.0, key="ic2_nL")

            st.subheader("Cost")
            st.latex(r"C(q) = c \cdot q^2 / 2 \quad \text{(convex cost of quality)}")
            c_cost = st.number_input("c (cost parameter)", value=1.0, min_value=0.01, step=0.5, key="ic2_c")

        # ---- First-Best (Perfect Price Discrimination) ----
        # Firm observes type, extracts all surplus
        # max v_i * q_i - c*q_i^2/2 => q_i* = v_i / c
        # Price: p_i = v_i * q_i (extract all surplus)
        q_H_fb = v_H / c_cost
        q_L_fb = v_L / c_cost
        p_H_fb = v_H * q_H_fb
        p_L_fb = v_L * q_L_fb
        profit_H_fb = p_H_fb - c_cost * q_H_fb**2 / 2
        profit_L_fb = p_L_fb - c_cost * q_L_fb**2 / 2
        total_profit_fb = n_H * profit_H_fb + n_L * profit_L_fb

        # ---- Second-Best (IC + IR Constrained) ----
        # With two types, binding constraints are:
        #   IR_L: v_L * q_L - p_L >= 0
        #   IC_H: v_H * q_H - p_H >= v_H * q_L - p_L
        #
        # Optimal: IR_L binds => p_L = v_L * q_L
        #          IC_H binds => p_H = v_H * q_H - (v_H - v_L) * q_L
        #
        # q_H stays at first-best: q_H* = v_H / c
        # q_L is distorted downward:
        #   q_L* = v_L/c - (n_H / n_L) * (v_H - v_L) / c
        #        = [v_L - (n_H/n_L)(v_H - v_L)] / c
        q_H_sb = v_H / c_cost  # no distortion at the top
        q_L_sb_raw = (v_L - (n_H / n_L) * (v_H - v_L)) / c_cost
        q_L_sb = max(q_L_sb_raw, 0)  # can't be negative

        # Prices from binding constraints
        p_L_sb = v_L * q_L_sb  # IR_L binds: zero surplus for L
        p_H_sb = v_H * q_H_sb - (v_H - v_L) * q_L_sb  # IC_H binds

        # Surpluses
        surplus_H_sb = v_H * q_H_sb - p_H_sb  # = (v_H - v_L) * q_L_sb (info rent)
        surplus_L_sb = v_L * q_L_sb - p_L_sb  # = 0

        # Profits per customer
        profit_H_sb = p_H_sb - c_cost * q_H_sb**2 / 2
        profit_L_sb = p_L_sb - c_cost * q_L_sb**2 / 2
        total_profit_sb = n_H * profit_H_sb + n_L * profit_L_sb

        # ---- Uniform Pricing (single bundle) ----
        # Sell same (q, p) to everyone — constrained by low type's IR
        # max (n_H + n_L) * [v_L * q - c*q^2/2]  (price = v_L * q to keep L)
        q_uniform = v_L / c_cost
        p_uniform = v_L * q_uniform
        profit_uniform = (n_H + n_L) * (p_uniform - c_cost * q_uniform**2 / 2)

        # ---- Exclude Low Type ----
        # Only sell to H at first-best
        profit_exclude = n_H * profit_H_fb

        # Check IC constraints
        ic_H_check = (v_H * q_H_sb - p_H_sb) - (v_H * q_L_sb - p_L_sb)  # should be >= 0
        ic_L_check = (v_L * q_L_sb - p_L_sb) - (v_L * q_H_sb - p_H_sb)  # should be >= 0
        ir_H_check = v_H * q_H_sb - p_H_sb  # should be >= 0
        ir_L_check = v_L * q_L_sb - p_L_sb  # should be >= 0

        with col2:
            st.subheader("Second-Best Menu (IC Optimal)")
            st.markdown("The menu that maximizes profit while satisfying all IC and IR constraints:")

            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("**High-Type Bundle**")
                st.metric("Quality (q_H)", fmt_qty2(q_H_sb))
                st.caption("No distortion at the top — High type gets the efficient quality level v_H / c.")
                st.metric("Price (p_H)", fmt_dollar(p_H_sb))
                st.caption("Set so High type is just indifferent between this bundle and the Low-type bundle.")
            with rc2:
                st.markdown("**Low-Type Bundle**")
                st.metric("Quality (q_L)", fmt_qty2(q_L_sb))
                st.caption("Distorted downward to make it less attractive to the High type, reducing the information rent.")
                st.metric("Price (p_L)", fmt_dollar(p_L_sb))
                st.caption("Extracts all Low-type surplus (IR_L binds). Low type gets zero surplus.")

            if q_L_sb <= 0:
                st.warning("Low-type quality is zero — it's more profitable to exclude the Low segment entirely.")

            st.divider()

            st.metric("High-Type Surplus (Information Rent)", fmt_dollar(surplus_H_sb))
            st.caption("The 'information rent' the firm must leave to the High type to prevent them from mimicking the Low type. Equals (v_H − v_L) × q_L.")
            st.metric("Total Profit (Second-Best)", fmt_dollar(total_profit_sb))
            st.caption("Maximum profit under incentive compatibility — less than first-best due to information rent and quality distortion.")

        # Constraint verification
        st.subheader("Constraint Verification")
        vc1, vc2, vc3, vc4 = st.columns(4)
        vc1.metric("IC_H (≥ 0)", f"{ic_H_check:.4f}")
        vc1.caption("High type weakly prefers own bundle. IC_H binds at optimum (= 0).")
        vc2.metric("IC_L (≥ 0)", f"{ic_L_check:.4f}")
        vc2.caption("Low type weakly prefers own bundle.")
        vc3.metric("IR_H (≥ 0)", f"{ir_H_check:.4f}")
        vc3.caption("High type gets non-negative surplus (slack — gets info rent).")
        vc4.metric("IR_L (≥ 0)", f"{ir_L_check:.4f}")
        vc4.caption("Low type gets non-negative surplus. IR_L binds at optimum (= 0).")

        all_satisfied = ic_H_check >= -0.01 and ic_L_check >= -0.01 and ir_H_check >= -0.01 and ir_L_check >= -0.01
        if all_satisfied:
            st.success("All IC and IR constraints are satisfied.")
        else:
            st.error("Some constraints are violated — check parameter values.")

        # Comparison table
        st.subheader("Strategy Comparison")
        import pandas as pd
        comparison = pd.DataFrame({
            "Strategy": ["First-Best (observe types)", "Second-Best (IC menu)", "Uniform Pricing (one bundle)", "Exclude Low Type"],
            "q_H": [fmt_qty2(q_H_fb), fmt_qty2(q_H_sb), fmt_qty2(q_uniform), fmt_qty2(q_H_fb)],
            "p_H": [fmt_dollar(p_H_fb), fmt_dollar(p_H_sb), fmt_dollar(p_uniform), fmt_dollar(p_H_fb)],
            "q_L": [fmt_qty2(q_L_fb), fmt_qty2(q_L_sb), fmt_qty2(q_uniform), "—"],
            "p_L": [fmt_dollar(p_L_fb), fmt_dollar(p_L_sb), fmt_dollar(p_uniform), "—"],
            "Total Profit": [fmt_dollar(total_profit_fb), fmt_dollar(total_profit_sb),
                             fmt_dollar(profit_uniform), fmt_dollar(profit_exclude)],
        })
        st.table(comparison)
        st.caption("First-best is unattainable without observing types. The IC menu is the best achievable outcome. Compare against simpler alternatives to see the value of segmentation.")

        # Python printout
        python_output("Two-Segment IC Menu", [
            f"v_H, v_L = {v_H}, {v_L}   # valuations per unit quality",
            f"n_H, n_L = {n_H}, {n_L}   # number of each type",
            f"c = {c_cost}               # cost parameter (C(q) = c*q^2/2)",
            f"",
            f"# First-best (observe types): q_i = v_i / c",
            f"q_H_fb = v_H / c  # = {q_H_fb:.2f}",
            f"q_L_fb = v_L / c  # = {q_L_fb:.2f}",
            f"",
            f"# Second-best (IC constrained):",
            f"# No distortion at top: q_H = v_H / c",
            f"q_H = v_H / c  # = {q_H_sb:.2f}",
            f"# Downward distortion: q_L = [v_L - (n_H/n_L)(v_H-v_L)] / c",
            f"q_L = max((v_L - (n_H/n_L)*(v_H-v_L)) / c, 0)  # = {q_L_sb:.2f}",
            f"",
            f"# Prices from binding constraints:",
            f"p_L = v_L * q_L           # IR_L binds (zero surplus)",
            f"p_H = v_H*q_H - (v_H-v_L)*q_L  # IC_H binds",
            f"info_rent = (v_H - v_L) * q_L   # surplus left to H type",
        ], [
            f"{'='*55}",
            f"Incentive Compatible Menu (Second-Best)",
            f"{'='*55}",
            f"High Bundle:  q_H = {q_H_sb:.2f},  p_H = ${p_H_sb:.2f}",
            f"Low Bundle:   q_L = {q_L_sb:.2f},  p_L = ${p_L_sb:.2f}",
            f"{'─'*55}",
            f"Information Rent (H surplus): ${surplus_H_sb:.2f}",
            f"Low-Type Surplus:             ${surplus_L_sb:.2f}",
            f"{'─'*55}",
            f"IC_H (>= 0):  {ic_H_check:.4f}  {'✓' if ic_H_check >= -0.01 else '✗'}",
            f"IC_L (>= 0):  {ic_L_check:.4f}  {'✓' if ic_L_check >= -0.01 else '✗'}",
            f"IR_H (>= 0):  {ir_H_check:.4f}  {'✓' if ir_H_check >= -0.01 else '✗'}",
            f"IR_L (>= 0):  {ir_L_check:.4f}  {'✓' if ir_L_check >= -0.01 else '✗'}",
            f"{'─'*55}",
            f"Second-Best Profit:  ${total_profit_sb:.2f}",
            f"First-Best Profit:   ${total_profit_fb:.2f}",
            f"Uniform Pricing:     ${profit_uniform:.2f}",
            f"Exclude Low Type:    ${profit_exclude:.2f}",
        ])

        # Charts
        fig_ic = make_subplots(rows=1, cols=2,
                                subplot_titles=("Bundles in (Quality, Price) Space",
                                                "Profit Comparison"))

        # Bundle plot with indifference curves
        q_range = np.linspace(0, max(q_H_fb, q_H_sb) * 1.3, 300)

        # H-type indifference curve through H bundle
        ic_H_curve = v_H * q_range - surplus_H_sb  # p = v_H * q - surplus_H
        # L-type indifference curve through L bundle (zero surplus)
        ic_L_curve = v_L * q_range  # p = v_L * q

        fig_ic.add_trace(go.Scatter(x=q_range, y=ic_H_curve, name=f"H indifference (surplus={fmt_dollar(surplus_H_sb)})",
                                     line=dict(color="red", dash="dash")), row=1, col=1)
        fig_ic.add_trace(go.Scatter(x=q_range, y=ic_L_curve, name="L indifference (surplus=$0)",
                                     line=dict(color="blue", dash="dash")), row=1, col=1)

        # Plot bundles
        fig_ic.add_trace(go.Scatter(x=[q_H_sb], y=[p_H_sb], mode='markers+text',
                                     name='H Bundle (IC)', text=["H*"], textposition="top right",
                                     marker=dict(size=16, color='red', symbol='star')), row=1, col=1)
        fig_ic.add_trace(go.Scatter(x=[q_L_sb], y=[p_L_sb], mode='markers+text',
                                     name='L Bundle (IC)', text=["L*"], textposition="top right",
                                     marker=dict(size=16, color='blue', symbol='star')), row=1, col=1)

        # First-best bundles
        fig_ic.add_trace(go.Scatter(x=[q_H_fb], y=[p_H_fb], mode='markers+text',
                                     name='H First-Best', text=["H_FB"], textposition="bottom right",
                                     marker=dict(size=12, color='red', symbol='diamond')), row=1, col=1)
        fig_ic.add_trace(go.Scatter(x=[q_L_fb], y=[p_L_fb], mode='markers+text',
                                     name='L First-Best', text=["L_FB"], textposition="bottom right",
                                     marker=dict(size=12, color='blue', symbol='diamond')), row=1, col=1)

        # Profit comparison bar chart
        strategies = ["First-Best", "IC Menu", "Uniform", "Exclude L"]
        profits_compare = [total_profit_fb, total_profit_sb, profit_uniform, profit_exclude]
        colors_bar = ["gray", "green", "dodgerblue", "orange"]
        fig_ic.add_trace(go.Bar(x=strategies, y=profits_compare, name="Total Profit",
                                 marker_color=colors_bar, showlegend=False), row=1, col=2)

        fig_ic.update_xaxes(title_text="Quality (q)", row=1, col=1)
        fig_ic.update_yaxes(title_text="Price (p)", row=1, col=1)
        fig_ic.update_yaxes(title_text="Profit ($)", row=1, col=2)
        fig_ic.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig_ic, use_container_width=True)

        st.markdown("""
        **Key Insights:**
        - **No distortion at the top**: the High type always gets efficient quality (q_H = v_H / c).
        - **Downward distortion at the bottom**: Low-type quality is reduced to limit the information rent paid to High types.
        - **IC_H binds**: the High type is just indifferent between the two bundles — any cheaper H-bundle and they'd switch to L.
        - **IR_L binds**: the Low type gets zero surplus — the firm extracts everything from them.
        - The cost of private information = First-Best profit − Second-Best profit.
        """)

    # =================================================================
    # SUB-MODE: Multi-Segment Menu Design
    # =================================================================
    elif ic_mode == "Multi-Segment Menu Design":
        st.subheader("Multi-Segment Menu Design")
        st.markdown("""
        Extend the two-type model to **N customer types** with valuations v₁ < v₂ < ... < vₙ.
        The IC-optimal menu distorts quality downward for all types except the highest.
        """)

        num_types = st.slider("Number of customer types", 2, 6, 3, key="icm_n")

        default_v = [3.0, 6.0, 10.0, 14.0, 18.0, 22.0]
        default_n = [50.0, 30.0, 20.0, 15.0, 10.0, 8.0]

        types_data = []
        type_cols = st.columns(num_types)
        for i, tc in enumerate(type_cols):
            with tc:
                st.markdown(f"**Type {i+1}**")
                vi = st.number_input(f"v_{i+1} (valuation)", value=default_v[i] if i < 6 else 5.0 * (i + 1),
                                     min_value=0.01, step=1.0, key=f"icm_v{i}")
                ni = st.number_input(f"n_{i+1} (count)", value=default_n[i] if i < 6 else 20.0,
                                     min_value=1.0, step=5.0, key=f"icm_n{i}")
                types_data.append((vi, ni))

        c_cost_m = st.number_input("c (cost parameter, C(q) = c·q²/2)", value=1.0, min_value=0.01, step=0.5, key="icm_c")

        # Sort types by valuation
        types_data.sort(key=lambda x: x[0])
        valuations = [t[0] for t in types_data]
        counts = [t[1] for t in types_data]

        # Check valuations are distinct
        if len(set(valuations)) < len(valuations):
            st.warning("Valuations should be distinct for meaningful segmentation.")

        # First-best qualities
        q_fb = [v / c_cost_m for v in valuations]

        # Second-best: solve using the standard screening result
        # "Virtual valuation" for type i: φ_i = v_i - (sum of n_j for j>i) / n_i * (v_{i+1} - v_i)
        # For the highest type, φ_N = v_N
        # q_i* = max(φ_i / c, 0)
        n_types = len(valuations)
        virtual_vals = []
        for i in range(n_types):
            if i == n_types - 1:
                phi_i = valuations[i]
            else:
                # Weight of higher types that would mimic
                higher_mass = sum(counts[j] for j in range(i + 1, n_types))
                phi_i = valuations[i] - (higher_mass / counts[i]) * (valuations[i + 1] - valuations[i])
            virtual_vals.append(phi_i)

        q_sb = [max(phi / c_cost_m, 0) for phi in virtual_vals]

        # Ensure monotonicity (iron if needed): q must be non-decreasing
        # If q_sb is not monotone, we need to pool types (ironing)
        q_sb_ironed = list(q_sb)
        for i in range(len(q_sb_ironed) - 2, -1, -1):
            if q_sb_ironed[i] > q_sb_ironed[i + 1]:
                q_sb_ironed[i] = q_sb_ironed[i + 1]

        # Prices: from binding IC constraints (downward adjacent)
        # p_1 = v_1 * q_1 (IR_1 binds)
        # p_{i+1} = p_i + v_{i+1} * (q_{i+1} - q_i)  ... actually:
        # IC_i,i-1: v_i * q_i - p_i >= v_i * q_{i-1} - p_{i-1}
        # Binding: p_i = p_{i-1} + v_i * (q_i - q_{i-1})
        # Starting from p_1 = v_1 * q_1
        p_sb = [0.0] * n_types
        p_sb[0] = valuations[0] * q_sb_ironed[0]  # IR_1 binds
        for i in range(1, n_types):
            p_sb[i] = p_sb[i - 1] + valuations[i] * (q_sb_ironed[i] - q_sb_ironed[i - 1])

        # Surpluses
        surpluses = [valuations[i] * q_sb_ironed[i] - p_sb[i] for i in range(n_types)]

        # Profits
        profits_per = [p_sb[i] - c_cost_m * q_sb_ironed[i]**2 / 2 for i in range(n_types)]
        total_profit_m = sum(counts[i] * profits_per[i] for i in range(n_types))

        # First-best profit
        p_fb = [valuations[i] * q_fb[i] for i in range(n_types)]
        profits_fb_per = [p_fb[i] - c_cost_m * q_fb[i]**2 / 2 for i in range(n_types)]
        total_profit_fb_m = sum(counts[i] * profits_fb_per[i] for i in range(n_types))

        # Display results
        st.subheader("Optimal IC Menu")
        import pandas as pd
        menu_data = []
        for i in range(n_types):
            menu_data.append({
                "Type": f"Type {i+1}",
                "Valuation (v)": fmt_qty2(valuations[i]),
                "Count (n)": fmt_int(counts[i]),
                "Virtual Val (φ)": fmt_qty2(virtual_vals[i]),
                "FB Quality": fmt_qty2(q_fb[i]),
                "IC Quality": fmt_qty2(q_sb_ironed[i]),
                "IC Price": fmt_dollar(p_sb[i]),
                "Surplus": fmt_dollar(surpluses[i]),
                "Profit/cust": fmt_dollar(profits_per[i]),
            })
        st.table(menu_data)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Total IC Profit", fmt_dollar(total_profit_m))
        mc1.caption("Maximum profit under incentive compatibility — the best the firm can do without observing types.")
        mc2.metric("First-Best Profit", fmt_dollar(total_profit_fb_m))
        mc2.caption("Profit if the firm could observe each customer's type and charge accordingly.")
        mc3.metric("Information Rent (cost of IC)", fmt_dollar(total_profit_fb_m - total_profit_m))
        mc3.caption("The profit sacrificed due to private information. This is the total surplus left to customers to keep them self-selecting.")

        # Python printout
        _results_icm = [f"{'='*65}", f"Multi-Segment IC Menu", f"{'='*65}"]
        _results_icm.append(f"{'Type':<8} {'v':>6} {'n':>6} {'phi':>8} {'q_FB':>8} {'q_IC':>8} {'Price':>10} {'Surplus':>10}")
        _results_icm.append(f"{'─'*65}")
        for i in range(n_types):
            _results_icm.append(
                f"Type {i+1:<3} {valuations[i]:>6.1f} {counts[i]:>6.0f} {virtual_vals[i]:>8.2f} "
                f"{q_fb[i]:>8.2f} {q_sb_ironed[i]:>8.2f} {p_sb[i]:>9.2f} {surpluses[i]:>9.2f}"
            )
        _results_icm.append(f"{'─'*65}")
        _results_icm.append(f"IC Profit:          ${total_profit_m:.2f}")
        _results_icm.append(f"First-Best Profit:  ${total_profit_fb_m:.2f}")
        _results_icm.append(f"Information Rent:    ${total_profit_fb_m - total_profit_m:.2f}")
        python_output("Multi-Segment IC Menu", [
            f"# {n_types} customer types, sorted by valuation",
            f"valuations = {[round(v, 1) for v in valuations]}",
            f"counts     = {[round(n, 0) for n in counts]}",
            f"c = {c_cost_m}",
            f"",
            f"# Virtual valuation: phi_i = v_i - (sum n_j for j>i)/n_i * (v_{{i+1}} - v_i)",
            f"# IC quality: q_i = max(phi_i / c, 0)",
            f"# Price: p_1 = v_1*q_1, then p_i = p_{{i-1}} + v_i*(q_i - q_{{i-1}})",
        ], _results_icm)

        # Charts
        fig_icm = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Quality by Type (FB vs IC)", "Surplus Distribution"))

        type_labels = [f"Type {i+1}" for i in range(n_types)]
        fig_icm.add_trace(go.Bar(x=type_labels, y=q_fb, name="First-Best Quality",
                                  marker_color="lightblue"), row=1, col=1)
        fig_icm.add_trace(go.Bar(x=type_labels, y=q_sb_ironed, name="IC Quality",
                                  marker_color="steelblue"), row=1, col=1)

        fig_icm.add_trace(go.Bar(x=type_labels, y=surpluses, name="Customer Surplus",
                                  marker_color="orange"), row=1, col=2)
        fig_icm.add_trace(go.Bar(x=type_labels, y=profits_per, name="Firm Profit/cust",
                                  marker_color="green"), row=1, col=2)

        fig_icm.update_yaxes(title_text="Quality", row=1, col=1)
        fig_icm.update_yaxes(title_text="$ per customer", row=1, col=2)
        fig_icm.update_layout(height=450, showlegend=True, barmode='group')
        st.plotly_chart(fig_icm, use_container_width=True)

        st.markdown("""
        **Key Insights:**
        - **Virtual valuation** φᵢ adjusts each type's true valuation for the information rent cost. Quality is set where φᵢ = MC.
        - Only the **highest type** gets efficient quality. All others are **distorted downward**.
        - **Information rents increase** with type — higher types get more surplus because they could always mimic lower types.
        - If a virtual valuation is negative, that type is **excluded** (quality = 0).
        """)

    # =================================================================
    # SUB-MODE: Versioning / Quality Differentiation
    # =================================================================
    elif ic_mode == "Versioning / Quality Differentiation":
        st.subheader("Versioning / Quality Differentiation")
        st.markdown("""
        A firm offers **two versions** of a product (e.g., Economy vs. Premium, Basic vs. Pro).
        Customers self-select based on willingness-to-pay for quality. The firm chooses **quality levels**
        and **prices** for each version to maximize total profit subject to IC and IR constraints.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Market Parameters")
            st.markdown("**Customer Types**")
            wtp_H = st.number_input("High-type WTP per unit quality (θ_H)", value=100.0, min_value=0.01, step=10.0, key="ver_tH")
            wtp_L = st.number_input("Low-type WTP per unit quality (θ_L)", value=40.0, min_value=0.01, step=10.0, key="ver_tL")
            frac_H = st.slider("Fraction of High types (λ)", 0.05, 0.95, 0.4, step=0.05, key="ver_frac")
            frac_L = 1.0 - frac_H
            market_size_v = st.number_input("Total market size", value=1000.0, min_value=1.0, step=100.0, key="ver_M")

            st.markdown("**Cost of Quality**")
            st.latex(r"C(q) = \frac{1}{2} q^2")
            st.caption("Quadratic cost: higher quality is increasingly expensive to produce.")

        n_H_v = frac_H * market_size_v
        n_L_v = frac_L * market_size_v

        # First-best: q_i = θ_i (since C'(q) = q, set θ_i = q_i)
        q_H_v_fb = wtp_H
        q_L_v_fb = wtp_L

        # Second-best
        q_H_v_sb = wtp_H  # no distortion at top
        q_L_v_sb_raw = wtp_L - (frac_H / frac_L) * (wtp_H - wtp_L)
        q_L_v_sb = max(q_L_v_sb_raw, 0)

        # Prices
        p_L_v = wtp_L * q_L_v_sb  # IR_L binds
        p_H_v = wtp_H * q_H_v_sb - (wtp_H - wtp_L) * q_L_v_sb  # IC_H binds

        surplus_H_v = wtp_H * q_H_v_sb - p_H_v
        surplus_L_v = wtp_L * q_L_v_sb - p_L_v

        profit_per_H = p_H_v - q_H_v_sb**2 / 2
        profit_per_L = p_L_v - q_L_v_sb**2 / 2
        total_profit_v = n_H_v * profit_per_H + n_L_v * profit_per_L

        # Alternative: sell only premium
        profit_premium_only = n_H_v * (wtp_H * q_H_v_fb - q_H_v_fb**2 / 2)
        # Alternative: sell one version at low WTP
        q_one = wtp_L  # set quality for low type
        profit_one_version = market_size_v * (wtp_L * q_one - q_one**2 / 2)

        with col2:
            st.subheader("Optimal Versioning Menu")

            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("**Premium Version**")
                st.metric("Quality", fmt_qty2(q_H_v_sb))
                st.caption("Efficient quality — no distortion for the top tier.")
                st.metric("Price", fmt_dollar(p_H_v))
                st.caption("Discounted relative to first-best to prevent High types from buying Economy.")
                st.metric("Margin per unit", fmt_dollar(profit_per_H))
            with rc2:
                st.markdown("**Economy Version**")
                st.metric("Quality", fmt_qty2(q_L_v_sb))
                st.caption("Deliberately degraded to make Premium attractive to High types.")
                st.metric("Price", fmt_dollar(p_L_v))
                st.caption("Extracts all Low-type surplus.")
                st.metric("Margin per unit", fmt_dollar(profit_per_L))

            if q_L_v_sb <= 0:
                st.warning("Economy version quality is zero — better to only offer Premium and exclude Low types.")

            st.divider()
            st.metric("Total Profit (Two Versions)", fmt_dollar(total_profit_v))
            st.caption("Profit from offering the IC-optimal two-version menu.")

            st.subheader("Alternative Strategies")
            st.metric("Profit: Premium Only", fmt_dollar(profit_premium_only))
            st.caption(f"Only sell to High types ({fmt_int(n_H_v)} customers) at full surplus extraction.")
            st.metric("Profit: One Version for All", fmt_dollar(profit_one_version))
            st.caption(f"Sell one product at Low-type quality to all {fmt_int(market_size_v)} customers.")

            best_strategy = max(
                [("Two Versions (IC)", total_profit_v), ("Premium Only", profit_premium_only), ("One Version", profit_one_version)],
                key=lambda x: x[1]
            )
            st.success(f"Best strategy: **{best_strategy[0]}** with profit {fmt_dollar(best_strategy[1])}")

        # Python printout
        python_output("Versioning", [
            f"theta_H, theta_L = {wtp_H}, {wtp_L}  # WTP per unit quality",
            f"frac_H = {frac_H}  # fraction of High types",
            f"market_size = {market_size_v}",
            f"# Cost: C(q) = q^2 / 2",
            f"",
            f"# Premium: q_H = theta_H = {q_H_v_sb:.2f} (no distortion)",
            f"# Economy: q_L = max(theta_L - (frac_H/frac_L)*(theta_H-theta_L), 0)",
            f"q_L = max({wtp_L} - ({frac_H}/{frac_L:.2f})*({wtp_H}-{wtp_L}), 0)  # = {q_L_v_sb:.2f}",
            f"",
            f"p_L = theta_L * q_L          # = ${p_L_v:.2f}",
            f"p_H = theta_H*q_H - (theta_H-theta_L)*q_L  # = ${p_H_v:.2f}",
        ], [
            f"{'='*50}",
            f"Versioning — Optimal Menu",
            f"{'='*50}",
            f"Premium: Quality = {q_H_v_sb:.2f}, Price = ${p_H_v:.2f}, Margin = ${profit_per_H:.2f}",
            f"Economy: Quality = {q_L_v_sb:.2f}, Price = ${p_L_v:.2f}, Margin = ${profit_per_L:.2f}",
            f"{'─'*50}",
            f"Two Versions profit:  ${total_profit_v:.2f}",
            f"Premium Only profit:  ${profit_premium_only:.2f}",
            f"One Version profit:   ${profit_one_version:.2f}",
            f"Best: {best_strategy[0]} (${best_strategy[1]:.2f})",
        ])

        # Quality distortion chart
        fig_ver = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Version Quality Comparison", "Profit by Strategy"))

        fig_ver.add_trace(go.Bar(x=["Premium", "Economy"], y=[q_H_v_fb, q_L_v_fb],
                                  name="First-Best Quality", marker_color="lightblue"), row=1, col=1)
        fig_ver.add_trace(go.Bar(x=["Premium", "Economy"], y=[q_H_v_sb, q_L_v_sb],
                                  name="IC Quality", marker_color="steelblue"), row=1, col=1)

        fig_ver.add_trace(go.Bar(
            x=["Two Versions", "Premium Only", "One Version"],
            y=[total_profit_v, profit_premium_only, profit_one_version],
            marker_color=["green", "orange", "dodgerblue"],
            showlegend=False
        ), row=1, col=2)

        fig_ver.update_yaxes(title_text="Quality Level", row=1, col=1)
        fig_ver.update_yaxes(title_text="Total Profit ($)", row=1, col=2)
        fig_ver.update_layout(height=450, barmode='group')
        st.plotly_chart(fig_ver, use_container_width=True)

        st.markdown("""
        **Real-World Examples:**
        - **Software**: Free vs. Pro tiers — free version deliberately limited (not just cheaper to produce)
        - **Airlines**: Economy cramped seats aren't just cost savings — they make Business Class worth the premium
        - **Streaming**: Ad-supported vs. ad-free tiers
        - The economy version is **intentionally degraded** below the cost-efficient level — this is rational because it increases High-type willingness to pay for Premium.
        """)

    # =================================================================
    # SUB-MODE: Fare-Class Seat Allocation
    # =================================================================
    elif ic_mode == "Fare-Class Seat Allocation":
        st.subheader("Fare-Class Seat Allocation with IC Constraints")
        st.markdown("""
        Allocate seats across **fare classes** where each class is a bundle of (restrictions, price).
        IC constraints ensure business travelers don't buy discount fares and leisure travelers don't buy
        up to business class. Restrictions (advance purchase, Saturday stay, refundability) act as the
        **quality differentiation** mechanism.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Fare Classes")
            st.markdown("**Business Class (High fare, flexible)**")
            f_B = st.number_input("Business fare ($)", value=800.0, min_value=0.01, step=50.0, key="fc_fB")
            v_B_flex = st.number_input("Business traveler value of flexibility ($)", value=300.0, min_value=0.0, step=50.0, key="fc_vBflex")

            st.markdown("**Discount Class (Low fare, restricted)**")
            f_D = st.number_input("Discount fare ($)", value=200.0, min_value=0.01, step=50.0, key="fc_fD")
            v_L_flex = st.number_input("Leisure traveler value of flexibility ($)", value=50.0, min_value=0.0, step=10.0, key="fc_vLflex")

            st.subheader("Demand & Capacity")
            cap_fc = st.number_input("Total seats", value=150.0, min_value=1.0, step=10.0, key="fc_cap")
            d_B_mean = st.number_input("Expected business demand (mean)", value=40.0, min_value=1.0, step=5.0, key="fc_dB")
            d_B_std = st.number_input("Business demand std dev", value=15.0, min_value=1.0, step=5.0, key="fc_dBs")
            d_D_mean = st.number_input("Expected discount demand (mean)", value=120.0, min_value=1.0, step=10.0, key="fc_dD")

        # IC analysis
        # Business traveler's net value from each class:
        #   Buy Business: trip_value - f_B + v_B_flex (gets flexibility)
        #   Buy Discount: trip_value - f_D + 0 (no flexibility)
        # IC_B: f_B - v_B_flex <= f_D  =>  f_B <= f_D + v_B_flex (business won't switch down)
        # Equivalently: f_B - f_D <= v_B_flex (fare premium <= flexibility value)

        fare_gap = f_B - f_D
        ic_B_satisfied = fare_gap <= v_B_flex
        ic_B_slack = v_B_flex - fare_gap

        # Leisure traveler:
        #   Buy Discount: trip_value - f_D
        #   Buy Business: trip_value - f_B + v_L_flex
        # IC_L: f_D <= f_B - v_L_flex  =>  f_B - f_D >= v_L_flex (leisure won't switch up)
        ic_L_satisfied = fare_gap >= v_L_flex
        ic_L_slack = fare_gap - v_L_flex

        # EMSR protection level
        from scipy import stats
        if f_B > f_D:
            protection_ratio_fc = f_D / f_B
            prot_level = stats.norm.ppf(1 - protection_ratio_fc, loc=d_B_mean, scale=d_B_std)
            prot_level = max(0, min(prot_level, cap_fc))
            booking_limit_D = cap_fc - prot_level
        else:
            prot_level = 0
            booking_limit_D = cap_fc

        # Expected revenues
        exp_B_sold = min(d_B_mean, prot_level)
        exp_D_sold = min(d_D_mean, booking_limit_D)
        exp_rev = f_B * exp_B_sold + f_D * exp_D_sold

        # Revenue if no IC (business buys discount)
        rev_no_ic = f_D * min(d_B_mean + d_D_mean, cap_fc)

        with col2:
            st.subheader("IC Constraint Analysis")

            st.metric("Fare Premium (f_B − f_D)", fmt_dollar(fare_gap))
            st.caption("The extra amount charged for the flexible business fare over the restricted discount fare.")

            st.metric("Business Value of Flexibility", fmt_dollar(v_B_flex))
            st.caption("How much a business traveler values the flexibility features (refundability, no advance purchase, etc.).")

            if ic_B_satisfied:
                st.success(f"IC_Business satisfied: fare premium ({fmt_dollar(fare_gap)}) ≤ flexibility value ({fmt_dollar(v_B_flex)}). Business travelers willingly pay the premium.")
                st.metric("IC_B Slack", fmt_dollar(ic_B_slack))
                st.caption("How much room before business travelers would switch to discount. Positive = constraint is satisfied with margin.")
            else:
                st.error(f"IC_Business VIOLATED: fare premium ({fmt_dollar(fare_gap)}) > flexibility value ({fmt_dollar(v_B_flex)}). Business travelers would buy discount!")
                st.metric("IC_B Violation", fmt_dollar(-ic_B_slack))
                st.caption("The fare premium exceeds the flexibility value — reduce business fare or increase restrictions on discount.")

            if ic_L_satisfied:
                st.success(f"IC_Leisure satisfied: fare gap ({fmt_dollar(fare_gap)}) ≥ leisure flexibility value ({fmt_dollar(v_L_flex)}). Leisure travelers won't buy up.")
            else:
                st.warning(f"IC_Leisure VIOLATED: fare gap ({fmt_dollar(fare_gap)}) < leisure flexibility value ({fmt_dollar(v_L_flex)}). Some leisure travelers may buy business class!")

            st.divider()
            st.subheader("Seat Allocation (EMSR)")
            st.metric("Protection Level (Business)", fmt_qty(prot_level))
            st.caption("Seats reserved for business-fare demand. Discount sales are capped to protect these seats.")
            st.metric("Booking Limit (Discount)", fmt_qty(booking_limit_D))
            st.caption("Maximum discount tickets to sell. Once reached, remaining seats are held for business travelers.")

            st.divider()
            st.metric("Expected Revenue (with IC)", fmt_dollar(exp_rev))
            st.caption("Revenue when fare fences work and each type buys the intended fare class.")
            st.metric("Revenue if IC Fails (all buy discount)", fmt_dollar(rev_no_ic))
            st.caption("Revenue if business travelers switch to the discount fare — the cost of failed segmentation.")
            st.metric("Value of IC / Fare Fences", fmt_dollar(exp_rev - rev_no_ic))
            st.caption("The additional revenue earned because fare fences keep business travelers in the higher fare class.")

        # Python printout
        python_output("Fare-Class IC Analysis", [
            f"f_B = {f_B}   # business fare",
            f"f_D = {f_D}   # discount fare",
            f"v_B_flex = {v_B_flex}  # business value of flexibility",
            f"v_L_flex = {v_L_flex}  # leisure value of flexibility",
            f"",
            f"fare_gap = f_B - f_D  # = ${fare_gap:.2f}",
            f"",
            f"# IC_Business: fare_gap <= v_B_flex? {fare_gap:.2f} <= {v_B_flex:.2f} => {'YES' if ic_B_satisfied else 'NO'}",
            f"# IC_Leisure:  fare_gap >= v_L_flex? {fare_gap:.2f} >= {v_L_flex:.2f} => {'YES' if ic_L_satisfied else 'NO'}",
            f"",
            f"# EMSR Protection Level",
            f"from scipy import stats",
            f"protection = stats.norm.ppf(1 - f_D/f_B, loc={d_B_mean}, scale={d_B_std})",
            f"booking_limit = capacity - protection",
        ], [
            f"{'='*55}",
            f"Fare-Class IC Analysis",
            f"{'='*55}",
            f"Fare Premium:       ${fare_gap:.2f}",
            f"IC_Business (<=):   {'PASS' if ic_B_satisfied else 'FAIL'}  (slack = ${ic_B_slack:.2f})",
            f"IC_Leisure (>=):    {'PASS' if ic_L_satisfied else 'FAIL'}  (slack = ${ic_L_slack:.2f})",
            f"{'─'*55}",
            f"Protection Level:   {prot_level:.1f} seats",
            f"Booking Limit (D):  {booking_limit_D:.1f} seats",
            f"{'─'*55}",
            f"Revenue (with IC):         ${exp_rev:.2f}",
            f"Revenue (IC fails):        ${rev_no_ic:.2f}",
            f"Value of Fare Fences:      ${exp_rev - rev_no_ic:.2f}",
        ])

        # Chart: IC regions
        fig_fc = make_subplots(rows=1, cols=2,
                                subplot_titles=("Incentive Compatibility Region",
                                                "Revenue Impact"))

        # IC feasible region in (f_D, f_B) space
        fd_range = np.linspace(0, f_B * 1.5, 300)
        # IC_B: f_B <= f_D + v_B_flex  => f_B boundary = f_D + v_B_flex
        ic_B_line = fd_range + v_B_flex
        # IC_L: f_B >= f_D + v_L_flex  => f_B boundary = f_D + v_L_flex
        ic_L_line = fd_range + v_L_flex

        fig_fc.add_trace(go.Scatter(x=fd_range, y=ic_B_line,
                                     name=f"IC_B: f_B = f_D + {fmt_dollar(v_B_flex)}",
                                     line=dict(color="red", dash="dash")), row=1, col=1)
        fig_fc.add_trace(go.Scatter(x=fd_range, y=ic_L_line,
                                     name=f"IC_L: f_B = f_D + {fmt_dollar(v_L_flex)}",
                                     line=dict(color="blue", dash="dash")), row=1, col=1)

        # Shade feasible region
        fb_feasible_upper = np.minimum(ic_B_line, f_B * 2)
        fb_feasible_lower = ic_L_line
        fig_fc.add_trace(go.Scatter(
            x=np.concatenate([fd_range, fd_range[::-1]]),
            y=np.concatenate([fb_feasible_upper, fb_feasible_lower[::-1]]),
            fill='toself', fillcolor='rgba(0,200,0,0.15)',
            line=dict(width=0), name="IC Feasible Region",
            showlegend=True
        ), row=1, col=1)

        # Current point
        fig_fc.add_trace(go.Scatter(x=[f_D], y=[f_B], mode='markers+text',
                                     name='Current Fares', text=["Current"],
                                     textposition="top right",
                                     marker=dict(size=16, color='red' if not (ic_B_satisfied and ic_L_satisfied) else 'green',
                                                 symbol='star')), row=1, col=1)

        # Revenue comparison
        fig_fc.add_trace(go.Bar(
            x=["With IC (fences work)", "Without IC (all buy cheap)"],
            y=[exp_rev, rev_no_ic],
            marker_color=["green", "red"],
            showlegend=False
        ), row=1, col=2)

        fig_fc.update_xaxes(title_text="Discount Fare (f_D)", row=1, col=1)
        fig_fc.update_yaxes(title_text="Business Fare (f_B)", row=1, col=1)
        fig_fc.update_yaxes(title_text="Revenue ($)", row=1, col=2)
        fig_fc.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig_fc, use_container_width=True)

        st.markdown("""
        **Common Fare Fences (IC Mechanisms):**
        | Fence | Why It Works |
        |-------|-------------|
        | Advance purchase (7/14/21 days) | Business trips are booked late |
        | Saturday night stay | Business travelers want to go home |
        | Non-refundable | Business travelers need flexibility |
        | Round-trip requirement | Prevents one-way cherry-picking |
        | Capacity controls | Limit discount seats available |

        **Key principle**: the fare premium must be **less than** the business traveler's value of flexibility,
        but **more than** the leisure traveler's value, creating a corridor where both types self-select correctly.
        """)


# =====================================================================
# MODE 9 – Loan Pricing Optimization (Nomis Case)
# =====================================================================
elif mode == "Loan Pricing Optimization":
    st.header("Loan Pricing Optimization")
    st.markdown("""
    Estimate the probability of a customer **accepting** a loan offer as a function of the
    quoted **APR** using **logistic regression**, then find the APR that maximizes
    **expected net revenue per quote** = P(Accept | APR) × Net Revenue(APR).
    """)

    # --- Additional variables ---
    st.subheader("Model Specification")
    include_fico = st.checkbox("Include FICO Score", value=False, key="nomis_fico")
    include_comp = st.checkbox("Include Competition Rate", value=False, key="nomis_comp")
    extra_vars = []
    if include_fico:
        extra_vars.append("FICO")
    if include_comp:
        extra_vars.append("Competition Rate")

    # --- Data source ---
    data_source = st.radio("Data source", ["Manual logistic parameters", "Upload Excel / CSV"], horizontal=True, key="nomis_src")

    beta0_val = None
    beta1_val = None
    beta_extra = {}  # {var_name: coefficient}
    extra_avgs = {}  # {var_name: average value for curve plotting}
    uploaded_df = None
    apr_col_name = "APR"
    accept_col_name = "Accept?"

    if data_source == "Upload Excel / CSV":
        st.markdown("Upload a file with an **APR** column and an **Accept?** (0/1) column.")
        uploaded_file = st.file_uploader("Upload data file", type=["xlsx", "xls", "csv"], key="nomis_file")
        if uploaded_file is not None:
            import pandas as pd
            try:
                if uploaded_file.name.endswith(".csv"):
                    uploaded_df = pd.read_csv(uploaded_file)
                else:
                    uploaded_df = pd.read_excel(uploaded_file, index_col=0)
                st.success(f"Loaded {len(uploaded_df)} rows.")
                st.dataframe(uploaded_df.head(10), use_container_width=True)

                # Detect columns — prefer exact "APR" match
                all_cols = list(uploaded_df.columns)
                if "APR" in all_cols:
                    apr_default_idx = 0
                    possible_apr = ["APR"] + [c for c in all_cols if c != "APR"]
                else:
                    possible_apr = [c for c in all_cols if "apr" in c.lower()]
                    if not possible_apr:
                        possible_apr = all_cols
                    apr_default_idx = 0
                possible_accept = [c for c in all_cols if "accept" in c.lower() or "response" in c.lower() or "buy" in c.lower()]
                apr_col_name = st.selectbox("APR column", possible_apr, index=apr_default_idx, key="nomis_apr_col")
                accept_col_name = st.selectbox("Accept (0/1) column", possible_accept if possible_accept else all_cols, key="nomis_acc_col")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                uploaded_df = None
        else:
            st.info("Upload a data file or switch to **Manual logistic parameters** to enter coefficients directly.")
    else:
        extra_formula = " + ".join([f"Beta{i+2} × {v}" for i, v in enumerate(extra_vars)])
        full_formula = f"Beta0 + Beta1 × APR" + (f" + {extra_formula}" if extra_formula else "")
        st.markdown(f"Enter the logistic regression coefficients directly: **P(Accept) = 1 / (1 + exp(-({full_formula})))**")
        beta0_val = st.number_input("Beta0 (intercept)", value=5.4501, step=0.1, format="%.4f", key="nomis_b0")
        beta1_val = st.number_input("Beta1 (APR coefficient, typically negative)", value=-0.9790, step=0.01, format="%.4f", key="nomis_b1")
        for idx_ev, ev_name in enumerate(extra_vars):
            beta_extra[ev_name] = st.number_input(f"Beta{idx_ev+2} ({ev_name} coefficient)", value=0.0, step=0.01, format="%.4f", key=f"nomis_bx_{idx_ev}")
            extra_avgs[ev_name] = st.number_input(f"Average {ev_name} (for plotting)", value=700.0 if "FICO" in ev_name else 5.0, step=0.1, format="%.2f", key=f"nomis_avg_{idx_ev}")

    st.divider()

    # --- Loan parameters ---
    st.subheader("Loan Parameters")
    col_lp1, col_lp2, col_lp3 = st.columns(3)
    with col_lp1:
        loan_amount = st.number_input("Loan amount ($)", value=21000.0, min_value=100.0, step=1000.0, key="nomis_amt")
    with col_lp2:
        term_months = st.number_input("Term (months)", value=60, min_value=1, step=12, key="nomis_term")
    with col_lp3:
        cost_funds_rate = st.number_input("Cost of funds rate (% APR)", value=1.40, min_value=0.0, step=0.1, format="%.2f", key="nomis_cof")

    apr_min = st.number_input("APR range — min (%)", value=4.0, min_value=0.01, step=0.5, key="nomis_aprmin")
    apr_max = st.number_input("APR range — max (%)", value=13.0, min_value=0.02, step=0.5, key="nomis_aprmax")

    # --- Monthly payment helper ---
    def monthly_payment(rate_annual_pct, amount, nperiods):
        """Monthly payment for a fixed-rate amortizing loan."""
        r = rate_annual_pct / 1200.0  # monthly rate
        if r < 1e-12:
            return amount / nperiods
        return amount * r * np.power(1 + r, nperiods) / (np.power(1 + r, nperiods) - 1)

    # --- Fit logistic regression or use manual params ---
    from scipy.special import expit

    run_ok = False
    if data_source == "Upload Excel / CSV" and uploaded_df is not None:
        try:
            from sklearn.linear_model import LogisticRegression
            # Build feature matrix: APR + optional extra variables
            feature_cols = [apr_col_name] + extra_vars
            missing_cols = [c for c in extra_vars if c not in uploaded_df.columns]
            if missing_cols:
                st.error(f"Columns not found in data: {missing_cols}. Available: {list(uploaded_df.columns)}")
            else:
                X_arr = uploaded_df[feature_cols].to_numpy()
                Accept_arr = uploaded_df[[accept_col_name]].to_numpy().ravel()

                modelLR = LogisticRegression(solver='liblinear', random_state=0)
                modelLR.fit(X_arr, Accept_arr)
                r_sq = modelLR.score(X_arr, Accept_arr)

                beta0_val = modelLR.intercept_[0]
                beta1_val = modelLR.coef_[0, 0]
                for idx_ev, ev_name in enumerate(extra_vars):
                    beta_extra[ev_name] = modelLR.coef_[0, idx_ev + 1]
                    extra_avgs[ev_name] = uploaded_df[ev_name].mean()
                run_ok = True

                st.subheader("Logistic Regression Results")
                n_coefs = 2 + len(extra_vars)
                lr_cols = st.columns(min(n_coefs + 1, 5))
                with lr_cols[0]:
                    st.metric("Beta0 (intercept)", f"{beta0_val:.4f}")
                    st.caption("Baseline log-odds of acceptance.")
                with lr_cols[1]:
                    st.metric("Beta1 (APR)", f"{beta1_val:.4f}")
                    st.caption("Change in log-odds per 1% APR increase.")
                for idx_ev, ev_name in enumerate(extra_vars):
                    with lr_cols[2 + idx_ev]:
                        st.metric(f"Beta{idx_ev+2} ({ev_name})", f"{beta_extra[ev_name]:.4f}")
                        st.caption(f"Effect of {ev_name} on log-odds of acceptance.")
                with lr_cols[min(n_coefs, 4)]:
                    st.metric("Classification Accuracy", f"{r_sq:.4f}")
                    st.caption("Fraction of correctly predicted Accept/Reject decisions on training data.")

        except ImportError:
            st.error("scikit-learn is required. Run: `pip install scikit-learn`")
        except Exception as e:
            st.error(f"Logistic regression failed: {e}")
    elif data_source == "Manual logistic parameters" and beta0_val is not None:
        run_ok = True

    if run_ok and beta0_val is not None and beta1_val is not None:
        # --- Compute curves ---
        x_apr = np.linspace(apr_min, apr_max, 500)
        # Log-odds: Beta0 + Beta1*APR + sum(Beta_k * avg_k) for extra vars
        extra_contribution = sum(beta_extra.get(v, 0) * extra_avgs.get(v, 0) for v in extra_vars)
        logodds = beta0_val + beta1_val * x_apr + extra_contribution
        prob_accept = expit(logodds)

        cost_per_month = monthly_payment(cost_funds_rate, loan_amount, term_months)
        revenue_per_month = np.array([monthly_payment(a, loan_amount, term_months) for a in x_apr])
        net_revenue = revenue_per_month - cost_per_month

        expected_net_rev = prob_accept * net_revenue

        # --- Optimal APR ---
        opt_idx = np.argmax(expected_net_rev)
        opt_apr = x_apr[opt_idx]
        opt_exp_rev = expected_net_rev[opt_idx]
        opt_prob = prob_accept[opt_idx]
        opt_net = net_revenue[opt_idx]

        st.divider()
        st.subheader("Optimization Results")
        if extra_vars:
            avg_note = ", ".join([f"{v}={extra_avgs[v]:.2f}" for v in extra_vars])
            st.caption(f"Curves computed at average values: {avg_note}")
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.metric("Optimal APR", f"{opt_apr:.2f}%")
            st.caption("The APR that maximizes expected net revenue per quote — the sweet spot between acceptance probability and margin.")
        with col_r2:
            st.metric("Max Expected Net Revenue", fmt_dollar(opt_exp_rev))
            st.caption("P(Accept) × Net Revenue at the optimal APR — the expected profit per quote.")
        with col_r3:
            st.metric("P(Accept) at Optimal APR", f"{opt_prob:.3f}")
            st.caption("Probability a customer accepts the loan at the optimal APR.")
        with col_r4:
            st.metric("Net Revenue if Accepted", fmt_dollar(opt_net))
            st.caption("Monthly net revenue (payment − cost of funds) if the loan is accepted at the optimal APR.")

        col_r5, col_r6 = st.columns(2)
        with col_r5:
            st.metric("Monthly Payment at Optimal APR", fmt_dollar(revenue_per_month[opt_idx]))
            st.caption("The borrower's monthly payment at the optimal APR.")
        with col_r6:
            st.metric("Cost of Funds (monthly)", fmt_dollar(cost_per_month))
            st.caption("The lender's monthly cost of funds — what it costs to finance the loan.")

        # --- Segmented analysis by extra variables ---
        if extra_vars and data_source == "Upload Excel / CSV" and uploaded_df is not None:
            st.divider()
            st.subheader("Segmented Analysis")
            for ev_name in extra_vars:
                st.markdown(f"**Optimal APR by {ev_name} level** (other variables at average)")
                ev_data = uploaded_df[ev_name]
                levels = {
                    "Low (P25)": ev_data.quantile(0.25),
                    "Med (P50)": ev_data.quantile(0.50),
                    "High (P75)": ev_data.quantile(0.75),
                }
                seg_cols = st.columns(len(levels))
                for idx_lv, (lv_label, lv_val) in enumerate(levels.items()):
                    # Recompute P(Accept) with this level replacing the average
                    other_contrib = sum(beta_extra.get(v, 0) * extra_avgs.get(v, 0) for v in extra_vars if v != ev_name)
                    lo_seg = beta0_val + beta1_val * x_apr + beta_extra[ev_name] * lv_val + other_contrib
                    pa_seg = expit(lo_seg)
                    enr_seg = pa_seg * net_revenue
                    seg_idx = np.argmax(enr_seg)
                    with seg_cols[idx_lv]:
                        st.metric(f"{lv_label}", f"{ev_name}={lv_val:.2f}")
                        st.metric("Optimal APR", f"{x_apr[seg_idx]:.2f}%")
                        st.metric("Max E[Net Rev]", fmt_dollar(enr_seg[seg_idx]))
                        st.caption(f"P(Accept)={pa_seg[seg_idx]:.3f} at this {ev_name} level.")

        # --- Chart 1: P(Accept) and Net Revenue vs APR (dual axis) ---
        st.divider()
        st.subheader("Charts")

        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(go.Scatter(x=x_apr, y=prob_accept, name="P(Accept)",
                                   line=dict(color="red")), secondary_y=False)
        fig1.add_trace(go.Scatter(x=x_apr, y=net_revenue, name="Net Revenue / month",
                                   line=dict(color="blue")), secondary_y=True)
        fig1.add_trace(go.Scatter(x=[opt_apr], y=[opt_prob], mode="markers",
                                   name="Optimal APR", marker=dict(size=14, symbol="star", color="gold"),
                                   showlegend=True), secondary_y=False)
        fig1.update_xaxes(title_text="APR (%)")
        fig1.update_yaxes(title_text="P(Accept)", secondary_y=False, color="red")
        fig1.update_yaxes(title_text="Net Revenue ($)", secondary_y=True, color="blue")
        fig1.update_layout(title="Acceptance Probability & Net Revenue vs. APR", height=450)
        st.plotly_chart(fig1, use_container_width=True)

        # --- Chart 2: Expected Net Revenue vs APR ---
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x_apr, y=expected_net_rev, name="Expected Net Revenue",
                                   line=dict(color="green", width=2)))
        fig2.add_trace(go.Scatter(x=[opt_apr], y=[opt_exp_rev], mode="markers+text",
                                   name=f"Max @ APR={opt_apr:.2f}%",
                                   text=[f"${opt_exp_rev:.2f}"], textposition="top center",
                                   marker=dict(size=14, symbol="star", color="gold")))
        fig2.update_layout(title="Expected Net Revenue per Quote vs. Quoted APR",
                           xaxis_title="APR (%)", yaxis_title="Expected Net Revenue ($)", height=450)
        st.plotly_chart(fig2, use_container_width=True)

        # --- Chart 3: Goodness of fit (if data was uploaded) ---
        if data_source == "Upload Excel / CSV" and uploaded_df is not None:
            import math
            apr_data = uploaded_df[apr_col_name].values
            accept_data = uploaded_df[accept_col_name].values

            minAPR = math.floor(apr_data.min())
            maxAPR = math.ceil(apr_data.max())
            bucket_size = 0.5
            cut_bins = np.arange(minAPR, maxAPR + bucket_size, bucket_size)
            mid_points = (cut_bins[:-1] + cut_bins[1:]) / 2

            bin_indices = np.digitize(apr_data, cut_bins) - 1
            bin_means = []
            bin_errors = []
            bin_counts = []
            for b in range(len(cut_bins) - 1):
                mask = bin_indices == b
                vals = accept_data[mask]
                if len(vals) > 0:
                    bin_means.append(np.mean(vals))
                    bin_errors.append(1.96 * np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
                    bin_counts.append(len(vals))
                else:
                    bin_means.append(None)
                    bin_errors.append(0)
                    bin_counts.append(0)

            fig3 = make_subplots(rows=1, cols=2,
                                  subplot_titles=("Logistic Regression Fit", "Quote Distribution & Expected Revenue"))

            # Scatter of raw data
            fig3.add_trace(go.Scatter(x=apr_data, y=accept_data, mode='markers',
                                       name='Accept/Reject', marker=dict(size=3, color='black', opacity=0.3)),
                           row=1, col=1)
            # Fitted curve
            fig3.add_trace(go.Scatter(x=x_apr, y=prob_accept, name='P(Accept) fit',
                                       line=dict(color='red', width=2)), row=1, col=1)
            # Bin means with error bars
            valid_mid = [mid_points[i] for i in range(len(mid_points)) if bin_means[i] is not None]
            valid_means = [bin_means[i] for i in range(len(bin_means)) if bin_means[i] is not None]
            valid_errors = [bin_errors[i] for i in range(len(bin_errors)) if bin_means[i] is not None]
            fig3.add_trace(go.Scatter(x=valid_mid, y=valid_means, mode='markers',
                                       name='Bin means (95% CI)',
                                       error_y=dict(type='data', array=valid_errors, visible=True),
                                       marker=dict(size=6, color='blue')), row=1, col=1)
            fig3.update_xaxes(title_text="APR (%)", row=1, col=1)
            fig3.update_yaxes(title_text="P(Accept)", row=1, col=1)

            # Expected revenue curve + quote count bars
            valid_counts_mid = [mid_points[i] for i in range(len(mid_points)) if bin_counts[i] > 0]
            valid_counts = [bin_counts[i] for i in range(len(bin_counts)) if bin_counts[i] > 0]
            fig3.add_trace(go.Bar(x=valid_counts_mid, y=valid_counts, name='# Quotes',
                                   marker_color='lightblue', width=0.4, opacity=0.7), row=1, col=2)
            fig3.add_trace(go.Scatter(x=x_apr, y=expected_net_rev, name='Exp Net Rev',
                                       line=dict(color='green', width=2)), row=1, col=2)
            fig3.update_xaxes(title_text="APR (%)", row=1, col=2)
            fig3.update_yaxes(title_text="Count / Expected Revenue ($)", row=1, col=2)

            fig3.update_layout(height=500, showlegend=True)
            st.plotly_chart(fig3, use_container_width=True)

        # --- Python printout ---
        extra_formula_str = "".join([f" + {beta_extra.get(v,0):.4f}*{v.replace(' ','_')}" for v in extra_vars])
        _code_nomis = [
            f"from sklearn.linear_model import LogisticRegression",
            f"from scipy.special import expit",
            f"import numpy as np",
            f"",
            f"# Logistic regression: P(Accept) = expit(Beta0 + Beta1*APR{' + ...' if extra_vars else ''})",
            f"Beta0 = {beta0_val:.4f}",
            f"Beta1 = {beta1_val:.4f}",
        ]
        for ev_name in extra_vars:
            _code_nomis.append(f"Beta_{ev_name.replace(' ','_')} = {beta_extra[ev_name]:.4f}")
            _code_nomis.append(f"avg_{ev_name.replace(' ','_')} = {extra_avgs[ev_name]:.4f}")
        _code_nomis += [
            f"",
            f"# Loan parameters",
            f"loan_amount = {loan_amount}",
            f"term = {term_months}  # months",
            f"cost_of_funds_rate = {cost_funds_rate}  # annual %",
            f"",
            f"def monthly_payment(rate_pct, amount, n):",
            f"    r = rate_pct / 1200",
            f"    return amount * r * (1+r)**n / ((1+r)**n - 1)",
            f"",
            f"cost = monthly_payment({cost_funds_rate}, {loan_amount}, {term_months})",
            f"",
            f"# Expected net revenue = P(Accept|APR) * (payment(APR) - cost)",
            f"apr_range = np.linspace({apr_min}, {apr_max}, 500)",
            f"logodds = {beta0_val:.4f} + {beta1_val:.4f} * apr_range{extra_formula_str}",
            f"exp_net_rev = expit(logodds) * (monthly_payment(apr_range, {loan_amount}, {term_months}) - cost)",
            f"opt_idx = np.argmax(exp_net_rev)",
        ]
        _results_nomis = [
            f"{'='*55}",
            f"Loan Pricing Optimization (Nomis)",
            f"{'='*55}",
            f"Beta0:              {beta0_val:.4f}",
            f"Beta1 (APR):        {beta1_val:.4f}",
        ]
        for ev_name in extra_vars:
            _results_nomis.append(f"Beta ({ev_name}): {beta_extra[ev_name]:.4f}  (avg={extra_avgs[ev_name]:.2f})")
        _results_nomis += [
            f"{'─'*55}",
            f"Optimal APR:        {opt_apr:.2f}%",
            f"P(Accept):          {opt_prob:.4f}",
            f"Net Revenue/month:  ${opt_net:.2f}",
            f"Max Exp Net Rev:    ${opt_exp_rev:.2f}",
            f"{'─'*55}",
            f"Monthly Payment:    ${revenue_per_month[opt_idx]:.2f}",
            f"Cost of Funds:      ${cost_per_month:.2f}",
        ]
        python_output("Loan Pricing Optimization", _code_nomis, _results_nomis)

        st.markdown("""
        **Key Insight**: Raising the APR increases the margin on each accepted loan but decreases the
        probability of acceptance. The **optimal APR** balances these two forces to maximize
        the expected revenue per quote. This is the classic **price–response** tradeoff in lending.
        """)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("Revenue Management & Pricing Optimizer — MSBAi Module V | Built with Streamlit + SciPy")

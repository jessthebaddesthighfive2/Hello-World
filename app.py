import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Derivative Detective", page_icon="📈", layout="wide")

x = sp.symbols("x", real=True)

st.title("Derivative Detective")
st.write(
    "Enter a first derivative, a second derivative, or both. The app reconstructs a possible "
    "original function, applies any initial conditions you provide, and compares the functions "
    "on vertically aligned graphs."
)

with st.sidebar:
    st.header("Function inputs")
    input_mode = st.radio(
        "What do you know?",
        ["First derivative f'(x)", "Second derivative f''(x)", "Both f'(x) and f''(x)"],
    )
    first_input = ""
    second_input = ""
    if input_mode in ("First derivative f'(x)", "Both f'(x) and f''(x)"):
        first_input = st.text_input("Enter f'(x)", value="3*x**2 - 4*x")
    if input_mode in ("Second derivative f''(x)", "Both f'(x) and f''(x)"):
        second_input = st.text_input("Enter f''(x)", value="6*x - 4")

    st.header("Initial conditions (optional)")
    use_point = st.checkbox("Specify a point on f: f(a) = b")
    if use_point:
        a_value = st.number_input("a", value=0.0, step=1.0, format="%.4f")
        b_value = st.number_input("b", value=0.0, step=1.0, format="%.4f")
    else:
        a_value = b_value = None

    use_slope = st.checkbox("Specify a slope: f'(a) = m")
    if use_slope:
        slope_a = st.number_input("Slope point a", value=0.0, step=1.0, format="%.4f")
        slope_m = st.number_input("Slope m", value=0.0, step=1.0, format="%.4f")
    else:
        slope_a = slope_m = None

    st.header("Graph settings")
    x_min = st.number_input("x minimum", value=-5.0, step=1.0)
    x_max = st.number_input("x maximum", value=5.0, step=1.0)
    show_guides = st.checkbox("Show dashed critical/inflection guides", value=True)
    show_roots = st.checkbox("Mark zeros of f'(x) and f''(x)", value=True)

st.caption("Use Python-style math syntax: x**2 for x², 3*x for 3x, sin(x), exp(x), log(x).")

def parse_expression(source):
    """Parse a math expression using a restricted set of common SymPy functions."""
    if not source.strip():
        raise ValueError("Please enter a function expression.")
    allowed = {
        "x": x, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
        "abs": sp.Abs, "pi": sp.pi, "E": sp.E, "Abs": sp.Abs,
    }
    expr = sp.sympify(source, locals=allowed)
    if expr.free_symbols - {x}:
        raise ValueError("Only the variable x is supported.")
    return sp.simplify(expr)

def integrate_once(expr):
    return sp.integrate(expr, x)

def solve_constant(expr, point_x, point_y):
    """Choose the additive constant so expr(point_x) equals point_y."""
    return sp.simplify(expr + (sp.sympify(point_y) - expr.subs(x, point_x)))

def real_roots(expr, lo, hi):
    """Find real roots in the viewing interval; numeric scan fallback handles many non-polynomials."""
    roots = []
    try:
        for root in sp.solve(sp.Eq(expr, 0), x):
            if root.is_real is True:
                value = float(sp.N(root))
                if lo <= value <= hi and np.isfinite(value):
                    if not any(abs(value - r) < 1e-4 for r in roots):
                        roots.append(value)
    except Exception:
        pass
    if not roots:
        try:
            fn = sp.lambdify(x, expr, "numpy")
            grid = np.linspace(lo, hi, 1600)
            vals = np.asarray(fn(grid), dtype=float)
            vals = np.broadcast_to(vals, grid.shape)
            for i in range(len(grid) - 1):
                if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]):
                    if vals[i] == 0 or vals[i] * vals[i + 1] < 0:
                        left, right = grid[i], grid[i + 1]
                        for _ in range(35):
                            mid = (left + right) / 2
                            fm = float(fn(mid))
                            fl = float(fn(left))
                            if fl == 0:
                                right = left
                                break
                            if fl * fm <= 0:
                                right = mid
                            else:
                                left = mid
                        root = (left + right) / 2
                        if not any(abs(root - r) < 1e-3 for r in roots):
                            roots.append(float(root))
        except Exception:
            pass
    return sorted(roots)

def numeric_values(expr, grid):
    fn = sp.lambdify(x, expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        values = np.asarray(fn(grid))
    if values.ndim == 0:
        values = np.full(grid.shape, values, dtype=float)
    values = np.asarray(values, dtype=float)
    values[~np.isfinite(values)] = np.nan
    return values

if x_min >= x_max:
    st.error("The x minimum must be less than the x maximum.")
    st.stop()

try:
    f1_input = parse_expression(first_input) if first_input.strip() else None
    f2_input = parse_expression(second_input) if second_input.strip() else None

    if f1_input is None and f2_input is None:
        st.error("Enter at least one derivative.")
        st.stop()

    # Reconstruct f and its derivatives, keeping symbolic integration constants explicit.
    if f1_input is not None:
        f = integrate_once(f1_input)
        f = sp.simplify(f + sp.Symbol("C"))
        if use_slope:
            # f'(a)=m is a consistency condition on the supplied derivative, not an integration constant.
            actual_slope = sp.simplify(f1_input.subs(x, slope_a))
            if not actual_slope.is_number or abs(float(sp.N(actual_slope)) - slope_m) > 1e-5:
                st.warning(
                    f"The entered derivative gives f'({slope_a:g}) = {sp.N(actual_slope)}, "
                    f"not {slope_m:g}. A vertical shift cannot change the slope."
                )
        if use_point:
            f = solve_constant(f, a_value, b_value)
        f1 = sp.diff(f, x)
        f2 = sp.diff(f1, x)
    else:
        # Integrating a second derivative twice introduces two constants.
        f1 = sp.integrate(f2_input, x) + sp.Symbol("C1")
        f = sp.integrate(f1, x) + sp.Symbol("C2")
        if use_slope:
            f1 = solve_constant(f1, slope_a, slope_m)
            f = sp.integrate(f1, x) + sp.Symbol("C2")
        if use_point:
            f = solve_constant(f, a_value, b_value)
        f1 = sp.diff(f, x)
        f2 = sp.diff(f1, x)

    # If both derivatives were supplied, flag disagreements instead of silently ignoring one.
    if f1_input is not None and f2_input is not None:
        mismatch = sp.simplify(sp.diff(f1_input, x) - f2_input)
        if mismatch != 0:
            st.warning(f"The inputs are not consistent: d/dx of your f'(x) minus your f''(x) is {mismatch}. The app reconstructs f from the first derivative.")
        else:
            st.success("The first and second derivative inputs are consistent.")

    st.subheader("Reconstructed functions")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Original function f(x)**")
    c1.latex(sp.latex(sp.simplify(f)))
    c2.markdown("**First derivative f'(x)**")
    c2.latex(sp.latex(sp.simplify(f1)))
    c3.markdown("**Second derivative f''(x)**")
    c3.latex(sp.latex(sp.simplify(f2)))

    if not use_point and not use_slope:
        st.info("Without an initial condition, the original function may include an unknown vertical shift (and, when starting from f'', an unknown linear term).")

    critical = real_roots(f1, x_min, x_max)
    inflections = real_roots(f2, x_min, x_max)

    left, right = st.columns(2)
    with left:
        st.markdown("**Critical points**")
        if critical:
            st.write(", ".join(f"x ≈ {v:.4g}" for v in critical))
        else:
            st.write("No zeros of f'(x) found in this interval.")
    with right:
        st.markdown("**Possible inflection points**")
        if inflections:
            st.write(", ".join(f"x ≈ {v:.4g}" for v in inflections))
        else:
            st.write("No zeros of f''(x) found in this interval.")
    st.caption("Inflection candidates are zeros of f''(x); a sign change in f'' should be checked to confirm an inflection point.")

    grid = np.linspace(float(x_min), float(x_max), 1200)
    functions = [f, f1, f2]
    labels = ["f(x) — original function", "f'(x) — first derivative", "f''(x) — second derivative"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.subplots_adjust(hspace=0.16)
    for index, (ax, expr, label) in enumerate(zip(axes, functions, labels)):
        vals = numeric_values(expr, grid)
        ax.plot(grid, vals, linewidth=2, label=label)
        ax.axhline(0, linewidth=0.8, alpha=0.6)
        ax.axvline(0, linewidth=0.8, alpha=0.6)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel(label)
        ax.legend(loc="upper right")
        if show_guides:
            for root in critical:
                ax.axvline(root, color="tab:red", linestyle="--", linewidth=1.1, alpha=0.8)
            for root in inflections:
                ax.axvline(root, color="tab:green", linestyle="--", linewidth=1.1, alpha=0.8)
        if show_roots and index in (1, 2):
            roots = critical if index == 1 else inflections
            if roots:
                try:
                    ys = [float(sp.N(expr.subs(x, r))) for r in roots]
                    ax.scatter(roots, ys, zorder=5, s=28)
                except Exception:
                    pass
    axes[-1].set_xlabel("x")
    if show_guides:
        axes[0].text(
            0.01, 0.03,
            "Red dashed: f'(x)=0 (critical candidates)   |   Green dashed: f''(x)=0 (inflection candidates)",
            transform=axes[0].transAxes, fontsize=8, va="bottom"
        )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("How to read the comparison")
    st.write(
        "The three graphs share the same x-axis and are stacked vertically. Red dashed lines mark zeros of "
        "the first derivative, where the original function may have a local maximum or minimum. Green dashed "
        "lines mark zeros of the second derivative, which are candidates for inflection points."
    )

except Exception as exc:
    st.error(f"Could not process the expression: {exc}")
    st.write("Check your syntax. Examples: `x**2`, `3*x**2 - 4*x`, `sin(x)`, `exp(x)`, `sqrt(x)`.")

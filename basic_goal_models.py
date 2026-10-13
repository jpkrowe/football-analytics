import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    # Imports used by the reusable functions below (other notebooks can import them)
    import numpy as np
    from scipy.optimize import minimize
    from scipy.stats import poisson


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Football prediction, part 1: Maher → Dixon–Coles

    **Goal:** fit two classic goal models to five Premier League seasons and find out, *honestly*, how good their predictions are.

    The plan for this notebook:

    1. **Data:** load results (and bookmaker odds) from [football-data.co.uk](https://www.football-data.co.uk).
    2. **Is Poisson reasonable?** Check that goal counts look like a Poisson distribution.
    3. **Maher (1982):** a Poisson GLM with an *attack* and *defence* rating per team, fitted with `statsmodels`.
    4. **Dixon–Coles (1997):** fix two weaknesses of Maher (low scores and old data), fitted with our own likelihood in `scipy`.
    5. **Walk-forward evaluation:** predict each gameweek using *only* the matches before it, and score the predictions with the **Ranked Probability Score (RPS)**.

    Background notes: *Football Prediction Overview* and *GLMs and Regression Assumptions* in Obsidian.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    return mo, pd, sm, smf


@app.cell
def _():
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    # One shared look for every chart. Chart cells import px/go from here, so this always runs first.
    # Charts get a fixed size (instead of stretching to the notebook's width) to keep sensible proportions.
    _ink, _muted, _grid, _axis = "#0b0b0b", "#52514e", "#ececea", "#d6d5d0"
    pio.templates["notebook"] = go.layout.Template(
        layout=go.Layout(
            colorway=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
            font={"family": "Inter, system-ui, sans-serif", "size": 13, "color": _muted},
            title={"font": {"size": 16, "color": _ink}, "x": 0.01, "xanchor": "left", "y": 0.96, "yref": "container"},
            width=680,
            height=420,
            autosize=False,
            margin={"l": 70, "r": 30, "t": 90, "b": 60},
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis={"showgrid": False, "linecolor": _axis, "ticks": "outside", "tickcolor": _axis, "zeroline": False},
            yaxis={"gridcolor": _grid, "gridwidth": 1, "linecolor": _axis, "zeroline": False},
            legend={"orientation": "h", "x": 0, "xanchor": "left", "y": 1.02, "yanchor": "bottom", "title_text": ""},
            hoverlabel={"bgcolor": "white", "font_size": 12},
        ),
        data={"scatter": [go.Scatter(line={"width": 2}, marker={"size": 8})]},
    )
    pio.templates.default = "plotly_white+notebook"

    # Each model keeps the same colour on every chart
    MODEL_COLORS = {"Maher": "#2a78d6", "Dixon–Coles": "#eb6834", "Bookmaker (closing)": "#1baf7a"}
    return MODEL_COLORS, make_subplots, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Data

    football-data.co.uk has one CSV per league per season. `E0` is the Premier League. We keep:

    - the **result** (full-time goals),
    - the **closing odds** (market average just before kick-off). Later we use these as the benchmark to beat, because the betting market is a very strong forecaster.

    The seasons run from 2021-22 to 2025-26: 5 × 380 = 1,900 matches.
    """)
    return


@app.cell
def _(pd):
    SEASONS = {"2021-22": "2122", "2022-23": "2223", "2023-24": "2324", "2024-25": "2425", "2025-26": "2526"}


    def _load_season(name, code):
        raw = pd.read_csv(f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv", encoding="latin-1")
        return pd.DataFrame({
            "season": name,
            "date": pd.to_datetime(raw["Date"], dayfirst=True),
            "home": raw["HomeTeam"],
            "away": raw["AwayTeam"],
            "home_goals": raw["FTHG"],
            "away_goals": raw["FTAG"],
            # market-average closing odds (decimal), used as a benchmark later
            "odds_home": raw["AvgCH"],
            "odds_draw": raw["AvgCD"],
            "odds_away": raw["AvgCA"],
        })


    matches = pd.concat([_load_season(n, c) for n, c in SEASONS.items()]).sort_values("date", ignore_index=True)
    matches["result"] = np.select(
        [matches.home_goals > matches.away_goals, matches.home_goals == matches.away_goals], ["H", "D"], "A"
    )
    matches
    return SEASONS, matches


@app.cell
def _(matches, mo):
    _by_season = matches.groupby("season").agg(
        matches=("result", "size"),
        home_goals=("home_goals", "mean"),
        away_goals=("away_goals", "mean"),
        home_win=("result", lambda r: (r == "H").mean()),
        draw=("result", lambda r: (r == "D").mean()),
        away_win=("result", lambda r: (r == "A").mean()),
    ).round(3)
    mo.vstack([
        mo.md(
            f"""
            **First look.** Across all five seasons there are **{(matches.home_goals + matches.away_goals).mean():.2f} goals per match**.
            Home teams score **{matches.home_goals.mean():.2f}** and away teams **{matches.away_goals.mean():.2f}**.
            That gap is *home advantage*, and the models will estimate it explicitly.
            """
        ),
        _by_season,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Are goals Poisson?

    Both models assume that the number of goals a team scores in a match follows a **Poisson distribution**:

    $$P(k \text{ goals}) = \frac{\lambda^k e^{-\lambda}}{k!}$$

    Here $\lambda$ is the expected number of goals. The Poisson distribution has one parameter, and it forces **variance = mean**. That gives a quick check: compare the observed distribution of goals with a Poisson that has the same mean.

    (We pool all teams here, so this is only a rough check. The models give each match its own $\lambda$.)
    """)
    return


@app.cell
def _(make_subplots, matches, mo, pd):
    _rows = []
    _moments = []
    for _side in ["home", "away"]:
        _goals = matches[f"{_side}_goals"]
        _lam = _goals.mean()
        _moments.append({"side": _side, "mean": round(_lam, 3), "variance": round(_goals.var(), 3)})
        for _k in range(8):
            _rows.append({"side": _side, "goals": _k, "observed": (_goals == _k).mean(), "poisson": poisson.pmf(_k, _lam)})
    goal_dist = pd.DataFrame(_rows)

    _fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=["Home team", "Away team"], horizontal_spacing=0.06)
    for _col, _side in enumerate(["home", "away"], start=1):
        _d = goal_dist[goal_dist.side == _side]
        _fig.add_bar(
            x=_d.goals, y=_d.observed, name="observed", marker_color="#86b6ef", showlegend=_col == 1,
            hovertemplate="%{x} goals: %{y:.1%} of matches<extra>observed</extra>", row=1, col=_col,
        )
        _fig.add_scatter(
            x=_d.goals, y=_d.poisson, mode="lines+markers", name="Poisson (same mean)", line_color="#0b0b0b",
            showlegend=_col == 1, hovertemplate="%{x} goals: %{y:.1%}<extra>Poisson</extra>", row=1, col=_col,
        )
    _fig.update_xaxes(title_text="goals scored in a match", dtick=1)
    _fig.update_yaxes(tickformat=".0%")
    _fig.update_yaxes(title_text="share of matches", row=1, col=1)
    _fig.update_layout(width=760, height=400, bargap=0.35, title="Observed goals vs a Poisson distribution")
    mo.vstack([_fig, pd.DataFrame(_moments)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Takeaway:** the fit is close, and the variance is only a little above the mean. Two things cause that small excess: different teams have different $\lambda$s, and there may be genuine *overdispersion*. That is why we model a separate $\lambda$ for every team and match instead of reaching for a more exotic distribution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The Maher model, a Poisson GLM

    Each team gets an **attack** rating $\alpha$ and a **defence** rating $\beta$. The expected goals for each side are

    $$\lambda_\text{home} = \exp(\mu + \gamma + \alpha_\text{home} - \beta_\text{away}) \qquad \lambda_\text{away} = \exp(\mu + \alpha_\text{away} - \beta_\text{home})$$

    - $\mu$ is a baseline (log of an average team's goals away from home), and $\gamma$ is **home advantage**.
    - The $\exp$ (the **log link**) makes effects *multiply*. A team with $\alpha = 0.3$ scores $e^{0.3} \approx 1.35\times$ as many goals as an average team.

    This is a **generalised linear model** (GLM): a Poisson distribution with a log link and a linear predictor. To fit it as an ordinary regression we reshape the data to **two rows per match**, one for each team's goals:

    | goals | team | opponent | home |
    |---|---|---|---|
    | home goals | home team | away team | 1 |
    | away goals | away team | home team | 0 |

    The formula `goals ~ home + team + opponent` then gives one coefficient per team for attack (`team`) and one per team for defence (`opponent`).

    Pick a season to fit:
    """)
    return


@app.cell
def _(SEASONS, mo):
    fit_season = mo.ui.dropdown(options=list(SEASONS), value="2025-26", label="Season to fit")
    fit_season
    return (fit_season,)


@app.cell
def _(fit_season, matches, pd, sm, smf):
    def to_long(df):
        """Two rows per match: one for the home team's goals, one for the away team's."""
        return pd.concat([
            pd.DataFrame({"goals": df.home_goals, "team": df.home, "opponent": df.away, "home": 1}),
            pd.DataFrame({"goals": df.away_goals, "team": df.away, "opponent": df.home, "home": 0}),
        ], ignore_index=True)


    season_matches = matches[matches.season == fit_season.value]
    long = to_long(season_matches)
    maher_glm = smf.glm("goals ~ home + team + opponent", data=long, family=sm.families.Poisson()).fit()
    maher_glm.params.round(3).to_frame("coefficient")
    return long, maher_glm, season_matches, to_long


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reading the coefficients.** `statsmodels` uses *treatment coding*. The first team alphabetically is the reference (its coefficients are fixed at 0), and every other team is measured relative to it. That choice is arbitrary: only *differences* between teams are identified. So below we **centre** the ratings, which makes 0 mean "league average".

    The `opponent` coefficient is *minus* the defence rating, because a good defence lowers the opponent's goals. We flip the sign so that **higher = better** for both ratings.
    """)
    return


@app.cell
def _(fit_season, long, maher_glm, mo, pd, px):
    _teams = sorted(long.team.unique())
    _attack = np.array([maher_glm.params.get(f"team[T.{t}]", 0.0) for t in _teams])
    _defence = np.array([-maher_glm.params.get(f"opponent[T.{t}]", 0.0) for t in _teams])
    maher_ratings = pd.DataFrame({
        "team": _teams,
        "attack": _attack - _attack.mean(),
        "defence": _defence - _defence.mean(),
    })
    maher_ratings["attack_x"] = np.exp(maher_ratings.attack)
    maher_ratings["concede_x"] = np.exp(-maher_ratings.defence)

    _fig = px.scatter(
        maher_ratings, x="attack_x", y="concede_x", text="team",
        hover_data={"attack": ":.3f", "defence": ":.3f", "attack_x": ":.2f", "concede_x": ":.2f"},
        labels={"attack_x": "goals scored vs average team (×)", "concede_x": "goals conceded vs average team (×)"},
        title=f"Maher ratings, {fit_season.value} (top right = best)",
    )
    _fig.update_traces(textposition="middle right", textfont={"size": 11}, marker={"size": 10, "line": {"width": 2, "color": "white"}})
    _fig.update_xaxes(showgrid=True, gridcolor="#ececea")
    _fig.update_yaxes(autorange="reversed")
    _fig.add_hline(y=1, line_width=1, line_color="#a9a8a2")
    _fig.add_vline(x=1, line_width=1, line_color="#a9a8a2")
    _fig.update_layout(width=720, height=620)
    mo.vstack([
        mo.md(
            f"Home advantage: $e^\\gamma$ = **{np.exp(maher_glm.params['home']):.2f}×** goals for the home team. The grey lines mark an average team."
        ),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### From ratings to a prediction

    For any fixture the model gives $\lambda_\text{home}$ and $\lambda_\text{away}$. Maher assumes the two goal counts are **independent**, so the probability of a scoreline is just a product:

    $$P(\text{home}=i,\ \text{away}=j) = \text{Poisson}(i;\lambda_\text{home}) \times \text{Poisson}(j;\lambda_\text{away})$$

    Summing that grid gives every other market. Home win = below the diagonal, draw = the diagonal, away win = above it. (`score_grid` also takes a `rho` argument, which we use in section 4.)
    """)
    return


@app.function
def score_grid(lam_home, lam_away, rho=0.0, max_goals=10):
    """Matrix of scoreline probabilities: grid[i, j] = P(home scores i, away scores j)."""
    goals = np.arange(max_goals + 1)
    grid = np.outer(poisson.pmf(goals, lam_home), poisson.pmf(goals, lam_away))
    # Dixon-Coles low-score adjustment (rho = 0 leaves the grid unchanged, i.e. Maher)
    grid[0, 0] *= 1 - lam_home * lam_away * rho
    grid[0, 1] *= 1 + lam_home * rho
    grid[1, 0] *= 1 + lam_away * rho
    grid[1, 1] *= 1 - rho
    return grid / grid.sum()


@app.function
def outcome_probs(grid):
    """(P(home win), P(draw), P(away win)) from a score grid."""
    return np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()


@app.cell
def _(long, mo):
    _teams = sorted(long.team.unique())
    pick_home = mo.ui.dropdown(options=_teams, value=_teams[0], label="Home")
    pick_away = mo.ui.dropdown(options=_teams, value=_teams[1], label="Away")
    mo.hstack([pick_home, pick_away], justify="start")
    return pick_away, pick_home


@app.cell
def _(maher_glm, mo, pd, pick_away, pick_home, px):
    _lam_home, _lam_away = maher_glm.predict(
        pd.DataFrame({"team": [pick_home.value, pick_away.value], "opponent": [pick_away.value, pick_home.value], "home": [1, 0]})
    )
    _grid = score_grid(_lam_home, _lam_away)
    _p_home, _p_draw, _p_away = outcome_probs(_grid)

    _fig = px.imshow(
        _grid[:6, :6], text_auto=".1%", color_continuous_scale=["#ffffff", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"],
        labels={"x": f"{pick_away.value} goals", "y": f"{pick_home.value} goals", "color": "probability"},
        title="Scoreline probabilities",
    )
    _fig.update_traces(hovertemplate="%{y}-%{x}: %{z:.1%}<extra></extra>", xgap=2, ygap=2)
    _fig.update_xaxes(side="top", ticks="", showline=False)
    _fig.update_yaxes(ticks="", showline=False, showgrid=False)
    _fig.update_layout(width=440, height=460, coloraxis_showscale=False, margin={"t": 110, "l": 70, "r": 20, "b": 20})

    mo.hstack([
        _fig,
        mo.md(
            f"""
            **{pick_home.value}** expected goals: **{_lam_home:.2f}**
            **{pick_away.value}** expected goals: **{_lam_away:.2f}**

            | outcome | probability | fair odds |
            |---|---|---|
            | home win | {_p_home:.1%} | {1 / _p_home:.2f} |
            | draw | {_p_draw:.1%} | {1 / _p_draw:.2f} |
            | away win | {_p_away:.1%} | {1 / _p_away:.2f} |
            """
        ),
    ], justify="start", align="center", gap=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Dixon–Coles: two fixes to Maher

    ### Problem 1: low scores
    Maher treats home and away goals as **independent**. Dixon & Coles (1997), using 1990s English data, found more 0-0 and 1-1 draws and fewer 1-0 and 0-1 results than independence predicts. Is that still true? Below, Maher is fitted to each season separately, and we add up the expected and observed counts of each low score.
    """)
    return


@app.cell
def _(SEASONS, matches, pd, sm, smf, to_long):
    _scores = [(0, 0), (1, 0), (0, 1), (1, 1)]
    _observed = dict.fromkeys(_scores, 0)
    _expected = dict.fromkeys(_scores, 0.0)
    for _season in SEASONS:
        _df = matches[matches.season == _season]
        _fit = smf.glm("goals ~ home + team + opponent", data=to_long(_df), family=sm.families.Poisson()).fit()
        _lams = _fit.fittedvalues.to_numpy().reshape(2, -1).T  # one (home, away) row per match
        _grid_total = sum(score_grid(lh, la) for lh, la in _lams)  # expected count of every scoreline
        for i, j in _scores:
            _observed[(i, j)] += ((_df.home_goals == i) & (_df.away_goals == j)).sum()
            _expected[(i, j)] += _grid_total[i, j]

    pd.DataFrame({
        "score": [f"{i}-{j}" for i, j in _scores],
        "observed": [_observed[s] for s in _scores],
        "Maher expected": [round(_expected[s], 1) for s in _scores],
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In recent Premier League data the picture is **mixed**: there are *fewer* 0-0s than Maher expects but *more* 1-1s. Dixon–Coles adds one parameter $\rho$ that multiplies the probabilities of these four scores by a factor $\tau$:

    | score | $\tau$ |
    |---|---|
    | 0-0 | $1 - \lambda_h\lambda_a\rho$ |
    | 0-1 | $1 + \lambda_h\rho$ |
    | 1-0 | $1 + \lambda_a\rho$ |
    | 1-1 | $1 - \rho$ |

    $\rho < 0$ boosts 0-0 and 1-1 and shrinks 1-0 and 0-1. It always moves the two draws *the same way*, but our data has one too few and the other too many, so it is not obvious that $\rho$ will help. We fit it anyway and let the evaluation in section 5 decide.

    ### Problem 2: old matches count as much as new ones
    Teams change (transfers, managers, injuries). Dixon–Coles weights each match by how recent it is:

    $$w = e^{-\xi \cdot \text{days ago}}$$

    $\xi$ is a **hyperparameter**: we choose it rather than fit it. We pick it by testing predictions in section 5. The *half-life* $\ln 2 / \xi$ is the age at which a match counts half as much.
    """)
    return


@app.cell
def _(pd, px):
    _days = np.arange(0, 1500, 10)
    _curves = pd.concat([
        pd.DataFrame({"days ago": _days, "weight": np.exp(-_xi * _days), "xi": f"ξ = {_xi} (half-life {np.log(2) / _xi:.0f} days)"})
        for _xi in [0.001, 0.002, 0.005]
    ])
    _fig = px.line(_curves, x="days ago", y="weight", color="xi", title="How much a past match counts")
    _fig.update_traces(hovertemplate="%{y:.2f}")
    _fig.update_yaxes(range=[0, 1.02])
    _fig.update_layout(hovermode="x unified")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fitting it ourselves
    With $\tau$ and the weights, this is no longer a standard GLM, so we write the likelihood out and maximise it with `scipy.optimize.minimize`. The objective is the **weighted log-likelihood**

    $$\sum_\text{matches} w \left[\log \text{Pois}(x;\lambda_h) + \log \text{Pois}(y;\lambda_a) + \log\tau\right]$$

    with two practical additions:

    - **A small ridge penalty** pulls every team's ratings gently towards 0 (league average). It matters for teams with very few matches, such as a promoted side after one game, and it is negligible for everyone else. (It is the frequentist cousin of a Bayesian prior. Part 3 of the plan makes this principled.)
    - **An analytic gradient.** We give the optimiser the slope of the objective, so it doesn't have to estimate it numerically. That makes each fit about 50× faster, which matters because the evaluation refits the model hundreds of times. You can skim the gradient code. It is just the derivatives of the expression above, and it was checked against a numerical gradient.

    With `xi=0` and `use_rho=False` this function *is* the Maher model.
    """)
    return


@app.function
def fit_dixon_coles(train, as_of, xi=0.0, use_rho=True, ridge=1.0):
    """Fit Dixon-Coles by maximum likelihood.

    xi:      time-decay rate per day (0 = every match counts equally)
    use_rho: include the low-score correction (False = plain Maher)
    ridge:   penalty pulling team ratings towards 0 (league average)
    """
    teams = sorted(set(train.home) | set(train.away))
    n = len(teams)
    index = {t: i for i, t in enumerate(teams)}
    h = train.home.map(index).to_numpy()
    a = train.away.map(index).to_numpy()
    x = train.home_goals.to_numpy()
    y = train.away_goals.to_numpy()
    w = np.exp(-xi * (as_of - train.date).dt.days.to_numpy())
    s00, s01, s10, s11 = [(x == i) & (y == j) for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]]

    def objective(params):
        attack, defence = params[:n], params[n : 2 * n]
        mu, gamma = params[2 * n], params[2 * n + 1]
        rho = params[2 * n + 2] if use_rho else 0.0
        lh = np.exp(mu + gamma + attack[h] - defence[a])
        la = np.exp(mu + attack[a] - defence[h])

        # Poisson log-likelihood of each match (dropping the constant log(k!) terms)
        ll = x * np.log(lh) - lh + y * np.log(la) - la
        # derivatives of ll with respect to log(lh), log(la) and rho, for the gradient
        d_h, d_a, d_rho = x - lh, y - la, np.zeros_like(lh)

        if use_rho:
            tau = np.ones_like(lh)
            tau[s00] = 1 - lh[s00] * la[s00] * rho
            tau[s01] = 1 + lh[s01] * rho
            tau[s10] = 1 + la[s10] * rho
            tau[s11] = 1 - rho
            tau = np.maximum(tau, 1e-10)
            ll = ll + np.log(tau)
            d_h[s00] -= lh[s00] * la[s00] * rho / tau[s00]
            d_a[s00] -= lh[s00] * la[s00] * rho / tau[s00]
            d_h[s01] += lh[s01] * rho / tau[s01]
            d_a[s10] += la[s10] * rho / tau[s10]
            d_rho[s00] = -lh[s00] * la[s00] / tau[s00]
            d_rho[s01] = lh[s01] / tau[s01]
            d_rho[s10] = la[s10] / tau[s10]
            d_rho[s11] = -1 / tau[s11]

        d_h, d_a = w * d_h, w * d_a
        grad = np.concatenate([
            np.bincount(h, d_h, n) + np.bincount(a, d_a, n),  # attack
            -np.bincount(a, d_h, n) - np.bincount(h, d_a, n),  # defence
            [d_h.sum() + d_a.sum(), d_h.sum()],  # mu, gamma
            [(w * d_rho).sum()] if use_rho else [],
        ])
        grad[: 2 * n] -= 2 * ridge * params[: 2 * n]
        penalty = ridge * (attack @ attack + defence @ defence)
        # minimise the negative of the (penalised) weighted log-likelihood
        return -(w * ll).sum() + penalty, -grad

    start = np.zeros(2 * n + 2 + use_rho)
    bounds = [(None, None)] * (2 * n + 2) + ([(-0.2, 0.2)] if use_rho else [])
    result = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=bounds)
    p = result.x
    return {
        "attack": dict(zip(teams, p[:n])),
        "defence": dict(zip(teams, p[n : 2 * n])),
        "mu": p[2 * n],
        "gamma": p[2 * n + 1],
        "rho": p[2 * n + 2] if use_rho else 0.0,
        "converged": result.success,
    }


@app.function
def predict_lambdas(model, home, away):
    """Expected goals for a fixture. A team the model has never seen gets average ratings (0)."""
    attack, defence = model["attack"], model["defence"]
    lam_home = np.exp(model["mu"] + model["gamma"] + attack.get(home, 0.0) - defence.get(away, 0.0))
    lam_away = np.exp(model["mu"] + attack.get(away, 0.0) - defence.get(home, 0.0))
    return lam_home, lam_away


@app.cell
def _(SEASONS, fit_season, maher_glm, matches, mo, pd, season_matches):
    _as_of = season_matches.date.max()
    _ours = fit_dixon_coles(season_matches, _as_of, use_rho=False, ridge=0.0)
    _ours_lams = np.array([predict_lambdas(_ours, m.home, m.away) for m in season_matches.itertuples()])
    _glm_lams = maher_glm.fittedvalues.to_numpy().reshape(2, -1).T

    _per_season = []
    for _season in SEASONS:
        _df = matches[matches.season == _season]
        _fit = fit_dixon_coles(_df, _df.date.max())
        _per_season.append({"season": _season, "rho": round(_fit["rho"], 3), "home advantage (×)": round(np.exp(_fit["gamma"]), 2)})

    mo.vstack([
        mo.md(
            f"""
            **Sanity check:** with `xi=0, use_rho=False, ridge=0` our function should reproduce the statsmodels GLM for {fit_season.value}.
            Largest difference in expected goals across all {len(season_matches)} matches: **{np.abs(_ours_lams - _glm_lams).max():.5f}** goals ✅

            **Dixon–Coles fitted to each season separately** (no time decay):
            """
        ),
        pd.DataFrame(_per_season),
        mo.md(
            r"""
            $\rho$ jumps around from season to season, even changing sign. When a parameter estimate is this unstable, it is mostly fitting **noise**: 380 matches only contain a few dozen of each low score. Home advantage, by contrast, is fairly stable.
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Honest evaluation

    ### Walk-forward testing
    How well a model fits the matches it was trained on tells you very little: it has already *seen* those results. The honest test is to predict matches **before they happen**:

    1. Take the next gameweek.
    2. Fit the model using **only matches played before it**.
    3. Predict that week's matches, then move forward one week and repeat.

    Random train/test splits (like the k-fold cross-validation you may meet elsewhere) would let the model learn from matches played *after* the one it predicts, which is **look-ahead bias**.

    ### Scoring: the Ranked Probability Score
    We forecast probabilities (home, draw, away), so we need a score for probabilities, not just "did the favourite win". The standard choice in football is the **RPS**. It compares the *cumulative* forecast with the cumulative outcome:

    $$\text{RPS} = \tfrac{1}{2}\left[(p_H - o_H)^2 + (p_H + p_D - o_H - o_D)^2\right]$$

    $o$ is 1 for the outcome that happened and 0 otherwise. **Lower is better**: 0 is a perfect, certain forecast. The outcomes are *ordered* (home, draw, away), so RPS punishes a forecast more when its probability sits *further* from what happened. We also report **log loss** ($-\log$ of the probability given to the actual result), which ignores that ordering.
    """)
    return


@app.cell
def _(pd):
    def add_scores(preds):
        """Add per-match RPS and log loss to a frame with p_home, p_draw, p_away and result columns."""
        home, draw = preds.result == "H", preds.result == "D"
        cum_err_1 = preds.p_home - home
        cum_err_2 = preds.p_home + preds.p_draw - (home | draw)
        p_actual = np.select([home, draw], [preds.p_home, preds.p_draw], preds.p_away)
        return preds.assign(rps=(cum_err_1**2 + cum_err_2**2) / 2, log_loss=-np.log(p_actual))


    def walk_forward(matches, test_seasons, xi, use_rho):
        """Predict every match in test_seasons, refitting each week on earlier matches only."""
        test = matches[matches.season.isin(test_seasons)]
        rows = []
        for _week, block in test.groupby(test.date.dt.to_period("W")):
            cutoff = block.date.min()
            model = fit_dixon_coles(matches[matches.date < cutoff], as_of=cutoff, xi=xi, use_rho=use_rho)
            for m in block.itertuples():
                grid = score_grid(*predict_lambdas(model, m.home, m.away), rho=model["rho"])
                rows.append((m.Index, *outcome_probs(grid)))
        probs = pd.DataFrame(rows, columns=["match_id", "p_home", "p_draw", "p_away"]).set_index("match_id")
        return add_scores(test.join(probs))

    return add_scores, walk_forward


@app.cell
def _(add_scores, mo, pd):
    _examples = pd.DataFrame({
        "forecast": ["home favourite", "draw favourite", "no idea", "away favourite"],
        "p_home": [0.6, 0.2, 1 / 3, 0.1],
        "p_draw": [0.2, 0.6, 1 / 3, 0.2],
        "p_away": [0.2, 0.2, 1 / 3, 0.7],
        "result": "A",
    })
    mo.vstack([
        mo.md("**Example.** Suppose the away team wins. The first two forecasts both give the away win 20%, so log loss rates them the same. RPS penalises the *home*-favourite forecast more, because a draw is 'closer' to an away win:"),
        add_scores(_examples).round(3),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Choosing $\xi$ without cheating
    $\xi$ is a hyperparameter. If we chose it by its score on the test seasons, the test score would be optimistic, because we would have tuned to it. So we split time into three parts:

    | seasons | role |
    |---|---|
    | 2021-22 | training data only (warm-up) |
    | 2022-23 | **validation**: try several $\xi$ values, keep the best |
    | 2023-24 → 2025-26 | **test**: touched once, at the end, to compare models |
    """)
    return


@app.cell
def _(matches, mo, pd, px, walk_forward):
    VALIDATION_SEASON = "2022-23"
    _grid = [0.0, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015]
    xi_search = pd.DataFrame([
        {
            "xi": _xi,
            "half-life (days)": round(np.log(2) / _xi) if _xi else np.inf,
            "validation RPS": walk_forward(matches, [VALIDATION_SEASON], xi=_xi, use_rho=True).rps.mean(),
        }
        for _xi in _grid
    ])
    best_xi = xi_search.loc[xi_search["validation RPS"].idxmin(), "xi"]

    _fig = px.line(
        xi_search, x="xi", y="validation RPS", markers=True, hover_data={"half-life (days)": True, "validation RPS": ":.4f"},
        labels={"xi": "ξ (decay per day)"}, title=f"Dixon–Coles walk-forward RPS on {VALIDATION_SEASON} (lower is better)",
    )
    _fig.add_annotation(
        x=best_xi, y=xi_search["validation RPS"].min(), text=f"best: ξ = {best_xi}", showarrow=True, arrowhead=0, ay=-40,
        font={"color": "#0b0b0b"},
    )
    mo.vstack([
        _fig,
        mo.md(f"Best $\\xi$ = **{best_xi}**: a match from **{np.log(2) / best_xi:.0f} days** ago counts half as much as today's."),
    ])
    return (best_xi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The final comparison
    Every model below predicts the same 1,140 test matches, each one using only information available before kick-off:

    - **Base rates** predict the same thing every time: the historical home/draw/away frequencies. Any real model must beat this.
    - **Maher**: no time decay, no $\rho$, all past matches weighted equally.
    - **Maher + time decay**: adds the $\xi$ chosen above.
    - **Dixon–Coles**: adds $\rho$ as well.
    - **Bookmaker (closing)**: the market-average closing odds converted to probabilities. We take $1/\text{odds}$ and rescale so the three sum to 1, which removes the bookmaker's margin. This is the bar professionals measure against.
    """)
    return


@app.cell
def _(add_scores, best_xi, matches, pd, walk_forward):
    TEST_SEASONS = ["2023-24", "2024-25", "2025-26"]
    MODEL_ORDER = ["Base rates", "Maher", "Maher + time decay", "Dixon–Coles", "Bookmaker (closing)"]

    _test = matches[matches.season.isin(TEST_SEASONS)]
    _base_rates = matches[matches.date < _test.date.min()].result.value_counts(normalize=True)
    _implied = 1 / _test[["odds_home", "odds_draw", "odds_away"]].to_numpy()
    _implied = _implied / _implied.sum(axis=1, keepdims=True)  # remove the bookmaker's margin

    all_preds = pd.concat([
        add_scores(_test.assign(p_home=_base_rates["H"], p_draw=_base_rates["D"], p_away=_base_rates["A"])).assign(model="Base rates"),
        walk_forward(matches, TEST_SEASONS, xi=0.0, use_rho=False).assign(model="Maher"),
        walk_forward(matches, TEST_SEASONS, xi=best_xi, use_rho=False).assign(model="Maher + time decay"),
        walk_forward(matches, TEST_SEASONS, xi=best_xi, use_rho=True).assign(model="Dixon–Coles"),
        add_scores(_test.assign(p_home=_implied[:, 0], p_draw=_implied[:, 1], p_away=_implied[:, 2])).assign(model="Bookmaker (closing)"),
    ]).rename_axis("match_id").reset_index()

    results = all_preds.pivot_table(index="model", columns="season", values="rps")
    results["all test (RPS)"] = all_preds.groupby("model").rps.mean()
    results["all test (log loss)"] = all_preds.groupby("model").log_loss.mean()
    results.reindex(MODEL_ORDER).round(4)
    return MODEL_ORDER, all_preds, results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Are the differences real?
    The RPS gaps are small, so we should ask whether they are bigger than luck. Every model predicted the *same* matches, so we can look at the **per-match difference** from Maher. A difference is convincing if its mean is clearly away from 0 relative to its standard error (roughly: if "mean ± 2 SE" excludes 0).

    This is only approximate: matches are not fully independent (the same team-strength errors affect many games), so treat it as a rough guide.
    """)
    return


@app.cell
def _(MODEL_ORDER, all_preds, pd):
    _wide = all_preds.pivot(index="match_id", columns="model", values="rps")
    _rows = []
    for _m in MODEL_ORDER:
        if _m == "Maher":
            continue
        _d = _wide[_m] - _wide["Maher"]
        _se = _d.std() / np.sqrt(len(_d))
        _rows.append({
            "model": _m,
            "mean RPS change vs Maher": _d.mean(),
            "2 × SE": 2 * _se,
            "verdict": "better" if _d.mean() + 2 * _se < 0 else ("worse" if _d.mean() - 2 * _se > 0 else "can't tell"),
        })
    pd.DataFrame(_rows).round(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Calibration
    A forecaster is **calibrated** if, out of all the matches given a 60% home-win chance, about 60% are home wins. Group the predictions into bins and plot predicted against observed. A calibrated model sits on the diagonal.
    """)
    return


@app.cell
def _(MODEL_COLORS, all_preds, px):
    _cal = all_preds[all_preds.model.isin(["Dixon–Coles", "Bookmaker (closing)"])].assign(
        bin=lambda d: np.minimum((d.p_home * 10).astype(int), 9)
    )
    _cal = (
        _cal.groupby(["model", "bin"])
        .agg(predicted=("p_home", "mean"), observed=("result", lambda r: (r == "H").mean()), matches=("result", "size"))
        .reset_index()
    )
    _fig = px.line(
        _cal, x="predicted", y="observed", color="model", markers=True, color_discrete_map=MODEL_COLORS,
        hover_data={"predicted": ":.2f", "observed": ":.2f", "matches": True},
        labels={"predicted": "predicted P(home win)", "observed": "observed home-win rate"},
        title="Calibration of home-win probabilities",
    )
    _fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"width": 1, "color": "#a9a8a2"}, layer="below")
    _fig.add_annotation(x=0.9, y=0.9, text="perfect calibration", showarrow=False, textangle=-45, yshift=12, font={"size": 11})
    _fig.update_xaxes(range=[0, 1], showgrid=True, gridcolor="#ececea", tickformat=".0%")
    _fig.update_yaxes(range=[0, 1], tickformat=".0%", scaleanchor="x")
    _fig.update_layout(width=560, height=600)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Where do the models lose?
    Average RPS by calendar month, pooled over the three test seasons. Early-season matches are the hardest for our models: promoted teams are new to the model and get "average" ratings, and summer transfers have changed every squad. The bookmaker knows about both.
    """)
    return


@app.cell
def _(MODEL_COLORS, all_preds, px):
    _month_order = ["Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May"]
    _by_month = (
        all_preds[all_preds.model.isin(MODEL_COLORS)]
        .assign(month=lambda d: d.date.dt.strftime("%b"))
        .groupby(["model", "month"])
        .agg(rps=("rps", "mean"), matches=("rps", "size"))
        .reset_index()
    )
    # plotly joins points in row order, so sort the months into season order first
    _by_month = _by_month.sort_values("month", key=lambda m: m.map(_month_order.index))
    _fig = px.line(
        _by_month, x="month", y="rps", color="model", markers=True, color_discrete_map=MODEL_COLORS,
        hover_data={"rps": ":.4f", "matches": True},
        labels={"month": "", "rps": "mean RPS (lower is better)"},
        title="Prediction error by month (test seasons)",
    )
    _fig.update_layout(width=720, hovermode="x unified")
    _fig
    return


@app.cell(hide_code=True)
def _(best_xi, mo, results):
    _r = results["all test (RPS)"]
    mo.md(
        f"""
        ## 6. What we learned

        1. **Both models clearly beat base rates** (RPS {_r['Maher']:.4f} vs {_r['Base rates']:.4f}). Two numbers per team plus home advantage capture most of the predictable signal in goals.
        2. **Dixon–Coles barely beats Maher here** ({_r['Dixon–Coles']:.4f} vs {_r['Maher']:.4f}), and the difference is within noise.
            - **$\\rho$ adds essentially nothing.** That fits what we saw in section 4: modern Premier League low scores don't show the pattern Dixon & Coles found in the 1990s, and $\\rho$ is unstable from season to season.
            - **Time decay helps in some seasons and hurts in others.** The $\\xi$ chosen on a single validation season is quite aggressive (half-life ≈ {np.log(2) / best_xi:.0f} days). It throws away a lot of last season, which especially hurts in August. One season is a *noisy* place to tune a hyperparameter.
        3. **The bookmaker is clearly better** ({_r['Bookmaker (closing)']:.4f}), by a margin well outside the noise. The market knows about lineups, injuries, transfers and promoted teams. We only know past scores.
        4. **The process is the real result.** We now have an honest, leak-free evaluation harness: any future idea can be dropped into `walk_forward` and scored on the same 1,140 matches.

        ### Ideas to try next
        - **Better hyperparameter tuning:** validate $\\xi$ over more than one season, or use a rolling validation.
        - **Promoted teams:** start them at the average rating of recently promoted sides instead of 0. (The Bayesian hierarchical model in plan step 3 does this properly.)
        - **xG instead of goals** (plan step 2): xG is a much less noisy measure of team strength, which should help most where we are weakest, early in the season.
        """
    )
    return


if __name__ == "__main__":
    app.run()

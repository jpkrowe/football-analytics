import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # FPL, part 1: predicting next week's points

    **Goal:** use each player's past performance to predict their **FPL points for the next gameweek**. Before trusting any prediction, we test it honestly on matches it hasn't seen.

    How FPL points are made matters for modelling:

    | Source | Roughly |
    |---|---|
    | Playing | 1 point for under 60 minutes, 2 points for 60+ |
    | Goals and assists | goals are worth more the more defensive the position; assists 3 |
    | Clean sheets | big for goalkeepers and defenders, small for midfielders |
    | Bonus | 1–3 points for the best performers in each match (from the BPS score) |
    | Defensive contributions | 2 points for enough tackles/clearances/blocks/etc. (**new rule from 2025-26**) |
    | Negatives | cards, goals conceded (GK/DEF), own goals, missed penalties |

    So the biggest question for most players is simply **will they play?** After that come attacking threat and fixture difficulty.

    The plan for this notebook:
    1. **Data:** three past seasons of gameweek-by-gameweek history (the community *vaastav* archive) plus the current season live from the FPL API (via `penaltyblog`).
    2. **Features:** summarise each player's recent past and their fixture (including Dixon–Coles fixture strength from Part 1) as numbers a model can use.
    3. **The parts model:** predict appearance, goals, assists, clean sheets and other points separately, then turn them into points with the FPL rules. Backtest it against simple form rules.
    4. **Predict the next gameweek:** an interactive dashboard.
    5. **Look further ahead:** project the next 1–10 gameweeks from each player's fixtures, and backtest that too.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import requests
    import penaltyblog as pb
    import penaltyblog.fpl as fpl
    from concurrent.futures import ThreadPoolExecutor
    from datetime import date
    from sklearn.ensemble import HistGradientBoostingRegressor

    # Dixon-Coles functions from the Part 1 notebook
    from basic_goal_models import fit_dixon_coles, outcome_probs, predict_lambdas, score_grid

    return (
        HistGradientBoostingRegressor,
        ThreadPoolExecutor,
        date,
        fit_dixon_coles,
        fpl,
        mo,
        np,
        outcome_probs,
        pd,
        predict_lambdas,
        requests,
        score_grid,
    )


@app.cell(hide_code=True)
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
            title={"font": {"size": 16, "color": _ink}, "x": 0.01, "xanchor": "left", "y": 1, "yanchor": "top", "yref": "container", "pad": {"t": 14}},
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

    # Each position keeps the same colour on every chart
    POSITIONS = ["GK", "DEF", "MID", "FWD"]
    POSITION_COLORS = dict(zip(POSITIONS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
    return POSITIONS, POSITION_COLORS, make_subplots, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Data

    **Past seasons** come from the [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) archive: one row per player per fixture, with points, minutes, goals, xG, bonus and more. We use 2023-24 to 2025-26.

    Two data-quality lessons from checking this archive:
    - It has an `xP` (expected points) column, but it **can't be trusted as a benchmark**. In 2023-25 its average *exactly* matches the actual points, which suggests it was recorded after the matches (leakage). In 2025-26 it averages a third of actual points. We ignore it.
    - Defensive-contribution stats only exist from 2025-26, when the new scoring rule arrived. We keep that column but leave it empty (NaN) for older seasons, rather than pretending they were 0.
    """)
    return


@app.cell
def _(np, pd):
    ARCHIVE_SEASONS = ["2023-24", "2024-25", "2025-26"]
    DEFCON_FROM = "2025-26"  # first season with defensive-contribution points
    _URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv"
    COLUMNS = [
        "season", "GW", "name", "position", "team", "fixture", "kickoff_time", "was_home", "team_h_score", "team_a_score",
        "price", "minutes", "starts", "total_points", "goals_scored", "assists", "clean_sheets", "bonus", "bps",
        "expected_goals", "expected_assists", "expected_goals_conceded", "defensive_contribution",
    ]


    def _load_archive(season):
        df = pd.read_csv(_URL.format(season=season)).assign(season=season)
        df = df[df.position != "AM"]  # 2024-25 "assistant manager" chip, not a player
        if "defensive_contribution" not in df:
            df["defensive_contribution"] = np.nan  # not recorded before 2025-26
        return df.assign(
            price=df["value"] / 10,  # the archive stores prices in tenths of £1m
            kickoff_time=pd.to_datetime(df.kickoff_time, utc=True),
        )[COLUMNS]


    archive = pd.concat([_load_archive(s) for s in ARCHIVE_SEASONS], ignore_index=True)
    archive
    return COLUMNS, DEFCON_FROM, archive


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The current season** comes live from the official FPL API:
    - `penaltyblog` gives each player's details (`get_player_data`) and gameweek history (`get_player_history`, one request per player).
    - The API's `fixtures/` endpoint gives the upcoming fixtures.

    The ~650 history requests are run 8 at a time and **cached on disk for the day** (`mo.persistent_cache`), so re-running the notebook doesn't hammer the API.
    """)
    return


@app.cell
def _(COLUMNS, ThreadPoolExecutor, date, fpl, mo, np, pd, requests):
    CURRENT_SEASON = "2026-27"
    _API = "https://fantasy.premierleague.com/api"

    players_now = fpl.get_player_data()
    _gameweeks = fpl.get_gameweek_info()
    NEXT_GW = int(_gameweeks.loc[_gameweeks.is_next, "id"].iloc[0])

    _team_names = dict(zip(players_now.team_id, players_now.team_name))
    fixtures = pd.DataFrame(requests.get(f"{_API}/fixtures/", timeout=30).json()).assign(
        home_team=lambda d: d.team_h.map(_team_names),
        away_team=lambda d: d.team_a.map(_team_names),
        kickoff_time=lambda d: pd.to_datetime(d.kickoff_time, utc=True),
    )


    @mo.persistent_cache
    def fetch_histories(player_ids, as_of):
        """Gameweek history for every player. `as_of` (today's date) makes the cache refresh daily."""
        with ThreadPoolExecutor(max_workers=8) as pool:
            return pd.concat(pool.map(fpl.get_player_history, player_ids), ignore_index=True)


    _info = players_now.assign(
        name=players_now.first_name + " " + players_now.second_name,
        position=players_now.position_short.replace({"GKP": "GK"}),
    )[["id", "name", "position"]]

    _hist = fetch_histories(players_now.id.tolist(), str(date.today()))
    _hist = _hist.merge(_info, left_on="element", right_on="id").join(fixtures.set_index("id")[["home_team", "away_team"]], on="fixture")
    current = _hist.assign(
        season=CURRENT_SEASON,
        GW=_hist["round"],
        team=np.where(_hist.was_home, _hist.home_team, _hist.away_team),  # the team they played *for* in that match
        price=_hist["value"],
        expected_goals=_hist.expected_goals.astype(float),
        expected_assists=_hist.expected_assists.astype(float),
        expected_goals_conceded=_hist.expected_goals_conceded.astype(float),
    )[COLUMNS]

    mo.md(f"Current season: **{len(current):,}** player-fixture rows up to now. Next gameweek to predict: **GW{NEXT_GW}**.")
    return CURRENT_SEASON, NEXT_GW, current, fixtures, players_now


@app.cell
def _(
    COLUMNS,
    CURRENT_SEASON,
    NEXT_GW,
    archive,
    current,
    fixtures,
    pd,
    players_now,
):
    # One row per player per fixture in the next gameweek, with the results left blank (NaN)
    _next = fixtures[fixtures.event == NEXT_GW]
    _sides = pd.concat([
        _next.assign(team=_next.home_team, was_home=True),
        _next.assign(team=_next.away_team, was_home=False),
    ])[["id", "team", "was_home", "kickoff_time"]].rename(columns={"id": "fixture"})

    upcoming = (
        players_now[players_now.status != "u"]  # "u" = unavailable (left the club, etc.)
        .assign(
            name=players_now.first_name + " " + players_now.second_name,
            position=players_now.position_short.replace({"GKP": "GK"}),
            price=players_now.now_cost,
        )
        .merge(_sides, left_on="team_name", right_on="team")
        .assign(season=CURRENT_SEASON, GW=NEXT_GW)
        # keep only who/when/where: players_now also has *season-total* stats (goals, minutes, ...) that must not leak in
        [["season", "GW", "name", "position", "team", "fixture", "kickoff_time", "was_home", "price"]]
        .reindex(columns=COLUMNS)
    )

    raw = pd.concat([archive, current, upcoming.assign(upcoming=True)], ignore_index=True)
    raw["upcoming"] = raw.upcoming.fillna(False).astype(bool)
    raw.groupby("season").agg(rows=("name", "size"), players=("name", "nunique"), gameweeks=("GW", "nunique"))
    return (raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Features: describing a player's past as numbers

    A model can't read a player's history directly, so we summarise it. For each player and gameweek we compute **averages over their previous 1, 3, 5 or 10 gameweeks** of:

    - **Points and minutes:** recent form and, crucially, whether they're getting picked.
    - **Starts, xG, xA, BPS, bonus:** underlying performance. These are less noisy than points (a player can be unlucky not to score while still getting chances).
    - **Defensive contributions:** for the new 2025-26 rule (empty for older seasons).
    - **xG and xA per 90 minutes** over the last 20 gameweeks: how dangerous a player is *when he plays*, kept separate from how much he plays. Left empty until a player has 270+ minutes, so a single cameo can't produce a silly rate.

    Plus context about **this** gameweek's fixture(s):
    - position, home/away, number of fixtures (double gameweeks)
    - **price, capped at £10m.** Price is a useful summary of how good FPL thinks a player is. But above £10m only one or two players exist (e.g. only Haaland and Salah have ever cost £13m+), so an uncapped price lets the model memorise individuals instead of learning general rules.
    - **team form:** goals scored and conceded over the player's team's last 10 matches, and the same for the **opponent**.
    - **fixture strength from Dixon–Coles (Part 1):** expected goals for and against, clean-sheet chance and win chance (see below).

    **The golden rule:** every feature uses only matches *before* the gameweek being predicted. That's what `.shift(1)` does: it moves each player's history down one row, so a gameweek never sees its own result. Seasons are joined by player name, so last season's form carries into the start of this season.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fixture strength from Part 1 (Dixon–Coles)

    "Goals scored/conceded over the last 10 matches" is a crude measure of fixture difficulty. It ignores who those goals came against, ignores home advantage, and is noisy. So we **reuse the Dixon–Coles model from Part 1**, imported straight from that notebook (`from basic_goal_models import fit_dixon_coles, ...`). marimo lets one notebook import another notebook's functions when they live in their own cells.

    Before each gameweek we refit Dixon–Coles on **all earlier matches only** (no peeking), using Part 1's time decay $\xi = 0.0075$. Then for every fixture we compute, from each team's point of view:

    - `dc_team_xg`: expected goals the team scores
    - `dc_opp_xg`: expected goals the team concedes
    - `dc_cs_prob`: P(clean sheet) = P(opponent scores 0), read off the score grid
    - `dc_win_prob`: P(win)

    For a double gameweek these are summed: total expected goals, and expected number of clean sheets. They're **added alongside** the old goals-for/against features, not instead of them. The first ~8 gameweeks of 2023-24 have no values (too little history to fit ratings), and the trees handle the gaps.
    """)
    return


@app.cell
def _(
    CURRENT_SEASON,
    NEXT_GW,
    fit_dixon_coles,
    mo,
    outcome_probs,
    pd,
    predict_lambdas,
    raw,
    score_grid,
):
    DC_XI = 0.0075  # time decay chosen on the validation season in Part 1 (half-life ≈ 92 days)
    DC_FEATURES = ["dc_team_xg", "dc_opp_xg", "dc_cs_prob", "dc_win_prob"]

    # One row per match, built from the FPL data (upcoming fixtures have no score yet)
    _sides = raw.drop_duplicates(["season", "fixture", "team"])
    fpl_matches = (
        _sides[_sides.was_home == True][["season", "GW", "fixture", "kickoff_time", "team", "team_h_score", "team_a_score"]]
        .rename(columns={"kickoff_time": "date", "team": "home", "team_h_score": "home_goals", "team_a_score": "away_goals"})
        .merge(_sides[_sides.was_home == False][["season", "fixture", "team"]].rename(columns={"team": "away"}), on=["season", "fixture"])
    )

    _played = fpl_matches.dropna(subset=["home_goals"])
    _rows = []
    for (_season, _gw), _block in fpl_matches.groupby(["season", "GW"], sort=False):
        _cutoff = _block.date.min()
        _train = _played[_played.date < _cutoff]  # only matches before this gameweek starts
        if len(_train) < 100:  # too little history for stable ratings (start of 2023-24)
            continue
        _model = fit_dixon_coles(_train, as_of=_cutoff, xi=DC_XI)
        for _m in _block.itertuples():
            _lam_home, _lam_away = predict_lambdas(_model, _m.home, _m.away)
            _grid = score_grid(_lam_home, _lam_away, _model["rho"])
            _p_home, _p_draw, _p_away = outcome_probs(_grid)
            # grid[:, 0] = away scores 0 (home clean sheet); grid[0, :] = home scores 0 (away clean sheet)
            _rows.append((_season, _m.fixture, _m.home, _lam_home, _lam_away, _grid[:, 0].sum(), _p_home))
            _rows.append((_season, _m.fixture, _m.away, _lam_away, _lam_home, _grid[0, :].sum(), _p_away))
    dc_fixture_features = pd.DataFrame(_rows, columns=["season", "fixture", "team"] + DC_FEATURES)

    _next = fpl_matches[(fpl_matches.season == CURRENT_SEASON) & (fpl_matches.GW == NEXT_GW)]
    _view = pd.concat([
        _next[["fixture", "home", "away"]].rename(columns={"home": "team", "away": "opponent"}).assign(venue="home"),
        _next[["fixture", "away", "home"]].rename(columns={"away": "team", "home": "opponent"}).assign(venue="away"),
    ]).merge(dc_fixture_features[dc_fixture_features.season == CURRENT_SEASON], on=["fixture", "team"])
    mo.vstack([
        mo.md(f"**GW{NEXT_GW} fixtures through Dixon–Coles**, sorted by clean-sheet chance:"),
        _view[["team", "opponent", "venue"] + DC_FEATURES].sort_values("dc_cs_prob", ascending=False).round(2).reset_index(drop=True),
    ])
    return DC_FEATURES, DC_XI, dc_fixture_features, fpl_matches


@app.cell
def _(DC_FEATURES, DEFCON_FROM, POSITIONS, dc_fixture_features, np, pd, raw):
    LAGS = {"pts": [1, 3, 10], "mins": [1, 3, 10], "starts": [5], "xg": [10], "xa": [10], "bps": [5], "bonus": [10], "dc": [5]}
    PRICE_CAP = 10.0  # above this, price mostly identifies individual players
    CONTEXT = ["position", "price_capped", "home", "n_fixtures", "team_gf_l10", "team_ga_l10", "opp_gf_l10", "opp_ga_l10", "defcon_rules", "n_prev"]
    PER90 = ["xg_p90_l20", "xa_p90_l20"]
    FEATURES = [f"{col}_l{n}" for col, windows in LAGS.items() for n in windows] + PER90 + CONTEXT + DC_FEATURES


    def build_features(raw, fixture_strength):
        """One row per player per gameweek, with features built only from earlier gameweeks.

        fixture_strength: Dixon-Coles features per (season, fixture, team).
        """
        # Team form: goals for/against over each team's previous 10 matches (shift(1) excludes the match itself)
        games = raw.drop_duplicates(["season", "fixture", "team"])[["season", "fixture", "team", "kickoff_time", "was_home", "team_h_score", "team_a_score"]]
        games = games.assign(
            gf=np.where(games.was_home, games.team_h_score, games.team_a_score),
            ga=np.where(games.was_home, games.team_a_score, games.team_h_score),
        ).sort_values("kickoff_time")
        for c in ["gf", "ga"]:
            games[f"team_{c}_l10"] = games.groupby("team")[c].transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
        team_form = games[["season", "fixture", "team", "team_gf_l10", "team_ga_l10"]]
        opp_form = team_form.rename(columns={"team": "opponent", "team_gf_l10": "opp_gf_l10", "team_ga_l10": "opp_ga_l10"})
        form = team_form.merge(opp_form, on=["season", "fixture"]).query("team != opponent")
        rows = raw.merge(form, on=["season", "fixture", "team"], how="left").merge(
            fixture_strength, on=["season", "fixture", "team"], how="left"
        )

        # One row per player per gameweek (a double gameweek has two fixtures)
        gw = rows.groupby(["season", "GW", "name"], as_index=False).agg(
            position=("position", "last"),
            team=("team", "last"),
            kickoff=("kickoff_time", "min"),
            upcoming=("upcoming", "max"),
            price=("price", "last"),
            n_fixtures=("fixture", "size"),
            home=("was_home", "mean"),
            pts=("total_points", "sum"),
            mins=("minutes", "sum"),
            starts=("starts", "sum"),
            xg=("expected_goals", "sum"),
            xa=("expected_assists", "sum"),
            bps=("bps", "sum"),
            bonus=("bonus", "sum"),
            dc=("defensive_contribution", "sum"),
            team_gf_l10=("team_gf_l10", "mean"),
            team_ga_l10=("team_ga_l10", "mean"),
            opp_gf_l10=("opp_gf_l10", "mean"),
            opp_ga_l10=("opp_ga_l10", "mean"),
        )
        # Dixon-Coles fixture strength, summed over a double gameweek (min_count=1 keeps "unknown" as NaN, not 0)
        strength = rows.groupby(["season", "GW", "name"])[DC_FEATURES].sum(min_count=1).reset_index()
        gw = gw.merge(strength, on=["season", "GW", "name"], how="left")

        # groupby-sum turns all-missing into 0, so restore the gaps: upcoming results, and pre-2025-26 defensive stats
        gw.loc[gw.upcoming, list(LAGS)] = np.nan
        gw.loc[gw.season < DEFCON_FROM, "dc"] = np.nan

        # Rolling averages of each player's *previous* gameweeks, across seasons
        gw = gw.sort_values(["name", "kickoff"]).reset_index(drop=True)
        history = gw.groupby("name")
        for col, windows in LAGS.items():
            for n in windows:
                gw[f"{col}_l{n}"] = history[col].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
        # Chances per 90 minutes over the last 20 gameweeks: how dangerous a player is *when he plays*
        minutes_l20 = history["mins"].transform(lambda s: s.shift(1).rolling(20, min_periods=1).sum())
        for col in ["xg", "xa"]:
            total_l20 = history[col].transform(lambda s: s.shift(1).rolling(20, min_periods=1).sum())
            gw[f"{col}_p90_l20"] = np.where(minutes_l20 >= 270, total_l20 / minutes_l20 * 90, np.nan)
        gw["price_capped"] = gw.price.clip(upper=PRICE_CAP)
        gw["n_prev"] = history.cumcount()  # how many earlier gameweeks we know about
        gw["defcon_rules"] = (gw.season >= DEFCON_FROM).astype(int)
        gw["position"] = pd.Categorical(gw.position, categories=POSITIONS)
        return gw


    player_gw = build_features(raw, dc_fixture_features)
    player_gw[["season", "GW", "name", "position", "team", "pts"] + FEATURES].tail(5)
    return FEATURES, player_gw


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The parts model

    Rather than predicting total points in one go, we predict **what happens** and let the **FPL rules** turn that into points:

    | Part | Target (per player per gameweek) | Loss | Points rule |
    |---|---|---|---|
    | Appearance | 1 per fixture for 1–59 minutes, 2 for 60+ | squared error | as is |
    | Goals | goals scored | Poisson | × 6 GK/DEF, 5 MID, 4 FWD |
    | Assists | assists | Poisson | × 3 |
    | Clean sheets | clean sheets (only awarded for 60+ minutes) | Poisson | × 4 GK/DEF, 1 MID, 0 FWD |
    | Defensive contributions | matches reaching the threshold: 10 actions for defenders, 12 for midfielders and forwards (from 2025-26) | Poisson | × 2 (0 for GK) |
    | Other | everything else: bonus, saves, cards, goals conceded... | squared error | as is |

    "Other" is defined as *total points minus the four rule-based parts*, so the parts always add up to the real total:

    $$E[\text{points}] = E[\text{app}] + g_\text{pos}E[\text{goals}] + 3E[\text{assists}] + c_\text{pos}E[\text{clean sheets}] + d_\text{pos}E[\text{def. contributions}] + E[\text{other}]$$

    The defensive-contribution model only trains on 2025-26 onwards, because that data (and the rule) doesn't exist before.

    This works because **expected values add up**, even when the parts are linked (for example, a defender who plays 90 minutes is more likely to get a clean sheet). Each part's model just has to get its own average right.

    **The models.** Each part gets its own *gradient-boosted regression model* (scikit-learn's `HistGradientBoostingRegressor`), all using the same features. It builds hundreds of small decision trees, each one correcting the errors of the ones before. Trees suit this problem:
    - they handle "not playing → nothing happens" naturally,
    - they cope with missing values (new players, older seasons),
    - they find interactions (e.g. a defender facing a weak attack) without being told.

    The parts differ only in their **target** and their **loss function**, which is how errors are measured during training. *Poisson* loss suits counts that are usually 0 and never predicts a negative. *Squared error* is the usual default, and is needed where values can be negative.

    **Why parts?**
    - **Position is handled by the rules**, not learned: a defender's goal is worth 6 and a forward's 4.
    - **Interpretable:** every prediction comes with expected goals, assists, clean sheets and minutes points.
    - **Easy to improve one piece at a time**, e.g. a better clean-sheet model.
    """)
    return


@app.cell
def _(
    DEFCON_FROM,
    FEATURES,
    HistGradientBoostingRegressor,
    np,
    pd,
    player_gw,
    raw,
):
    GOAL_POINTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
    CLEAN_SHEET_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
    DEFCON_POINTS = {"GK": 0, "DEF": 2, "MID": 2, "FWD": 2}
    DEFCON_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}  # actions needed in one match; goalkeepers can't earn them
    PARTS = {
        "app": "squared_error", "goals": "poisson", "assists": "poisson", "cs": "poisson", "defcon": "poisson",
        "other": "squared_error",
    }


    def rules_points(parts, position):
        """Combine parts (columns app, goals, assists, cs, defcon, other) into points using the FPL rules."""
        pos = position.astype(str)
        return (
            parts["app"]
            + pos.map(GOAL_POINTS) * parts["goals"]
            + 3 * parts["assists"]
            + pos.map(CLEAN_SHEET_POINTS) * parts["cs"]
            + pos.map(DEFCON_POINTS) * parts["defcon"]
            + parts["other"]
        )


    def fit_part(train, target, loss):
        train = train[train[target].notna()]  # e.g. defensive contributions only exist from 2025-26
        model = HistGradientBoostingRegressor(
            loss=loss, max_iter=300, learning_rate=0.05, min_samples_leaf=50, categorical_features="from_dtype", random_state=0
        )
        return model.fit(train[FEATURES], train[target])


    def fit_parts(train):
        """One model per part."""
        return {part: fit_part(train, part, loss) for part, loss in PARTS.items()}


    def predict_parts(models, df):
        """Expected value of each part, plus the total expected points from the FPL rules."""
        parts = pd.DataFrame({part: model.predict(df[FEATURES]) for part, model in models.items()}, index=df.index)
        return parts.assign(points=rules_points(parts, df.position))


    # Targets for each part, per player per gameweek (summed over double-gameweek fixtures)
    _targets = (
        raw.assign(
            app=np.select([raw.minutes >= 60, raw.minutes > 0], [2, 1], 0),
            # 1 if the player reached the defensive-contribution threshold in the match; NaN before the rule existed
            defcon=np.where(
                raw.season >= DEFCON_FROM,
                (raw.defensive_contribution >= raw.position.map(DEFCON_THRESHOLD)).astype(float),
                np.nan,
            ),
        )
        .rename(columns={"goals_scored": "goals", "clean_sheets": "cs"})
        .groupby(["season", "GW", "name"], as_index=False)[["app", "goals", "assists", "cs", "defcon"]]
        .sum(min_count=1)
    )
    player_gw_parts = player_gw.merge(_targets, on=["season", "GW", "name"], how="left")
    player_gw_parts.loc[player_gw_parts.upcoming, ["app", "goals", "assists", "cs", "defcon"]] = np.nan
    # "other" = whatever the rule-based parts don't explain, so the parts always add up to the real total
    player_gw_parts["other"] = player_gw_parts.pts - rules_points(
        player_gw_parts.assign(other=0.0, defcon=player_gw_parts.defcon.fillna(0)), player_gw_parts.position
    )

    player_gw_parts[~player_gw_parts.upcoming].groupby("position", observed=True)[list(PARTS) + ["pts"]].mean().round(3)
    return (
        CLEAN_SHEET_POINTS,
        DEFCON_POINTS,
        GOAL_POINTS,
        PARTS,
        fit_parts,
        player_gw_parts,
        predict_parts,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Backtest: does it actually predict?

    **The test** follows the same walk-forward idea as the goals notebook. For each gameweek from GW20 to GW38 of 2025-26, the parts are fitted on *everything before that gameweek*, then predict it. Training on the first half of 2025-26 also lets the models learn the new defensive-contribution rule.

    **Simple baselines to beat:** "a player will score what they averaged over the last 1 / 3 / 10 gameweeks".

    **Metrics:**
    - **MAE:** mean absolute error, the average number of points the prediction is off by.
    - **RMSE:** like MAE but punishes big misses more.
    - **Top-10 points:** what FPL actually cares about. Each gameweek, take each method's 10 highest predictions and see how many points those players really scored (averaged per player). *Perfect hindsight* shows the ceiling.
    """)
    return


@app.cell
def _(fit_parts, np, pd, player_gw_parts, predict_parts):
    TEST_SEASON, TEST_FROM_GW = "2025-26", 20


    def top10_points(df, col):
        """Average real points of each gameweek's 10 highest-predicted players."""
        return df.sort_values(col, ascending=False).groupby("GW").head(10).groupby("GW").pts.mean()


    _known = player_gw_parts[~player_gw_parts.upcoming]
    _is_test_season = _known.season == TEST_SEASON
    _weeks = []
    for _gw in range(TEST_FROM_GW, 39):
        _train = _known[(_known.season < TEST_SEASON) | (_is_test_season & (_known.GW < _gw))]
        _week = _known[_is_test_season & (_known.GW == _gw)]
        _weeks.append(_week.join(predict_parts(fit_parts(_train), _week).add_prefix("exp_")))

    backtest = pd.concat(_weeks).assign(**{
        "Parts model": lambda d: d.exp_points,
        "Last gameweek": lambda d: d.pts_l1.fillna(0),
        "Avg last 3": lambda d: d.pts_l3.fillna(0),
        "Avg last 10": lambda d: d.pts_l10.fillna(0),
        "Perfect hindsight": lambda d: d.pts,
    })
    METHODS = ["Last gameweek", "Avg last 3", "Avg last 10", "Parts model"]

    pd.DataFrame([
        {
            "method": _m,
            "MAE": (backtest[_m] - backtest.pts).abs().mean(),
            "RMSE": np.sqrt(((backtest[_m] - backtest.pts) ** 2).mean()),
            "top-10 points per player": top10_points(backtest, _m).mean(),
        }
        for _m in METHODS + ["Perfect hindsight"]
    ]).round(2)
    return TEST_FROM_GW, TEST_SEASON, backtest, top10_points


@app.cell
def _(backtest, pd, px, top10_points):
    _by_gw = pd.concat([top10_points(backtest, _m).rename(_m) for _m in ["Avg last 10", "Parts model", "Perfect hindsight"]], axis=1)
    _by_gw = _by_gw.reset_index().melt(id_vars="GW", var_name="method", value_name="points")
    _fig = px.line(
        _by_gw, x="GW", y="points", color="method", markers=True,
        color_discrete_map={"Avg last 10": "#2a78d6", "Parts model": "#eb6834", "Perfect hindsight": "#a9a8a2"},
        labels={"points": "avg real points of the top-10 picks", "GW": "gameweek (2025-26)"},
        title="Each gameweek's top-10 picks: how many points did they score?",
    )
    _fig.update_traces(hovertemplate="%{y:.1f}")
    _fig.update_layout(width=760, hovermode="x unified", legend_title_text="")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Are the parts right on average?** Across all test predictions, does each part's average prediction match what actually happened? And how does the model do for each position? (*Bias* is average predicted minus actual: positive means over-predicting.)
    """)
    return


@app.cell
def _(PARTS, backtest, mo, pd):
    _calib = pd.DataFrame({
        "part": list(PARTS),
        "avg predicted": [backtest[f"exp_{p}"].mean() for p in PARTS],
        "avg actual": [backtest[p].mean() for p in PARTS],
    }).round(3)
    _by_pos = (
        backtest.groupby("position", observed=True)
        .apply(lambda d: pd.Series({
            "player-gameweeks": len(d),
            "MAE": (d["Parts model"] - d.pts).abs().mean(),
            "bias": (d["Parts model"] - d.pts).mean(),
        }))
        .round(3)
        .reset_index()
    )
    mo.hstack([_calib, _by_pos], justify="start", gap=2, wrap=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Takeaway.**
    - The parts model beats every simple form rule. Its top-10 picks average about **4.8 points per player per gameweek**, against 4.15 for "average of the last 10 gameweeks", and its average error is lower too.
    - That's still far from the ~12.7 ceiling, because a single gameweek is very random.
    - Appearance, goals, assists, defensive contributions and other points are all **well calibrated**.
    - **Clean sheets are under-predicted** (≈0.06 per player-gameweek against ≈0.07 actual). That makes the clean-sheet model the obvious next part to improve.

    **What changed the results** (all on this backtest):
    - **Defensive contributions as their own part:** accuracy about the same, but they're now visible and follow the real threshold rule.
    - **Capping price at £10m and adding xG/xA per 90:** top-10 points rose by about 0.4 per player. That's just outside the noise, while MAE stayed the same. Several variants were compared on this test season, so the true gain is probably a bit smaller.
    - **Haaland is still over-predicted** by about half a point a gameweek. His xG is so far above everyone else's that the trees have very few comparable players to learn from.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Predictions for the next gameweek

    Fit the parts on **everything** we know, including this season so far, and predict the upcoming gameweek. Each player gets their expected appearance points, goals, assists, clean sheets and other points, and the total.

    The dashboard's **Next few gameweeks** view uses the projections from section 5.

    One extra adjustment: the FPL API publishes a `chance_of_playing_next_round` for flagged players (injuries, suspensions, illness). The models can't see injury news, so we **multiply every part by that chance**. For example, a 75%-flagged player keeps 75% of their predicted points. Unflagged players count as 100%.
    """)
    return


@app.cell(hide_code=True)
def _(
    NEXT_GW,
    dashboard_view,
    efficiency_fig,
    find_players,
    horizon,
    horizon_view,
    max_price,
    mo,
    next_gw_preds,
    pick_positions,
    rank_by,
    selected_view,
    top20_fig,
    value_fig,
):
    _shown = next_gw_preds[next_gw_preds.position.isin(pick_positions.value) & (next_gw_preds.price <= max_price.value)]
    _table_cols = [
        "player", "team", "position", "price", "predicted_points", "points_per_million", "exp_appearance_pts", "exp_goals",
        "exp_assists", "exp_clean_sheets", "exp_defcon", "exp_other", "chance", "avg pts last 3", "avg mins last 3", "news",
    ]
    _views = {
        "Top 20 & value": mo.vstack([
            mo.hstack([top20_fig, value_fig], justify="start", gap=1, wrap=True),
            mo.md("In the value scatter, points per £m is the slope from the origin to each player, so the best value sits towards the top left."),
        ]),
        "Next few gameweeks": mo.vstack([
            horizon,
            horizon_view,
            mo.md("Each cell shows the opponent and the predicted points. Weeks further ahead are rougher guesses: form, injuries and team selection will change."),
        ]),
        "Points per £m": mo.vstack([
            efficiency_fig,
            mo.md(
                "Points per £m rewards cheap regular starters, which is useful for filling out a squad within budget. "
                "Premium players earn their price in other ways: a captain's points are doubled, so raw points matter most there."
            ),
        ]),
        "Selected players": selected_view,
        "Table": mo.ui.table(_shown.sort_values(rank_by.value, ascending=False)[_table_cols].round(2), page_size=15, selection=None),
    }

    mo.vstack([
        mo.md(f"### GW{NEXT_GW} predictions dashboard"),
        mo.hstack([pick_positions, max_price, rank_by], justify="start", gap=2, wrap=True),
        find_players,
        dashboard_view,
        _views[dashboard_view.value],
    ], gap=1)
    return


@app.cell
def _(NEXT_GW, fit_parts, mo, player_gw_parts, players_now, predict_parts):
    part_models = fit_parts(player_gw_parts[~player_gw_parts.upcoming])

    _chance = (
        players_now.assign(name=players_now.first_name + " " + players_now.second_name)
        .set_index("name")[["chance_of_playing_next_round", "web_name", "news"]]
    )
    _chance = _chance[~_chance.index.duplicated()]
    _next = player_gw_parts[player_gw_parts.upcoming & (player_gw_parts.GW == NEXT_GW)].join(_chance, on="name")
    _chance_next = _next.chance_of_playing_next_round.fillna(100) / 100
    # Every part scaled by the chance of playing (a flagged player is less likely to do anything)
    _pred = predict_parts(part_models, _next).mul(_chance_next, axis=0)

    next_gw_preds = (
        _next.assign(
            chance=_chance_next,
            predicted_points=_pred.points,
            exp_appearance_pts=_pred.app,
            exp_goals=_pred.goals,
            exp_assists=_pred.assists,
            exp_clean_sheets=_pred.cs,
            exp_defcon=_pred.defcon,
            exp_other=_pred.other,
        )
        .assign(points_per_million=lambda d: d.predicted_points / d.price)
        .assign(label=lambda d: d.web_name + " (" + d.team + ")")  # unique: some players share a short name
        .rename(columns={"web_name": "player", "pts_l3": "avg pts last 3", "mins_l3": "avg mins last 3"})
        [[
            "player", "team", "position", "price", "predicted_points", "points_per_million",
            "exp_appearance_pts", "exp_goals", "exp_assists", "exp_clean_sheets", "exp_defcon", "exp_other", "chance",
            "avg pts last 3", "avg mins last 3", "n_fixtures", "news", "label",
        ]]
        .sort_values("predicted_points", ascending=False)
        .reset_index(drop=True)
    )
    mo.md(f"Predicted **{len(next_gw_preds)}** players for GW{NEXT_GW}.")
    return next_gw_preds, part_models


@app.cell
def _(HORIZON_MAX, POSITIONS, mo, next_gw_preds, np):
    pick_positions = mo.ui.multiselect(options=POSITIONS, value=POSITIONS, label="Positions")
    max_price = mo.ui.slider(
        start=3.5, stop=float(np.ceil(next_gw_preds.price.max())), step=0.5, value=float(np.ceil(next_gw_preds.price.max())),
        label="Max price (£m)", show_value=True,
    )
    rank_by = mo.ui.radio(
        options={"Predicted points": "predicted_points", "Points per £m": "points_per_million"},
        value="Predicted points", label="Rank by", inline=True,
    )
    horizon = mo.ui.slider(start=1, stop=HORIZON_MAX, value=5, label="Gameweeks ahead", show_value=True)
    dashboard_view = mo.ui.radio(
        options=["Top 20 & value", "Next few gameweeks", "Points per £m", "Selected players", "Table"],
        value="Top 20 & value", label="View", inline=True,
    )
    # These controls are displayed in the dashboard above, not here
    return dashboard_view, horizon, max_price, pick_positions, rank_by


@app.cell
def _(mo, next_gw_preds):
    find_players = mo.ui.multiselect(
        options=sorted(next_gw_preds.label), label="Find players (type to search)", full_width=True
    )


    def highlight_players(fig, x, y, labels):
        """Circle and name the chosen players on a scatter chart. They are shown even if the filters hide them."""
        picked = next_gw_preds[next_gw_preds.label.isin(labels)]
        fig.add_scatter(
            x=picked[x], y=picked[y], mode="markers+text", text=picked.player, textposition="top center",
            textfont={"size": 12, "color": "#0b0b0b"},
            marker={"size": 18, "color": "rgba(0,0,0,0)", "line": {"width": 2.5, "color": "#0b0b0b"}},
            name="selected", hoverinfo="skip", showlegend=False,
        )
        return fig

    return find_players, highlight_players


@app.cell
def _(
    CLEAN_SHEET_POINTS,
    DEFCON_POINTS,
    GOAL_POINTS,
    NEXT_GW,
    find_players,
    mo,
    next_gw_preds,
    pd,
    px,
):
    _picked = next_gw_preds.set_index("label").loc[[_l for _l in find_players.value if _l in set(next_gw_preds.label)]].reset_index()
    if _picked.empty:
        selected_view = mo.md("*Pick players above to see a breakdown of their predictions.*")
    else:
        # Prediction split into its parts, in points
        _parts = pd.DataFrame({
            "label": _picked.label,
            "appearance": _picked.exp_appearance_pts,
            "goals": _picked.position.astype(str).map(GOAL_POINTS) * _picked.exp_goals,
            "assists": 3 * _picked.exp_assists,
            "clean sheets": _picked.position.astype(str).map(CLEAN_SHEET_POINTS) * _picked.exp_clean_sheets,
            "def. contributions": _picked.position.astype(str).map(DEFCON_POINTS) * _picked.exp_defcon,
            "other": _picked.exp_other,
        }).melt(id_vars="label", var_name="part", value_name="points")

        _fig = px.bar(
            _parts, x="points", y="label", color="part", orientation="h", barmode="relative",
            category_orders={"part": ["appearance", "goals", "assists", "clean sheets", "def. contributions", "other"]},
            labels={"label": "", "points": "expected points"},
            title=f"Predicted points by part, GW{NEXT_GW}",
        )
        _fig.update_traces(hovertemplate="%{y}: %{x:.2f} pts<extra>%{fullData.name}</extra>")
        _fig.update_yaxes(categoryorder="array", categoryarray=_picked.label.tolist()[::-1], showgrid=False, ticksuffix="  ")
        _fig.update_xaxes(showgrid=True, gridcolor="#ececea", zeroline=True, zerolinecolor="#a9a8a2")
        _fig.update_layout(
            width=720, height=max(300, 150 + 48 * len(_picked)), bargap=0.35, margin={"t": 130}, legend_title_text=""
        )

        _cols = [
            "player", "team", "position", "price", "predicted_points", "exp_appearance_pts", "exp_goals",
            "exp_assists", "exp_clean_sheets", "exp_defcon", "exp_other", "chance", "avg pts last 3", "avg mins last 3", "news",
        ]
        selected_view = mo.vstack([_fig, mo.ui.table(_picked[_cols].round(2), selection=None, show_column_summaries=False)])
    return (selected_view,)


@app.cell
def _(
    NEXT_GW,
    POSITIONS,
    POSITION_COLORS,
    find_players,
    highlight_players,
    max_price,
    next_gw_preds,
    pick_positions,
    px,
    rank_by,
):
    _labels = {
        "predicted_points": "predicted points",
        "points_per_million": "predicted points per £m", "player": "", "price": "price (£m)",
    }
    _shown = next_gw_preds[next_gw_preds.position.isin(pick_positions.value) & (next_gw_preds.price <= max_price.value)]
    _shown = _shown.sort_values(rank_by.value, ascending=False)
    _top = _shown.head(20).iloc[::-1]  # reversed so the best is at the top of the bar chart

    top20_fig = px.bar(
        _top, x=rank_by.value, y="player", color="position", orientation="h",
        color_discrete_map=POSITION_COLORS, category_orders={"position": POSITIONS},
        hover_data={
            "team": True, "price": ":.1f", "chance": ":.0%", "predicted_points": ":.2f",
            "points_per_million": ":.2f", "exp_goals": ":.2f", "exp_assists": ":.2f", "exp_clean_sheets": ":.2f",
        },
        labels=_labels,
        title=f"Top 20 by {_labels[rank_by.value]}, GW{NEXT_GW}",
    )
    top20_fig.update_yaxes(categoryorder="array", categoryarray=_top.player.tolist(), showgrid=False)
    top20_fig.update_xaxes(showgrid=True, gridcolor="#ececea")
    top20_fig.update_layout(width=620, height=640, bargap=0.3, legend_title_text="")

    value_fig = px.scatter(
        _shown[_shown["avg mins last 3"] > 0], x="price", y="predicted_points", color="position",
        color_discrete_map=POSITION_COLORS, category_orders={"position": POSITIONS},
        hover_name="player",
        hover_data={"team": True, "price": ":.1f", "predicted_points": ":.2f", "points_per_million": ":.2f", "position": False},
        labels=_labels,
        title="Value: predicted points vs price",
    )
    value_fig.update_traces(marker={"size": 9, "opacity": 0.8, "line": {"width": 1, "color": "white"}})
    value_fig.update_xaxes(showgrid=True, gridcolor="#ececea")
    value_fig.update_layout(width=620, height=520, legend_title_text="")
    value_fig = highlight_players(value_fig, "price", "predicted_points", find_players.value)
    return top20_fig, value_fig


@app.cell
def _(
    NEXT_GW,
    POSITIONS,
    POSITION_COLORS,
    max_price,
    next_gw_preds,
    pick_positions,
    px,
):
    _eff = next_gw_preds[next_gw_preds.position.isin(pick_positions.value) & (next_gw_preds.price <= max_price.value)]
    _eff = _eff.nlargest(20, "points_per_million").iloc[::-1]  # reversed so the best is at the top

    efficiency_fig = px.bar(
        _eff, x="points_per_million", y="player", color="position", orientation="h",
        color_discrete_map=POSITION_COLORS, category_orders={"position": POSITIONS},
        text=_eff.price.map(lambda p: f"£{p:.1f}m"),
        hover_data={"team": True, "price": ":.1f", "predicted_points": ":.2f", "points_per_million": ":.2f", "chance": ":.0%"},
        labels={"points_per_million": "predicted points per £m", "player": "", "predicted_points": "predicted points", "price": "price (£m)"},
        title=f"20 most points-efficient players, GW{NEXT_GW}",
    )
    efficiency_fig.update_traces(textposition="outside", textfont={"size": 11, "color": "#52514e"}, cliponaxis=False)
    efficiency_fig.update_yaxes(categoryorder="array", categoryarray=_eff.player.tolist(), showgrid=False)
    efficiency_fig.update_xaxes(showgrid=True, gridcolor="#ececea", range=[0, _eff.points_per_million.max() * 1.15])
    efficiency_fig = efficiency_fig.update_layout(width=660, height=640, bargap=0.3, legend_title_text="")  # assigned so the chart only shows in the dashboard
    return (efficiency_fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Looking further ahead

    In FPL you plan transfers over the next few weeks, not one. To predict gameweek $t+k$ we keep each player's **form features as they are now**, since we don't know future results, and update only the **fixture features** for each future match:
    - home/away
    - the opponent's recent goals for/against
    - Dixon–Coles expected goals, clean-sheet and win chances, from one fit on all matches so far

    The same parts models then predict each future gameweek. A **blank** gameweek (no fixture) scores 0, and a **double** gameweek gets both fixtures.

    Injury flags: a player flagged **0%** is treated as out for the whole horizon. A partial flag (e.g. 75%) only applies to the next gameweek, since minor knocks usually clear up.

    **Limitation:** the further ahead, the less we know. Form changes, players get injured or dropped, and these projections don't account for any of that, so treat weeks 6–10 as a rough guide.
    """)
    return


@app.cell
def _(
    DC_FEATURES,
    DC_XI,
    POSITIONS,
    fit_dixon_coles,
    outcome_probs,
    pd,
    predict_lambdas,
    score_grid,
):
    HORIZON_MAX = 10
    FIXTURE_COLS = ["home", "n_fixtures", "team_gf_l10", "team_ga_l10", "opp_gf_l10", "opp_ga_l10"] + DC_FEATURES


    def fixture_context(played, future, as_of):
        """Fixture features for future matches, using only matches played before `as_of`.

        played/future: one row per match with date, GW, fixture, home, away (+ goals for played matches).
        Returns one row per team per future fixture.
        """
        past = played[played.date < as_of]
        # Each team's goals for/against over its last 10 matches (at least 3 needed)
        by_team = pd.concat([
            past[["date", "home", "home_goals", "away_goals"]].set_axis(["date", "team", "gf", "ga"], axis=1),
            past[["date", "away", "away_goals", "home_goals"]].set_axis(["date", "team", "gf", "ga"], axis=1),
        ]).sort_values("date").groupby("team").tail(10)
        form = by_team.groupby("team").agg(gf_l10=("gf", "mean"), ga_l10=("ga", "mean"), n=("gf", "size"))
        form = form[form.n >= 3].drop(columns="n")

        model = fit_dixon_coles(past, as_of=as_of, xi=DC_XI)
        rows = []
        for m in future.itertuples():
            lam_home, lam_away = predict_lambdas(model, m.home, m.away)
            grid = score_grid(lam_home, lam_away, model["rho"])
            p_home, _, p_away = outcome_probs(grid)
            rows.append((m.GW, m.fixture, m.home, m.away, 1.0, lam_home, lam_away, grid[:, 0].sum(), p_home))
            rows.append((m.GW, m.fixture, m.away, m.home, 0.0, lam_away, lam_home, grid[0, :].sum(), p_away))
        ctx = pd.DataFrame(rows, columns=["GW", "fixture", "team", "opponent", "home"] + DC_FEATURES)
        return ctx.join(form.add_prefix("team_"), on="team").join(form.add_prefix("opp_"), on="opponent")


    def project_fixtures(base, ctx):
        """Give each player (one row of current features in `base`) a row for every future gameweek of their team.

        History features stay as they are; fixture features come from `ctx`. Double gameweeks are combined
        the same way as in build_features, and blank gameweeks get no row.
        """
        static = [c for c in base.columns if c not in FIXTURE_COLS + ["GW"]]
        rows = base[static].merge(ctx, on="team")
        agg = {c: "first" for c in static if c != "name"}
        agg.update({c: "mean" for c in ["home", "team_gf_l10", "team_ga_l10", "opp_gf_l10", "opp_ga_l10"]})
        agg.update({c: "sum" for c in DC_FEATURES})
        agg.update({"fixture": "size", "opponent": " + ".join})
        out = rows.groupby(["name", "GW"], as_index=False).agg(agg).rename(columns={"fixture": "n_fixtures"})
        out["position"] = pd.Categorical(out.position, categories=POSITIONS)
        return out

    return HORIZON_MAX, fixture_context, project_fixtures


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Backtest: the next 5 gameweeks

    From each gameweek 20–34 of 2025-26, predict every player's **total points over the next 5 gameweeks**, using only what was known at that point (Dixon–Coles refitted then too). Compare with:
    - **Avg last 10 × fixtures:** recent points per gameweek times the number of fixtures coming up.
    - **Next-week model × fixtures:** the model's prediction for the *first* week, repeated for every fixture. This ignores who they play.
    """)
    return


@app.cell
def _(
    TEST_FROM_GW,
    TEST_SEASON,
    fit_parts,
    fixture_context,
    fpl_matches,
    pd,
    player_gw_parts,
    predict_parts,
    project_fixtures,
):
    HORIZON_BACKTEST = 5
    _known = player_gw_parts[~player_gw_parts.upcoming]
    _is_test_season = _known.season == TEST_SEASON
    _played = fpl_matches.dropna(subset=["home_goals"])
    _test_matches = fpl_matches[fpl_matches.season == TEST_SEASON]

    _windows = []
    for _start in range(TEST_FROM_GW, 39 - HORIZON_BACKTEST + 1):
        _as_of = _test_matches[_test_matches.GW == _start].date.min()
        _models = fit_parts(_known[(_known.season < TEST_SEASON) | (_is_test_season & (_known.GW < _start))])
        _future = _test_matches[(_test_matches.GW >= _start) & (_test_matches.GW < _start + HORIZON_BACKTEST)]
        _proj = project_fixtures(_known[_is_test_season & (_known.GW == _start)], fixture_context(_played, _future, _as_of))
        _proj["pred"] = predict_parts(_models, _proj).points.to_numpy()
        _first = _proj[_proj.GW == _start].set_index("name")
        _tot = _proj.groupby("name").agg(
            **{"Fixture-aware model": ("pred", "sum"), "fixtures": ("n_fixtures", "sum"), "pts_l10": ("pts_l10", "first")}
        )
        _tot["Avg last 10 × fixtures"] = _tot.pts_l10.fillna(0) * _tot.fixtures
        _tot["Next-week model × fixtures"] = (_first.pred / _first.n_fixtures).reindex(_tot.index) * _tot.fixtures
        _actual = _known[_is_test_season & (_known.GW >= _start) & (_known.GW < _start + HORIZON_BACKTEST)].groupby("name").pts.sum()
        _tot["actual"] = _actual.reindex(_tot.index).fillna(0)
        _windows.append(_tot.dropna(subset=["Next-week model × fixtures"]).assign(start=_start).reset_index())
    horizon_backtest = pd.concat(_windows)


    def _top10_total(col):
        """Average real 5-GW total of each window's 10 highest-predicted players."""
        return horizon_backtest.sort_values(col, ascending=False).groupby("start").head(10).groupby("start").actual.mean()


    pd.DataFrame([
        {
            "method": _m,
            "top-10 real points per player (5 GWs)": _top10_total(_m).mean(),
            "MAE (5-GW total)": (horizon_backtest[_m] - horizon_backtest.actual).abs().mean(),
            "bias": (horizon_backtest[_m] - horizon_backtest.actual).mean(),
        }
        for _m in ["Avg last 10 × fixtures", "Next-week model × fixtures", "Fixture-aware model", "actual"]
    ]).replace({"method": {"actual": "Perfect hindsight"}}).round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Takeaway.**
    - **Accuracy:** projecting each fixture is the most accurate of the three methods (lowest MAE, 3.19 vs 3.56 for the simple average).
    - **Picks:** its top-10 picks scored the most over 5 weeks, about **21.4 points per player** vs 20.2 for "avg last 10 × fixtures".
    - **But that gap is within noise** (+1.2 ± 1.9). There are only 15 starting points and their 5-week windows overlap. Perfect hindsight would have scored ~35, so a lot of every 5-week window is still luck.
    - **Fixtures matter a little, beyond simple form:** the fixture-aware version edges out "next week's prediction × fixtures".
    """)
    return


@app.cell
def _(
    HORIZON_MAX,
    NEXT_GW,
    fixture_context,
    fixtures,
    fpl_matches,
    mo,
    np,
    part_models,
    player_gw_parts,
    players_now,
    predict_parts,
    project_fixtures,
):
    # Upcoming fixtures from the FPL API, as one row per match
    _future = (
        fixtures[(fixtures.event >= NEXT_GW) & (fixtures.event < NEXT_GW + HORIZON_MAX)]
        .rename(columns={"event": "GW", "id": "fixture", "home_team": "home", "away_team": "away", "kickoff_time": "date"})
        [["GW", "fixture", "home", "away", "date"]]
    )
    _as_of = _future.date.min()
    _base = player_gw_parts[player_gw_parts.upcoming & (player_gw_parts.GW == NEXT_GW)]
    _proj = project_fixtures(_base, fixture_context(fpl_matches.dropna(subset=["home_goals"]), _future, _as_of))

    _info = players_now.assign(name=players_now.first_name + " " + players_now.second_name).drop_duplicates("name").set_index("name")
    _proj = _proj.join(_info[["web_name", "chance_of_playing_next_round"]], on="name")
    _chance = _proj.chance_of_playing_next_round.fillna(100) / 100
    # Partial flags only affect the next gameweek; a 0% flag rules the player out for the whole horizon
    _scale = np.where(_proj.GW == NEXT_GW, _chance, np.where(_chance == 0, 0.0, 1.0))
    _pred = predict_parts(part_models, _proj).mul(_scale, axis=0)

    _short = dict(zip(players_now.team_name, players_now.team_name_short))
    horizon_preds = _proj.assign(
        player=_proj.web_name,
        label=_proj.web_name + " (" + _proj.team + ")",
        vs=[" + ".join(_short.get(o, o) for o in opp.split(" + ")) + (" (H)" if h == 1 else " (A)" if h == 0 else "")
            for opp, h in zip(_proj.opponent, _proj.home)],
        points=_pred.points,
        exp_goals=_pred.goals,
        exp_assists=_pred.assists,
        exp_clean_sheets=_pred.cs,
        exp_defcon=_pred.defcon,
    )[["label", "player", "team", "position", "price", "GW", "vs", "points", "exp_goals", "exp_assists", "exp_clean_sheets", "exp_defcon"]]
    mo.md(f"Projected **{horizon_preds.label.nunique()}** players over GW{NEXT_GW}–GW{NEXT_GW + HORIZON_MAX - 1}.")
    return (horizon_preds,)


@app.cell
def _(
    NEXT_GW,
    POSITIONS,
    POSITION_COLORS,
    horizon,
    horizon_preds,
    make_subplots,
    max_price,
    mo,
    pick_positions,
):
    _window = horizon_preds[horizon_preds.GW < NEXT_GW + horizon.value]
    _window = _window[_window.position.isin(pick_positions.value) & (_window.price <= max_price.value)]
    _totals = (
        _window.groupby(["label", "player", "team", "position", "price"], observed=True, as_index=False)
        [["points", "exp_goals", "exp_assists", "exp_clean_sheets", "exp_defcon"]].sum()
        .nlargest(20, "points")
    )
    _order = _totals.label.tolist()[::-1]  # best at the top
    _cells = _window[_window.label.isin(_order)]
    _gws = list(range(NEXT_GW, NEXT_GW + horizon.value))
    _z = _cells.pivot_table(index="label", columns="GW", values="points", aggfunc="sum").reindex(index=_order, columns=_gws)
    _txt = _cells.pivot_table(index="label", columns="GW", values="vs", aggfunc="first").reindex(index=_order, columns=_gws).fillna("blank")

    _fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.32, 0.68], horizontal_spacing=0.02,
                         subplot_titles=[f"Total, next {horizon.value} GW", "Predicted points per gameweek"])
    for _pos in POSITIONS:
        _d = _totals[_totals.position == _pos]
        _fig.add_bar(
            x=_d.points, y=_d.label, orientation="h", name=_pos, marker_color=POSITION_COLORS[_pos],
            customdata=_d[["price"]], hovertemplate="%{y}: %{x:.1f} pts (£%{customdata[0]:.1f}m)<extra></extra>", row=1, col=1,
        )
    _fig.add_heatmap(
        z=_z.to_numpy(), x=[f"GW{g}" for g in _gws], y=_order, text=_txt.to_numpy(),
        texttemplate="%{text}<br>%{z:.1f}", textfont={"size": 10},
        colorscale=["#ffffff", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"], showscale=False, xgap=2, ygap=2,
        hovertemplate="%{y}, %{x}: %{text} → %{z:.2f} pts<extra></extra>", row=1, col=2,
    )
    _fig.update_yaxes(categoryorder="array", categoryarray=_order, showgrid=False)
    _fig.update_xaxes(showgrid=True, gridcolor="#ececea", row=1, col=1)
    _fig.update_layout(
        width=min(1300, 520 + 80 * horizon.value), height=760, barmode="stack", bargap=0.3,
        title=f"Best players over the next {horizon.value} gameweeks",
        margin={"t": 140}, legend={"y": 1.08},  # room for title, legend and subplot titles
    )
    horizon_view = mo.vstack([
        _fig,
        mo.ui.table(_totals.iloc[::-1].round(2).rename(columns={"points": f"points next {horizon.value} GW"}), selection=None, show_column_summaries=False),
    ])
    return (horizon_view,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. What we learned, and what's next

    - **Playing time dominates.** About 60% of player-gameweeks are zero minutes, and recent minutes are the strongest predictor.
    - **Single-gameweek form is mostly noise.** "Last gameweek's points" is the worst baseline. The models learn by themselves that hot streaks fade, because they are trained on what happened *next*.
    - **Predicting the parts works.** It beats simple form rules, and every prediction can be explained as expected minutes points, goals, assists, clean sheets, defensive contributions and other points. Each part follows the actual FPL scoring rule, including the defensive-contribution threshold.
    - **Earlier experiments** (removed to keep this notebook lean):
      - A single model predicting total points directly was about as accurate overall, but chased hot streaks. When it disagreed strongly with the parts model, the parts model was usually closer.
      - Extra skill-vs-luck features (short-window xG/xA, points above expectation) added nothing measurable.
      - Dixon–Coles fixture strength didn't measurably change the backtest either. It's kept because it's principled and shows each fixture's difficulty.
    - **Watch for features that identify individuals.** Uncapped price let the model memorise Haaland and Salah (the only £13m+ players). Capping it at £10m and measuring chances per 90 minutes improved the ranking of the best players.
    - **Looking ahead works, modestly.** Projecting each future fixture beats simple form × fixtures over 5 gameweeks (MAE 3.19 vs 3.56), but most of a 5-week total is still luck.
    - **Check your data.** The archive's `xP` column looked like a free benchmark but turned out to be leaky or broken.

    **Ideas to try next:**
    - **Improve the clean-sheet part:** it under-predicts. Fitting Dixon–Coles on more seasons of results (football-data.co.uk), or on xG, might help.
    - **Split "other" further:** defensive contributions now have their own part. Bonus (from BPS rank within each match) and saves are still lumped together.
    - **Pick a squad:** the next-1–10-gameweek projections now exist. Next, choose the best 15-man squad (or transfers) within £100m and the 3-per-team rule, e.g. with a linear programme.
    - **Better long-range projections:** form and minutes are frozen at today's values. Regressing them towards the player's longer-term average for later weeks may help.
    """)
    return


if __name__ == "__main__":
    app.run()

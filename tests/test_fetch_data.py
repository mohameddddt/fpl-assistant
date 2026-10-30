from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fetch_data import unfinished_clubs  # noqa: E402


def fixtures(rows: list) -> pd.DataFrame:
    """(event, team_h, team_a, finished), as fetch_season_olbauday writes fixtures.csv."""
    return pd.DataFrame(rows, columns=['event', 'team_h', 'team_a', 'finished'])


def test_a_finished_gameweek_holds_nothing_back() -> None:
    assert unfinished_clubs(fixtures([(7, 1, 2, True), (7, 3, 4, True)]), 7) == set()


def test_both_clubs_in_a_match_still_to_play_are_held_back() -> None:
    week = fixtures([(7, 1, 2, True), (7, 3, 4, False)])
    assert unfinished_clubs(week, 7) == {3, 4}


def test_a_double_gameweek_waits_for_the_second_match() -> None:
    # Club 1's per-gameweek row covers both matches, so it is not a result
    # until both are over.
    week = fixtures([(7, 1, 2, True), (7, 3, 1, False)])
    assert unfinished_clubs(week, 7) == {1, 3}


def test_other_gameweeks_and_blank_clubs_are_not_held_back() -> None:
    # Club 5 has no match in GW7; GW8 has not started.
    week = fixtures([(7, 1, 2, True), (8, 1, 5, False)])
    assert unfinished_clubs(week, 7) == set()


def test_finished_read_back_as_text_or_blank() -> None:
    week = fixtures([(7, 1, 2, 'True'), (7, 3, 4, 'False'), (7, 5, 6, np.nan)])
    assert unfinished_clubs(week, 7) == {3, 4, 5, 6}


def test_float_club_ids_from_a_player_map_still_match() -> None:
    # The player -> club map yields floats once any player is unmapped.
    team_ids = pd.Series([1.0, 3.0, np.nan])
    held = team_ids.isin(unfinished_clubs(fixtures([(7, 1, 2, True), (7, 3, 4, False)]), 7))
    assert held.tolist() == [False, True, False]


def test_no_fixtures_holds_nothing_back() -> None:
    assert unfinished_clubs(pd.DataFrame(), 7) == set()


def test_selected_percentage_becomes_a_manager_count() -> None:
    from fetch_data import selected_count
    out = selected_count(pd.Series([12.5, '0.4', None]), 10_000_000.0)
    assert out.iloc[0] == 1_250_000 and out.iloc[1] == 40_000 and np.isnan(out.iloc[2])


# ---------------------------------------------------------------- doubles/blanks
from fetch_data import attach_fixtures, fixtures_by_club  # noqa: E402


def scheduled(rows: list) -> pd.DataFrame:
    """(id, event, team_h, team_a, kickoff_time, team_h_score, team_a_score)."""
    return pd.DataFrame(rows, columns=['id', 'event', 'team_h', 'team_a', 'kickoff_time',
                                       'team_h_score', 'team_a_score'])


def week_rows(rows: list) -> pd.DataFrame:
    """(element, minutes, total_points, goals_scored, selected) for one gameweek."""
    return pd.DataFrame(rows, columns=['element', 'minutes', 'total_points',
                                       'goals_scored', 'selected'])


def no_history(element: int):
    raise AssertionError(f"element {element} should not need an FPL read")


def test_a_double_keeps_both_fixtures_in_kickoff_order() -> None:
    # Listed out of order: the later kickoff first.
    lookup = fixtures_by_club(scheduled([
        (2, 9, 3, 1, '2026-12-05T15:00:00Z', 0, 2),
        (1, 9, 1, 2, '2026-12-02T19:30:00Z', 1, 1),
    ]))
    assert [m['opponent_team'] for m in lookup[(9, 1)]] == [2, 3]
    assert [m['was_home'] for m in lookup[(9, 1)]] == [True, False]
    assert [m['team_score'] for m in lookup[(9, 1)]] == [1, 2]
    assert len(lookup[(9, 2)]) == 1 and len(lookup[(9, 3)]) == 1


def test_a_single_fixture_week_is_unchanged() -> None:
    lookup = fixtures_by_club(scheduled([(1, 7, 1, 2, '2026-10-01T14:00:00Z', 3, 0)]))
    out = attach_fixtures(week_rows([(10, 90, 8, 1, 500), (20, 90, 1, 0, 300)]),
                          pd.Series([1, 2]), lookup, 7, no_history)
    assert out['element'].tolist() == [10, 20]
    assert out['opponent_team'].tolist() == [2, 1]
    assert out['was_home'].tolist() == [True, False]
    assert out['team_h_score'].tolist() == [3, 3] and out['team_a_score'].tolist() == [0, 0]


def test_a_double_is_split_into_one_row_per_match_from_fpl_history() -> None:
    lookup = fixtures_by_club(scheduled([
        (1, 9, 1, 2, '2026-12-02T19:30:00Z', 1, 1),
        (2, 9, 3, 1, '2026-12-05T15:00:00Z', 0, 2),
    ]))
    history = {10: [
        {'round': 8, 'opponent_team': 4, 'was_home': True, 'minutes': 90, 'total_points': 2},
        {'round': 9, 'opponent_team': 2, 'was_home': True, 'minutes': 90,
         'total_points': 2, 'goals_scored': 0},
        {'round': 9, 'opponent_team': 3, 'was_home': False, 'minutes': 75,
         'total_points': 9, 'goals_scored': 1},
    ]}
    out = attach_fixtures(week_rows([(10, 165, 11, 1, 500)]), pd.Series([1]), lookup, 9,
                          history.get)
    assert out['opponent_team'].tolist() == [2, 3]
    assert out['minutes'].tolist() == [90, 75]
    assert out['total_points'].tolist() == [2, 9]
    assert out['goals_scored'].tolist() == [0, 1]
    # The week's ownership belongs to both halves.
    assert out['selected'].tolist() == [500, 500]
    assert out['fixture'].tolist() == [1, 2]
    assert out['team_h_score'].tolist() == [1, 0] and out['team_a_score'].tolist() == [1, 2]


def test_a_double_for_someone_who_did_not_play_needs_no_network_read() -> None:
    lookup = fixtures_by_club(scheduled([
        (1, 9, 1, 2, '2026-12-02T19:30:00Z', 1, 1),
        (2, 9, 3, 1, '2026-12-05T15:00:00Z', 0, 2),
    ]))
    out = attach_fixtures(week_rows([(10, 0, 0, 0, 500)]), pd.Series([1]), lookup, 9,
                          no_history)
    assert out['opponent_team'].tolist() == [2, 3]
    assert out['minutes'].tolist() == [0, 0]


def test_a_double_without_a_per_match_split_is_held_back() -> None:
    lookup = fixtures_by_club(scheduled([
        (1, 9, 1, 2, '2026-12-02T19:30:00Z', 1, 1),
        (2, 9, 3, 1, '2026-12-05T15:00:00Z', 0, 2),
    ]))
    # History unavailable for 10; history for 11 covers only one of the matches.
    history = {11: [{'round': 9, 'opponent_team': 2, 'was_home': True, 'minutes': 90,
                     'total_points': 2}]}
    out = attach_fixtures(week_rows([(10, 90, 2, 0, 1), (11, 90, 2, 0, 1),
                                     (12, 0, 0, 0, 1)]),
                          pd.Series([1, 1, 1]), lookup, 9, history.get)
    assert out['element'].tolist() == [12, 12]


def test_a_blank_drops_zero_minute_rows_but_keeps_anyone_who_played() -> None:
    # Club 5 has no GW7 match. A row with minutes means a bad club mapping, not
    # a non-result, so it is kept (without a fixture) and reported.
    lookup = fixtures_by_club(scheduled([(1, 7, 1, 2, '2026-10-01T14:00:00Z', 3, 0)]))
    out = attach_fixtures(week_rows([(10, 90, 8, 1, 1), (30, 0, 0, 0, 1), (31, 45, 2, 0, 1),
                                     (32, np.nan, np.nan, 0, 1)]),
                          pd.Series([1, 5, 5, np.nan]), lookup, 7, no_history)
    assert out['element'].tolist() == [10, 31]
    assert pd.isna(out.loc[out['element'] == 31, 'opponent_team']).all()


def test_fixture_ids_name_the_match_not_its_row() -> None:
    from fetch_data import fixture_ids
    ids = fixture_ids(pd.Series([14, 20, 3]), pd.Series([15, 5, 14]))
    assert ids.tolist() == [1415, 2005, 314]
    # Reordering the source rows leaves every match its id.
    assert fixture_ids(pd.Series([3, 14]), pd.Series([14, 15])).tolist() == [314, 1415]

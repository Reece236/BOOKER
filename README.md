# ESPN NBA injury-report change log

Written every ~30 min by .github/workflows/collect-injuries.yml (on main) via
full_model/rapm/data_ingest/fetch_injuries.py. injuries/injury_log.csv holds NEW / CHANGED /
CLEARED events; the report as of any time T = each athlete's latest event with snapshot_utc <= T.

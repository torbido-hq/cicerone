*** Settings ***
Documentation       System-spec follow-up (Postgres): track ingest -> second
...                 job -> dashboard Quality. Isolated from system_db.robot
...                 (own schema reset + seed + train) so the second job.run
...                 cannot change that suite's trained catalog mid-run.
Library             libraries/CiceroneSystemLibrary.py
Suite Setup         Set Up Postgres System
Suite Teardown      Reset Postgres Schema

*** Test Cases ***
Track Ingest Feeds The Next Run And Quality Page
    [Documentation]    POST /track, a conversion purchase, a second job run,
    ...                then CTR shows up on the dashboard Quality page.
    ${tracked}=    Track Impressions And Click For User    u1
    Append Postgres Conversion Event    u1    ${tracked}[first_item]
    ${n_events}=    Catalog Event Count    1
    Run System Job Again    system-spec-eval
    Manifest Should Record Successful Run    system-spec-eval    ${n_events}
    Track Store Should Contain Tracked Events    ${tracked}[event_ids]
    Manifest Track Eval Should Record Ctr    ${tracked}[impression_count]
    Track History Should Include    u1    ${tracked}[first_item]
    Dashboard Quality Page Should Show Tracked Metrics    ${tracked}[impression_count]
    Dashboard Quality Page Should Show Ctr    ${tracked}[impression_count]

*** Settings ***
Documentation       Mixed I/O Quality follow-up: Postgres catalog in, dataset
...                 track/eval/manifest out. Isolated from system_mixed.robot
...                 (own seed + train). The conversion purchase is appended to
...                 Postgres events; POST /track writes track.jsonl under the
...                 dataset output; the second job overwrites manifest.json
...                 (read_recent stays 1).
...
...                 Requires a test DB URL via TEST_DATABASE_URL or
...                 POSTGRES_TEST_HOST (see CONTRIBUTING.md).
Library             libraries/CiceroneSystemLibrary.py
Suite Setup         Set Up Mixed System
Suite Teardown      Reset Postgres Schema

*** Test Cases ***
Track Ingest Feeds The Next Run And Quality Page
    [Documentation]    POST /track, a Postgres conversion, a second job run, then CTR on Quality.
    Settings Should Read Db And Write Dataset
    ${tracked}=    Track Impressions And Click For User    u1
    Dataset Track File Should Hold Exactly    ${tracked}[event_ids]
    Postgres Should Hold Only Input Tables
    Append Postgres Conversion Event    u1    ${tracked}[first_item]
    ${n_events}=    Catalog Event Count    1
    Postgres Event Count Should Be    ${n_events}
    Mixed Output Files Should Match Contract
    Run System Job Again    system-spec-eval
    Postgres Should Hold Only Input Tables
    Track Store Should Hold Exactly    ${tracked}[event_ids]
    Dataset Manifest On Disk Should Match Reader
    Manifest Should Record Successful Run    system-spec-eval    ${n_events}
    Manifest Track Eval Should Record Ctr    ${tracked}[impression_count]
    Dataset Eval And History Files Should Exist    ${tracked}[impression_count]
    Dashboard Quality Page Should Show Tracked Metrics    ${tracked}[impression_count]
    Dashboard Quality Page Should Show Ctr    ${tracked}[impression_count]


*** Settings ***
Documentation       System-spec follow-up (dataset): track ingest -> second
...                 job -> dashboard Quality. Isolated from system_dataset.robot
...                 (own seed + train) so the second job.run cannot change
...                 that suite's trained catalog mid-run.
...
...                 Dataset-specific contracts: POST /track appends
...                 track.jsonl under the output path; the conversion
...                 purchase is appended to input events.parquet; the second
...                 job overwrites manifest.json (read_recent stays 1).
Library             libraries/CiceroneSystemLibrary.py
Suite Setup         Seed And Train Dataset System

*** Test Cases ***
Track Ingest Feeds The Next Run And Quality Page
    [Documentation]    POST /track, a conversion purchase, a second job run,
    ...                then CTR shows up on the dashboard Quality page.
    ${tracked}=    Track Impressions And Click For User    u1
    Dataset Track Files Should Exist Only Under Output
    Dataset Track File Should Hold Exactly    ${tracked}[event_ids]
    Append Dataset Conversion Event    u1    ${tracked}[first_item]
    Dataset Input Events Should Be Catalog Plus    u1    ${tracked}[first_item]
    ${n_events}=    Catalog Event Count    1
    Manifest Should Record Successful Run    system-spec
    Run System Job Again    system-spec-eval
    Track Store Should Hold Exactly    ${tracked}[event_ids]
    Dataset Manifest On Disk Should Match Reader
    Manifest Should Record Successful Run    system-spec-eval    ${n_events}
    Manifest Track Eval Should Record Ctr    ${tracked}[impression_count]
    Dataset Eval And History Files Should Exist    ${tracked}[impression_count]
    Track History Should Include    u1    ${tracked}[first_item]
    Dashboard Quality Page Should Show Tracked Metrics    ${tracked}[impression_count]
    Dashboard Quality Page Should Show Ctr    ${tracked}[impression_count]

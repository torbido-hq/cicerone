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
    Append Dataset Conversion Event    u1    ${tracked}[first_item]
    Run System Job Again    system-spec-eval
    Manifest Should Record Successful Run    system-spec-eval
    Dataset Manifest On Disk Should Match Reader
    Track Store Should Contain Tracked Events    ${tracked}[event_ids]
    Dataset Eval And History Files Should Exist    ${tracked}[impression_count]
    Dashboard Quality Page Should Show Tracked Metrics    ${tracked}[impression_count]

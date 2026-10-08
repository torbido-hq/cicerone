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
    Run System Job Again    system-spec-eval
    Manifest Should Record Successful Run    system-spec-eval
    Track Store Should Contain Tracked Events    ${tracked}[event_ids]
    Dashboard Quality Page Should Show Tracked Metrics    ${tracked}[impression_count]

*** Keywords ***
Set Up Postgres System
    ${available}=    Postgres Test Database Available
    IF    not ${available}
        Skip    TEST_DATABASE_URL / POSTGRES_TEST_HOST not set -- start compose postgres (docker compose --env-file docker/postgres/defaults.env --profile db up -d postgres) and export POSTGRES_TEST_HOST=localhost ALLOW_SCHEMA_RESET_FOR_TESTS=1, or run via docker-compose.ci.yml
    END
    Reset Postgres Schema
    Seed And Train Postgres System

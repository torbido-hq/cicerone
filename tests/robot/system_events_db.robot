*** Settings ***
Documentation       Webhook POST /events write-through on Postgres output.
...                 Runs the batch job (db in/out, events webhook enabled),
...                 then mounts serve with the events runtime: a queued
...                 purchase stays invisible until the worker flushes, after
...                 which serve, the Postgres tables and the dashboard
...                 incremental panel show it.
...
...                 Requires a test DB URL via TEST_DATABASE_URL or
...                 POSTGRES_TEST_HOST (see CONTRIBUTING.md).
Library             libraries/CiceroneSystemLibrary.py
Suite Setup         Set Up Postgres Events System
Suite Teardown      Tear Down Postgres Events System

*** Variables ***
${LIVE_USER}        u-live
${LIVE_ITEM}        i2
${LIVE_EVENT_ID}    system-spec-events-db-1

*** Test Cases ***
Webhook Purchase Is Queued Until The Worker Flushes
    [Documentation]    202 is a queue ack; serve and dashboard stay cold until tick().
    Dashboard Should Show No Incremental Flushes
    Start Serve With Events Runtime
    Serve Should Fall Back Without Incremental Rows    ${LIVE_USER}
    Post Webhook Purchase    ${LIVE_USER}    ${LIVE_ITEM}    ${LIVE_EVENT_ID}
    Serve Should Fall Back Without Incremental Rows    ${LIVE_USER}
    Dashboard Should Show No Incremental Flushes

Flush Writes Through To Serve Postgres And Dashboard
    [Documentation]    One worker tick applies the purchase to every reader.
    Flush Events Worker Once
    Serve Should Return Incremental Item    ${LIVE_USER}    ${LIVE_ITEM}    i3    i4
    Stop Serve Events Runtime
    Manifest Should Record Incremental Flush    1
    Postgres Recommendations Should Contain Incremental Row    ${LIVE_USER}    ${LIVE_ITEM}
    Dashboard Should Show Incremental Flush    1

*** Keywords ***
Set Up Postgres Events System
    ${available}=    Postgres Test Database Available
    IF    not ${available}
        Skip    TEST_DATABASE_URL / POSTGRES_TEST_HOST not set -- start compose postgres (docker compose --env-file docker/postgres/defaults.env --profile db up -d postgres) and export POSTGRES_TEST_HOST=localhost ALLOW_SCHEMA_RESET_FOR_TESTS=1, or run via docker-compose.ci.yml
    END
    Reset Postgres Schema
    Seed And Train Postgres System    events_webhook=${True}

Tear Down Postgres Events System
    Stop Serve Events Runtime
    Reset Postgres Schema

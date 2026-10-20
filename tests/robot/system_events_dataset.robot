*** Settings ***
Documentation       Webhook POST /events write-through on local parquet
...                 output. Same journey as system_events_db.robot: a queued
...                 purchase stays invisible until the worker flushes, after
...                 which serve, recommendations.parquet and the dashboard
...                 incremental panel show it. No live database required.
Library             libraries/CiceroneSystemLibrary.py
Suite Setup         Seed And Train Dataset System    events_webhook=${True}
Suite Teardown      Stop Serve Events Runtime

*** Variables ***
${LIVE_USER}        u-live
${LIVE_ITEM}        i2
${LIVE_EVENT_ID}    system-spec-events-1

*** Test Cases ***
Webhook Purchase Is Queued Until The Worker Flushes
    [Documentation]    202 is a queue ack; serve and dashboard stay cold until tick().
    Dashboard Should Show No Incremental Flushes
    Start Serve With Events Runtime
    Serve Should Fall Back Without Incremental Rows    ${LIVE_USER}
    Post Webhook Purchase    ${LIVE_USER}    ${LIVE_ITEM}    ${LIVE_EVENT_ID}
    Serve Should Fall Back Without Incremental Rows    ${LIVE_USER}
    Dashboard Should Show No Incremental Flushes

Flush Writes Through To Serve Parquet And Dashboard
    [Documentation]    One worker tick applies the purchase to every reader.
    Flush Events Worker Once
    Serve Should Return Incremental Item    ${LIVE_USER}    ${LIVE_ITEM}    i3    i4
    Stop Serve Events Runtime
    Manifest Should Record Incremental Flush    1
    Dataset Recommendations Should Contain Incremental Row    ${LIVE_USER}    ${LIVE_ITEM}
    Dashboard Should Show Incremental Flush    1

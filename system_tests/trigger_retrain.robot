*** Settings ***
Documentation    Black-box checks for the retrain webhook (docker-compose.
...              robot.yml) and the end-to-end effect of a trigger: a new
...              run shows up on the dashboard as triggered_by webhook.
...              See tests/test_trigger.py for in-process coverage.
Resource         resources/common.resource
Suite Setup      Run Keywords    Create Trigger Session    AND    Create Dashboard Session

*** Test Cases ***
Health Check Succeeds Without Auth
    ${response}=    GET On Session    trigger    /health
    Should Be Equal As Integers    ${response.status_code}    200

Retrain Requires A Bearer Token
    POST On Session    trigger    /trigger/retrain    expected_status=401

Retrain Accepts A Valid Token And Starts A Run
    ${headers}=    Trigger Auth Headers
    ${response}=    POST On Session    trigger    /trigger/retrain
    ...    headers=${headers}    expected_status=202

Triggered Run Eventually Appears On The Dashboard As Webhook
    Wait Until Keyword Succeeds    30s    1s    Dashboard Shows A Webhook Triggered Run

*** Keywords ***
Dashboard Shows A Webhook Triggered Run
    ${response}=    GET On Session    dashboard    /partials/status    expected_status=200
    ${latest_run}=    Get Regexp Matches    ${response.text}    (?s)data-latest-run.*?</dl>
    Should Not Be Empty    ${latest_run}
    Should Contain    ${latest_run}[0]    data-run-status="success"
    Should Contain    ${latest_run}[0]    >webhook<

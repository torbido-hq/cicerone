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
    ${before}=    Latest Run Generated At
    Set Suite Variable    ${RUN_BEFORE_TRIGGER}    ${before}
    # A cron run holds the run guard (429) until it finishes. Only a 429 is retried: a transport
    # error fails at once, since a lost reply may hide an accepted run.
    ${status}=    Post Trigger Until Not Busy
    Should Be Equal As Integers    ${status}    202

Triggered Run Eventually Appears On The Dashboard As Webhook
    Wait Until Keyword Succeeds    30s    1s    Dashboard Shows A Webhook Triggered Run

*** Keywords ***
Post Trigger Until Not Busy
    ${headers}=    Trigger Auth Headers
    WHILE    True    limit=90s    on_limit_message=Trigger still busy (429) after 90s
        ${response}=    POST On Session    trigger    /trigger/retrain    headers=${headers}    expected_status=any
        IF    ${response.status_code} != 429    BREAK
        Sleep    2s
    END
    RETURN    ${response.status_code}

Latest Run Block
    ${response}=    GET On Session    dashboard    /partials/status    expected_status=200
    ${latest_run}=    Get Regexp Matches    ${response.text}    (?s)data-latest-run.*?</dl>
    Should Not Be Empty    ${latest_run}
    RETURN    ${latest_run}[0]

Latest Run Generated At
    ${block}=    Latest Run Block
    ${generated}=    Get Regexp Matches    ${block}    data-run-generated="([^"]*)"    1
    RETURN    ${generated}[0]

Dashboard Shows A Webhook Triggered Run
    # Only a run newer than the one seen before the trigger counts.
    ${block}=    Latest Run Block
    Should Not Contain    ${block}    data-run-generated="${RUN_BEFORE_TRIGGER}"
    Should Contain    ${block}    data-run-status="success"
    Should Contain    ${block}    >webhook<

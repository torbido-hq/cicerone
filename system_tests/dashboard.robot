*** Settings ***
Documentation    Black-box smoke + functional checks against the real
...              dashboard container (docker-compose.robot.yml), over real
...              HTTP Basic Auth — not in-process (see tests/test_dashboard.py
...              for the pytest/TestClient coverage of the same endpoints).
Resource         resources/common.resource
Suite Setup      Create Dashboard Session

*** Test Cases ***
Health Check Succeeds Without Auth
    ${response}=    GET On Session    dashboard    /health
    Should Be Equal As Integers    ${response.status_code}    200

Robots Txt Disallows Everything
    ${response}=    GET On Session    dashboard    /robots.txt
    Should Be Equal As Integers    ${response.status_code}    200
    Should Contain    ${response.text}    Disallow: /

Dashboard Requires Auth
    Create Session    dashboard_no_auth    ${DASHBOARD_BASE_URL}
    GET On Session    dashboard_no_auth    /dashboard    expected_status=401

Dashboard Rejects Wrong Password
    @{bad_auth}=    Create List    ${DASHBOARD_USERNAME}    wrong-password
    Create Session    dashboard_bad_auth    ${DASHBOARD_BASE_URL}    auth=${bad_auth}
    GET On Session    dashboard_bad_auth    /dashboard    expected_status=401

Dashboard Shows The Seeded Job Run As Successful
    ${response}=    GET On Session    dashboard    /dashboard    expected_status=200
    Should Contain    ${response.text}    data-run-status="success"

Status Partial Reflects The Latest Run
    ${response}=    GET On Session    dashboard    /partials/status    expected_status=200
    Should Contain    ${response.text}    data-run-status="success"

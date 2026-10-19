*** Settings ***
Documentation    Black-box smoke + functional checks against the real serve
...              container (docker-compose.robot.yml), over HTTP — not
...              in-process (see tests/test_serve.py for the pytest/
...              TestClient coverage of the same endpoints).
Resource         resources/common.resource
Suite Setup      Create Serve Session

*** Test Cases ***
Health Check Succeeds Without Auth
    ${response}=    GET On Session    serve    /health
    Should Be Equal As Integers    ${response.status_code}    200
    Should Be Equal As Strings     ${response.json()}[status]    ok

Recommendations Require A Bearer Token
    ${response}=    GET On Session    serve    /recommendations/${KNOWN_USER_ID}
    ...    expected_status=401

Recommendations Reject An Invalid Token
    ${headers}=    Create Dictionary    Authorization=Bearer not-the-right-token
    GET On Session    serve    /recommendations/${KNOWN_USER_ID}
    ...    headers=${headers}    expected_status=401

Recommendations For A Known User Return Seeded Items
    ${headers}=    Serve Auth Headers
    ${response}=    GET On Session    serve    /recommendations/${KNOWN_USER_ID}
    ...    headers=${headers}    expected_status=200
    ${body}=    Set Variable    ${response.json()}
    Should Be Equal As Strings    ${body}[user_id]    ${KNOWN_USER_ID}
    Should Not Be Empty    ${body}[items]
    FOR    ${item}    IN    @{body}[items]
        Should Be True    ${item}[rank] >= 1
        Should Not Be Empty    ${item}[item_id]
        Should Not Be Empty    ${item}[source]
    END

Recommendations Respect The Limit Query Param
    ${headers}=    Serve Auth Headers
    ${response}=    GET On Session    serve    /recommendations/${KNOWN_USER_ID}
    ...    headers=${headers}    params=limit=1    expected_status=200
    ${items}=    Set Variable    ${response.json()}[items]
    Length Should Be    ${items}    1

Recommendations For An Unknown User Fall Back To Popular
    ${headers}=    Serve Auth Headers
    ${response}=    GET On Session    serve    /recommendations/${UNKNOWN_USER_ID}
    ...    headers=${headers}    expected_status=200
    ${body}=    Set Variable    ${response.json()}
    Should Be True    ${body}[fallback]
    Should Not Be Empty    ${body}[items]

*** Settings ***
Documentation    Shared variables/keywords for the Robot Framework system
...              tests (docker-compose.robot.yml). Base URLs/tokens default
...              to the in-network compose service names/ports but are
...              overridable via environment variables for local runs
...              against docker compose port-forwards.
Library          RequestsLibrary
Library          Collections
Library          String

*** Variables ***
${SERVE_BASE_URL}         %{SERVE_BASE_URL=http://serve:8000}
${SERVE_AUTH_TOKEN}       %{SERVE_AUTH_TOKEN=robot-serve-token}
${DASHBOARD_BASE_URL}     %{DASHBOARD_BASE_URL=http://dashboard:8090}
${DASHBOARD_USERNAME}     %{DASHBOARD_USERNAME=robot}
${DASHBOARD_PASSWORD}     %{DASHBOARD_PASSWORD=robot-password}
${TRIGGER_BASE_URL}       %{TRIGGER_BASE_URL=http://recommender:8080}
${TRIGGER_AUTH_TOKEN}     %{TRIGGER_AUTH_TOKEN=robot-trigger-token}

# Seeded by system_tests/seed_catalog.py — keep in sync.
${KNOWN_USER_ID}          robot-u1
${UNKNOWN_USER_ID}        robot-user-does-not-exist

*** Keywords ***
Create Serve Session
    Create Session    serve    ${SERVE_BASE_URL}

Create Dashboard Session
    @{dashboard_auth}=    Create List    ${DASHBOARD_USERNAME}    ${DASHBOARD_PASSWORD}
    Create Session    dashboard    ${DASHBOARD_BASE_URL}    auth=${dashboard_auth}

Create Trigger Session
    Create Session    trigger    ${TRIGGER_BASE_URL}

Serve Auth Headers
    ${headers}=    Create Dictionary    Authorization=Bearer ${SERVE_AUTH_TOKEN}
    RETURN    ${headers}

Trigger Auth Headers
    ${headers}=    Create Dictionary    Authorization=Bearer ${TRIGGER_AUTH_TOKEN}
    RETURN    ${headers}

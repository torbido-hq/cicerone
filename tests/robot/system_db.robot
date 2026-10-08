*** Settings ***
Documentation       System-style end-to-end check against a real Postgres
...                 (Rails system-spec analogue). Seeds events/users/items,
...                 runs the full batch job with db input/output + a model
...                 artifact, then hits the same serve and dashboard HTTP
...                 apps production uses.
...
...                 Requires a test DB URL via TEST_DATABASE_URL or
...                 POSTGRES_TEST_HOST (see support.postgres_defaults /
...                 CONTRIBUTING.md). Schema resets are gated by
...                 support.system_db.reset_schema.
Library             libraries/CiceroneSystemLibrary.py
Suite Setup         Set Up Postgres System
Suite Teardown      Reset Postgres Schema

*** Test Cases ***
Job Round-Trips Recommendations Manifest And Artifact
    [Documentation]    Postgres catalog -> job.run -> the readers serve/dashboard use.
    Recommendation Reader Should Cover Catalog Items
    Recommendations Should Be Served For All Known Users
    User U1 Recommendations Should Respect Requested Count
    Manifest Should Record Successful Run    system-spec
    Artifact Should Be Loadable And Produce Recommendations

Serve Reads Job Output Over HTTP
    [Documentation]    /recommendations matches the reader, with category/availability
    ...                filters and cold-start fallback for an unknown user.
    Serve Should Return Recommendations For User    u1
    Serve Should Return Recommendations For User    u4
    Serve Should Fallback For Unknown User
    Serve Should Support Category Filter    u1    beer    i1    i2
    Serve Should Support Exclude Unavailable    u1    i3    i4

Dashboard HTTP Matches Serve
    [Documentation]    Dashboard status page and per-user lookup mirror what serve returns.
    Dashboard Should Show Latest Run Summary    system-spec
    Dashboard Should Show Recommendations For User    u1

*** Keywords ***
Set Up Postgres System
    ${available}=    Postgres Test Database Available
    IF    not ${available}
        Skip    TEST_DATABASE_URL / POSTGRES_TEST_HOST not set -- start compose postgres (docker compose --env-file docker/postgres/defaults.env --profile db up -d postgres) and export POSTGRES_TEST_HOST=localhost ALLOW_SCHEMA_RESET_FOR_TESTS=1, or run via docker-compose.ci.yml
    END
    Reset Postgres Schema
    Seed And Train Postgres System

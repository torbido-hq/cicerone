*** Settings ***
Documentation       Mixed I/O system check: Postgres catalog in, local parquet
...                 out. Input and output kinds are independent; serve and
...                 dashboard read the parquet tree, and no output tables are
...                 created on the input database.
...
...                 Requires a test DB URL via TEST_DATABASE_URL or
...                 POSTGRES_TEST_HOST (see CONTRIBUTING.md).
Library             libraries/CiceroneSystemLibrary.py
Suite Setup         Set Up Mixed System
Suite Teardown      Reset Postgres Schema

*** Test Cases ***
Job Reads Postgres And Writes Parquet Output
    [Documentation]    Postgres events/users/items -> job.run -> files under the dataset output path.
    Settings Should Read Db And Write Dataset
    Postgres Should Hold Only Input Tables
    Mixed Output Files Should Match Contract
    ${n_events}=    Catalog Event Count
    Postgres Event Count Should Be    ${n_events}
    Recommendations File Should Cover Known Users And Match Reader
    Items Snapshot Should Carry Availability Flags
    Dataset Manifest On Disk Should Match Reader
    Manifest Should Record Successful Run    system-spec    ${n_events}
    Artifact On Disk Should Match Sink

Serve Reads Dataset Output Over HTTP
    [Documentation]    /recommendations matches the parquet files and the manifest stamp.
    Serve Should Match Output Files For User    u1
    Serve Should Match Output Files For User    u4
    Serve Should Fallback For Unknown User
    Serve Should Support Category Filter    u1    beer    i1    i2
    Serve Should Support Exclude Unavailable    u1    i3    i4

Dashboard HTTP Matches Dataset Output
    [Documentation]    Dashboard status page and lookup read the same parquet output.
    Dashboard Should Show Latest Run Summary    system-spec
    Dashboard Should Show Recommendations For User    u1


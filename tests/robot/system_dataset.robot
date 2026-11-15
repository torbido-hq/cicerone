*** Settings ***
Documentation       System-style end-to-end check against local parquet
...                 (dataset I/O). Same operator journeys as system_db.robot,
...                 but assertions are on the files the dataset backend
...                 actually writes. No live database required.
Library             libraries/CiceroneSystemLibrary.py
Suite Setup         Seed And Train Dataset System

*** Test Cases ***
Job Round-Trips Recommendations Manifest And Artifact Via Files
    [Documentation]    Local parquet catalog -> job.run -> the files and readers serve/dashboard use.
    Settings Should Read And Write    dataset    dataset
    Dataset Output Files Should Match Contract
    Dataset Input Events Should Be Catalog Plus
    Recommendation Reader Should Cover Catalog Items
    Recommendations Should Be Served For All Known Users
    User U1 Recommendations Should Respect Requested Count
    Recommendations File Should Cover Known Users And Match Reader
    Items Snapshot Should Carry Availability Flags
    ${n_events}=    Catalog Event Count
    Manifest Should Record Successful Run    system-spec    ${n_events}
    Dataset Manifest On Disk Should Match Reader
    Artifact Should Be Loadable And Produce Recommendations
    Artifact On Disk Should Match Sink

Serve Reads Job Output Over HTTP
    [Documentation]    /recommendations matches the files on disk, with category/availability
    ...                filters and cold-start fallback for an unknown user.
    Serve Should Match Output Files For User    u1
    Serve Should Match Output Files For User    u4
    ${fallback_ids}=    Serve Should Fallback For Unknown User
    Items Should Be Available And In Recommendations File    ${fallback_ids}
    Serve Should Support Category Filter    u1    beer    i1    i2
    Serve Should Support Exclude Unavailable    u1    i3    i4

Dashboard HTTP Matches Serve
    [Documentation]    Dashboard status page and per-user lookup mirror what serve returns.
    Dashboard Should Show Latest Run Summary    system-spec
    Dashboard Should Show Recommendations For User    u1

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
    Dataset Output Files Should Match Contract
    Recommendation Reader Should Cover Catalog Items
    Recommendations Should Be Served For All Known Users
    User U1 Recommendations Should Respect Requested Count
    Manifest Should Record Successful Run    system-spec
    Dataset Manifest On Disk Should Match Reader
    Artifact Should Be Loadable And Produce Recommendations

Serve Reads Job Output Over HTTP
    [Documentation]    /recommendations matches the files on disk, with category/availability
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

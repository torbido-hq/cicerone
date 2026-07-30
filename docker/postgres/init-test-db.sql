-- Extra databases created on first boot of the compose postgres service.
-- Application/tutorial data uses POSTGRES_DB ("cicerone"); pytest should
-- target cicerone_test so system/DB tests never wipe a shared catalog.
CREATE DATABASE cicerone_test OWNER cicerone;

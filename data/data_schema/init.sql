CREATE SCHEMA IF NOT EXISTS data;

CREATE TABLE IF NOT EXISTS data.Record (
	id INTEGER PRIMARY KEY SERIAL,
	name TEXT UNIQUE,
	freq CHAR(2) NOT NULL,
	root TEXT NOT NULL,
	basis TEXT NOT NULL,
	stage TEXT NOT NULL,
	dir TEXT NULL,
	hist TEXT NULL,
	description TEXT NULL,
	size INTEGER,
	dumptime FLOAT,
	hash INTEGER,
	created DATETIME,
	modified DATETIME,
	CONSTRAINT FKRecord_stageId FOREIGN KEY(stageId) REFERENCES Department(id)
);

-- Data Record table foreign keys out to stage tables
CREATE TABLE IF NOT EXISTS data.RecordXStage (
	id INTEGER PRIMARY KEY SERIAL
);

-- Data Record table foreign keys out to stage tables
CREATE TABLE IF NOT EXISTS data.Raw (
	id INTEGER PRIMARY KEY,
	category TEXT NULL
);

CREATE TABLE IF NOT EXISTS data.Mutate (
	id INTEGER PRIMARY KEY,
	type TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS data.Recon (
	id INTEGER PRIMARY KEY,
	type TEXT NOT NULL
);


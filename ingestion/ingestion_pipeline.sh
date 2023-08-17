psql postgres -h 127.0.0.1 -d stocks -f ingestion/build_tables.sql

poetry run python ingestion/run.py
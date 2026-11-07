import duckdb
import pytest

from immo.agent.sql_guard import UnsafeSQLError, validate_sql

AUTORISEES = [
    "select * from gold.communes",
    "WITH t AS (select 1 as a) select a from t;",
    "select * from gold.communes where nom_commune = 'Saint-Germain-en-Laye'",
    "select replace(nom_commune, 'e', 'E') from gold.communes -- commentaire drop",
    "select * from gold.communes where nom_commune = 'drop set use'",
]
REFUSEES = [
    "drop table gold.communes",
    "select 1; drop table gold.communes",
    "insert into gold.communes values (1)",
    "select * from read_csv('/etc/passwd')",
    "select * from read_parquet('s3://bucket/x.parquet')",
    "select * from '/etc/hosts'",
    "attach 'autre.db' as autre",
    "select * from gold.communes; set enable_external_access = true",
    "pragma database_list",
    "select getenv('HOME')",
    "/* commentaire */ create table x as select 1",
    "",
]


@pytest.mark.parametrize("sql", AUTORISEES)
def test_requetes_de_lecture_autorisees(sql):
    validate_sql(sql)


@pytest.mark.parametrize("sql", REFUSEES)
def test_requetes_dangereuses_refusees(sql):
    with pytest.raises(UnsafeSQLError):
        validate_sql(sql)


def test_la_connexion_bloque_aussi_les_fichiers(ctx):
    """Deuxième barrière : même si le filtre texte était contourné, DuckDB refuse."""
    with ctx.connect() as con, pytest.raises(duckdb.Error):
        con.execute("select * from read_csv('/etc/passwd')")
    with ctx.connect() as con, pytest.raises(duckdb.Error):
        con.execute("create table gold.pirate as select 1")


def test_resultat_tronque(ctx):
    df, tronque = ctx.run_agent_sql("select * from silver.ventes", max_rows=10)
    assert len(df) == 10 and tronque

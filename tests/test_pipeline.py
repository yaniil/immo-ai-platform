"""Le nettoyage silver applique bien les règles documentées, et les tables gold sont cohérentes."""

import duckdb


def q(settings, sql):
    with duckdb.connect(str(settings.warehouse_path), read_only=True) as con:
        return con.execute(sql).fetchall()


def test_les_ventes_sont_uniques_et_dans_les_bornes(settings):
    (n, distinct, pmin, pmax, smin, smax), = q(
        settings,
        "select count(*), count(distinct id_mutation), min(prix_m2), max(prix_m2), "
        "min(surface_reelle_bati), max(surface_reelle_bati) from silver.ventes",
    )
    assert n > 3000 and n == distinct
    assert 500 <= pmin and pmax <= 30000
    assert 9 <= smin and smax <= 1000


def test_seules_les_ventes_classiques_sont_gardees(settings):
    natures = q(settings, """
        select distinct b.nature_mutation from silver.ventes s
        join bronze.dvf_mutations b using (id_mutation)""")
    assert natures == [("Vente",)]


def test_les_ventes_de_plusieurs_logements_sont_exclues(settings):
    multi = q(settings, """
        with logements as (
            select distinct id_mutation, type_local, surface_reelle_bati, nombre_pieces_principales
            from bronze.dvf_mutations where type_local in ('Appartement', 'Maison'))
        select id_mutation from logements group by 1 having count(*) > 1""")
    assert multi, "le jeu synthétique doit contenir des ventes multi-logements"
    ids = {m[0] for m in multi}
    gardees = {r[0] for r in q(settings, "select id_mutation from silver.ventes")}
    assert not ids & gardees


def test_un_logement_en_double_ligne_reste_une_seule_vente(settings):
    doublons = q(settings, """
        select id_mutation from bronze.dvf_mutations
        where type_local in ('Appartement', 'Maison')
        group by id_mutation, type_local, surface_reelle_bati
        having count(*) > 1 and count(distinct surface_reelle_bati) = 1""")
    gardees = {r[0] for r in q(settings, "select id_mutation from silver.ventes")}
    assert doublons and any(d[0] in gardees for d in doublons)


def test_les_ventes_avec_local_commercial_sont_exclues(settings):
    mixtes = {r[0] for r in q(settings, "select id_mutation from bronze.dvf_mutations "
                                         "where type_local like 'Local industriel%'")}
    gardees = {r[0] for r in q(settings, "select id_mutation from silver.ventes")}
    assert mixtes and not mixtes & gardees


def test_les_tables_gold_sont_coherentes(settings):
    (source, nb), = q(settings, "select source_donnees, nb_ventes from gold.metadata")
    (total,), = q(settings, "select sum(nb_ventes) from gold.prix_m2_commune_annee")
    assert source == "sample" and total == nb
    assert q(settings, "select count(*) from gold.communes")[0][0] == 26

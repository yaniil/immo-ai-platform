-- Garde-fou : le nettoyage ne doit pas éliminer plus de 70 % des mutations « Vente ».
with brut as (
    select count(distinct id_mutation) as n
    from {{ ref('bronze_dvf_mutations') }}
    where nature_mutation = 'Vente'
),
propre as (
    select count(*) as n from {{ ref('silver_ventes') }}
)
select brut.n as mutations_brutes, propre.n as ventes_propres
from brut, propre
where propre.n < 0.3 * brut.n

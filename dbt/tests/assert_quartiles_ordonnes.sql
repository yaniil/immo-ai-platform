-- Les quartiles doivent encadrer la médiane : sinon l'agrégation est fausse.
select *
from {{ ref('gold_prix_m2_commune_annee') }}
where not (prix_m2_q1 <= prix_m2_median and prix_m2_median <= prix_m2_q3)

# Méthodologie de nettoyage des ventes

## Pourquoi nettoyer les données ?
Les fichiers DVF bruts mélangent des transactions de natures très différentes et une même vente peut apparaître sur plusieurs lignes. Pour calculer un prix au m² fiable, on ne garde que les ventes dont le prix peut être attribué à un seul logement. Les règles ci-dessous sont appliquées dans la couche silver de l'entrepôt (modèle dbt silver_ventes).

## Règle 1 : uniquement les ventes classiques
Seules les mutations de nature « Vente » sont conservées. Les ventes en l'état futur d'achèvement (VEFA, c'est-à-dire le neuf sur plan), les échanges, les adjudications, les expropriations et les ventes de terrains à bâtir sont exclues, car leurs prix ne sont pas comparables à ceux du marché de l'ancien.

## Règle 2 : un seul logement par vente
Quand une mutation contient plusieurs logements (par exemple un immeuble ou deux appartements vendus ensemble), DVF ne donne qu'un prix global. Il est impossible de savoir quelle part du prix revient à chaque logement, donc ces ventes sont exclues. Un même logement répété sur plusieurs lignes (une par parcelle) est d'abord dédoublonné.

## Règle 3 : aucun local d'activité
Les ventes qui incluent un local industriel, commercial ou assimilé sont exclues, car le prix mélange alors un logement et un local professionnel.

## Règle 4 : bornes de plausibilité
La surface réelle bâtie doit être comprise entre 9 et 1 000 m², et le prix au m² entre 500 et 30 000 €. Ces bornes éliminent les erreurs de saisie et les ventes à prix symbolique (par exemple une vente à 1 €).

## Dépendances
Les caves et parkings (dépendances) vendus avec un logement sont conservés : la vente reste valide, mais leur prix est inclus dans la valeur foncière, ce qui augmente légèrement le prix au m² calculé. Le nombre de dépendances est conservé dans la colonne nb_dependances et utilisé par le modèle d'estimation.

## Ordre de grandeur
Sur les vraies données, ces règles conservent généralement une large majorité des ventes de logements. Un test automatique vérifie que le nettoyage ne supprime jamais plus de 70 % des mutations de type « Vente ».

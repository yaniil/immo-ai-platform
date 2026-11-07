# Les données DVF

## Qu'est-ce que DVF ?
DVF (Demandes de valeurs foncières) est la base publique des transactions immobilières déclarées en France. Elle est produite par la Direction générale des finances publiques (DGFiP) à partir des actes notariés et publiée en open data sur data.gouv.fr depuis 2019, en application du décret n° 2018-1350 du 28 décembre 2018. Ce projet utilise la version « DVF géolocalisées » produite par Etalab : les colonnes y sont normalisées et chaque transaction reçoit une latitude et une longitude.

## Couverture géographique et temporelle
DVF couvre toute la France à l'exception de l'Alsace, de la Moselle et de Mayotte. L'Île-de-France est donc entièrement couverte. La base contient les transactions des cinq dernières années environ et elle est mise à jour deux fois par an (au printemps et à l'automne). Les ventes les plus récentes peuvent être incomplètes à cause des délais de publication des actes : les derniers mois d'une année doivent être interprétés avec prudence.

## Ce que contient une ligne DVF
Une mutation (une vente) peut occuper plusieurs lignes : une par local et par parcelle. Chaque ligne indique la date de la mutation, sa nature (vente, VEFA, échange...), la valeur foncière (le prix total de la mutation, répété sur chaque ligne), l'adresse, la commune, le type de local (appartement, maison, dépendance, local d'activité), la surface réelle bâtie, le nombre de pièces principales, la surface du terrain et, pour les lots de copropriété, la surface Carrez.

## Ce que DVF ne contient pas
DVF ne contient pas les loyers : il ne recense que des transactions à titre onéreux (ventes). Les donations et les successions n'y figurent pas. La base ne donne aucune information sur l'acheteur ou le vendeur, ni sur l'étage, l'état du bien, le diagnostic de performance énergétique (DPE), la vue ou l'exposition. Les frais de notaire ne sont pas inclus dans la valeur foncière.

## Conditions d'utilisation
Les données DVF sont publiques mais leur réutilisation est encadrée : il est notamment interdit de chercher à réidentifier les personnes concernées par les transactions.

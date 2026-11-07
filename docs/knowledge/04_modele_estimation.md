# Modèle d'estimation de prix

## Principe
L'estimation est produite par un modèle de gradient boosting LightGBM entraîné sur les ventes nettoyées. Le modèle prédit le logarithme du prix au m², ce qui rend l'erreur relative : se tromper de 10 % pèse autant pour un bien à Paris que pour un bien en grande couronne. Le prix total estimé est le prix au m² prédit multiplié par la surface.

## Variables utilisées
Le modèle utilise le type de bien (appartement ou maison), la surface réelle bâtie, le nombre de pièces principales, la surface du terrain (maisons), le nombre de dépendances, la commune, le département, la position géographique (latitude et longitude) et la date de vente.

## Fourchette de prix
Deux modèles supplémentaires, entraînés sur les quantiles 10 % et 90 %, donnent une fourchette basse et une fourchette haute. Environ 80 % des ventes réelles doivent tomber dans cette fourchette : la couverture réellement observée est mesurée sur la période de test et suivie dans MLflow.

## Validation
Le modèle est évalué sur les 6 derniers mois de données, qu'il n'a jamais vus pendant l'entraînement, comme en conditions réelles. La métrique principale est la MdAPE (erreur absolue médiane en pourcentage) : une MdAPE de 8 % signifie que la moitié des estimations sont à moins de 8 % du vrai prix. Le modèle est comparé à une référence simple, la médiane de la commune pour le même type de bien. Une fois validé, il est réentraîné sur tout l'historique.

## Limites
L'estimation ne connaît ni l'étage, ni l'état du bien, ni le DPE, ni la vue, ni la présence d'un ascenseur ou d'un extérieur. Elle est indicative et ne remplace pas l'avis d'un professionnel. Elle est moins fiable dans les communes avec peu de ventes et pour les biens atypiques (très grandes surfaces, biens d'exception). Le modèle estime un prix au niveau de marché de la date la plus récente des données ; il ne prédit pas l'évolution future des prix.

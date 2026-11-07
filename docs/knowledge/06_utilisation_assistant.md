# Utiliser l'assistant

## Ce que l'assistant sait faire
L'assistant répond aux questions sur les ventes immobilières en Île-de-France à partir des données DVF nettoyées : prix médian au m² d'une commune ou d'un département, évolution des prix, nombre de ventes, comparaison entre communes, et estimation du prix d'un appartement ou d'une maison.

## Exemples de questions
« Quel est le prix médian au m² des appartements à Clichy ? », « Comment ont évolué les prix à Montreuil depuis 2021 ? », « Quelles sont les communes les plus chères des Hauts-de-Seine ? », « Combien vaut un 3 pièces de 60 m² à Versailles ? », « Pourquoi certaines ventes sont-elles exclues ? ».

## Comment l'assistant garantit ses chiffres
Chaque chiffre donné par l'assistant doit provenir d'un outil : une requête SQL en lecture seule sur l'entrepôt, le calcul de tendance, le modèle d'estimation ou cette documentation. Un contrôle automatique vérifie, après chaque réponse, que tous les nombres cités apparaissent dans les résultats des outils ; sinon l'assistant doit corriger sa réponse.

## Limites
L'assistant ne couvre que l'Île-de-France et la période présente dans les données. Il ne donne pas de prévision des prix futurs, ni de conseil financier ou juridique, et il ne connaît pas les loyers.

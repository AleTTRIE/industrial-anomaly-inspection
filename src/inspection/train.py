"""Train and test anomalib models on MVTec AD categories.

À ÉCRIRE (semaine 1). Consignes :

Objectif : une commande du type
    python -m inspection.train --model patchcore --category bottle
qui entraîne un modèle sur les images "good" d'une catégorie, l'évalue sur le
split de test (AUROC image et pixel) et, en option, l'exporte en ONNX.

Étapes à coder :
1. Lire les arguments en ligne de commande (module argparse) : modèle, catégorie,
   option d'export.
2. Créer le datamodule anomalib pour MVTec AD, en pointant vers DATA_ROOT
   (voir config.py).
3. Créer le modèle (PatchCore ou PaDiM) à partir de son nom.
4. Créer un Engine anomalib, puis appeler fit, puis test.
5. Si l'option est donnée, exporter le modèle en ONNX.

Indices :
- Commence par lire la doc "Getting started" d'anomalib pour la version installée
  (`pip show anomalib`) : les noms de classes ont changé entre les versions 1.x et 2.x.
- Fais d'abord marcher les étapes 2 à 4 dans un notebook ou un petit script jetable,
  puis range le code ici.
- Garde les imports d'anomalib à l'intérieur des fonctions : la CI n'installe pas
  anomalib, un import en haut du fichier ferait échouer les tests.

Questions à pouvoir expliquer ensuite :
- Pourquoi PatchCore n'a-t-il besoin que d'une seule "époque" d'entraînement ?
- Que contient exactement le split de test de MVTec AD ?
"""

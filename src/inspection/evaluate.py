"""Collect test-set scores and choose the decision threshold.

À ÉCRIRE (semaine 2). Consignes :

Objectif : une commande du type
    python -m inspection.evaluate --model patchcore --category bottle --cost-fn 10
qui récupère le score d'anomalie de chaque image de test, les sauvegarde en CSV
dans reports/, puis choisit le seuil OK/KO avec threshold.select_threshold et
l'écrit dans models/threshold.json (threshold.save_threshold).

Étapes à coder :
1. Recharger le modèle entraîné (checkpoint écrit par train.py dans RESULTS_DIR).
2. Lancer la prédiction sur le split de test et récupérer, pour chaque image,
   son score et son vrai label (1 = défectueuse, 0 = bonne).
3. Sauvegarder (score, label) dans un CSV.
4. Choisir et sauvegarder le seuil.

Piège à traiter (important, et bon sujet d'entretien) :
Si le seuil est choisi sur les mêmes images que celles sur lesquelles on mesure
ensuite le résultat, le résultat est trop optimiste (fuite de données).
MVTec AD n'a pas d'images défectueuses en validation : il faut donc découper
toi-même le split de test en deux parties (choix du seuil / mesure finale).
Réfléchis à comment découper sans déséquilibrer les types de défauts.

Questions à pouvoir expliquer ensuite :
- Pourquoi ne pas simplement prendre le seuil qui maximise le F1 ?
- Comment le seuil change-t-il si un défaut raté coûte 10 fois, puis 100 fois
  plus cher qu'une pièce bonne rejetée ?
"""

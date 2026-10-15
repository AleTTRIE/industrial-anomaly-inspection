"""CPU latency benchmark: PyTorch vs ONNX Runtime vs OpenVINO.

À ÉCRIRE (semaine 2). Consignes :

Objectif : mesurer le temps d'inférence par image (taille de lot 1, sur CPU) pour
chaque moteur, et écrire un tableau dans reports/latency.md.

Une mesure de latence sérieuse :
- fait quelques exécutions "à vide" avant de mesurer (échauffement) ;
- répète la mesure (au moins 100 fois) ;
- rapporte la médiane ET le 95e percentile, pas seulement la moyenne ;
- indique la machine (modèle de CPU) et ce qui est inclus dans le temps mesuré
  (le prétraitement de l'image est-il compté ?).

Indice : time.perf_counter() pour chronométrer.

Questions à pouvoir expliquer ensuite :
- Pourquoi le 95e percentile compte-t-il plus que la moyenne pour une ligne de
  production ?
- Pourquoi la première inférence est-elle beaucoup plus lente que les suivantes ?
"""

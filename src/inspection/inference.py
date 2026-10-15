"""ONNX Runtime inference, used by the API. No PyTorch dependency.

À ÉCRIRE (semaine 3). Consignes :

L'API (api.py) attend une classe AnomalyModel avec :
- un constructeur AnomalyModel(model_path) qui charge le fichier .onnx ;
- une méthode predict(image: PIL.Image) -> tuple[float, np.ndarray | None]
  qui renvoie (score d'anomalie de l'image, carte d'anomalie ou None).

Étapes à coder :
1. Charger le modèle avec onnxruntime.InferenceSession, sur CPU.
2. Afficher les noms des entrées et sorties du modèle exporté : c'est là que tu
   sauras où se trouvent le score et la carte d'anomalie.
3. Préparer l'image comme pendant l'entraînement (taille, format des canaux,
   normalisation), sous forme de tableau numpy de forme (1, 3, H, W).
4. Lancer l'inférence et extraire score + carte.

Piège : vérifie si le modèle exporté par anomalib contient déjà le
redimensionnement et la normalisation. Si oui et que tu normalises aussi ici,
les scores seront faux sans aucun message d'erreur. Teste en comparant tes scores
ONNX avec ceux d'anomalib sur quelques images.

Questions à pouvoir expliquer ensuite :
- Pourquoi servir le modèle avec ONNX Runtime plutôt qu'avec PyTorch ?
- Que se passe-t-il si l'image envoyée à l'API n'a pas la même taille ?
"""

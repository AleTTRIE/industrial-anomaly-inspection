# Séance 1 : 15 Octobre
- pourquoi du non supervisé plutôt qu'un classifieur ;
Parce ce que dans ce projet, les défauts sont beaucoup moins présent que les bons exemples. De plus, il existe plusieurs types/formes de défauts dont certains que l'on ne connait potentiellement pas encore donc nous n'allons nécessairement pas avoir assez d'exemple pour faire un entrainement supervisé tel que pour un classifieur. Un classifieur a besoin d'avoir vu un défaut (plusieurs fois) pour pouvoir le reconnaitre. Alors qu'avec le non supervisé, on lui apprend ce qui est normal, ce qui sera donc "pas normal" sera éloigné et identifiable.

- PatchCore en 3 lignes : mémoire de patchs, une seule époque, score par le max ;
Pour récupérer les informations d'une image, nous allons le passer dans un CNN où les couches sont gelés, on ne prend pas toutes les couches, car on ne veut pas classifier mais extraire l'information. Du au faite que ce soit gelé, le résultat est déterministe pour une image (Pas besoin de le faire plusieurs fois par image). Vu que l'on extrait le résultat sur une couche intermédiare, on obtient une grille de vecteur en 28x28 donc 784 vecteurs, appelé ici patch. Tous ces vecteurs sont gardés en mémoire pour savoir à quoi ressemble une pièce normal. On préfère ne pas tout garder car la recherche du plus proche voisin est lente et lourde en mémoire donc on garde environ 10% tout en essayant de garder une diversité. 
Pour savoir si une image test est normal ou pas, nous extrayons de la meme facon les patchs puis pour chacun d'entre eux, on regarde le patch en mémoire qui est le plus proche avec celui ci. On calcul ensuite la distance entre ces deux patchs, distance faible -> ressemble à quelque chose que l'on connait, distance forte -> jamais vu donc suspect. Pour l'image test, nous prenons la distance maximum parmi les patchs testés.

- pourquoi l'AUROC ne suffit pas pour la production ;
Le modèle utilisé donne un score pour chaque image, et pour chaque seuil possible, on regarde deux valeurs : le rappel (part des défauts détectés) et le taux de faux rejets (bonnes pièces vu comme défectueuses). La coube ROC compare ces deux valeurs pour chaque seuil et AUROC nous donne l'aire sous cette courbe. Elle se lit simplement : c'est la probabilité qu'une image défectueuse tirée au hasard ait un score plus élevé qu'une image bonne tirée au hasard. 0,5 correspond au hasard, 1 à un classement parfait.
Mais elle ne nous dit pas quel seuil choisir en production (car elle résume tous les seuils).
Un AUROC de 0.99 peut paraitre un super bon résultat mais il ne prend pas en compte le déséquilibre du problème. (Ex: 100k pièces, 0.1% défectueux, si on rejette simplement 2%, ca fait environ 2k pièces bonnes, ce qui est énorme, même si les 100 pièces défectueuses sont dans ces 2%).
Il ne prend pas en compte la différence d'impacte entre les erreurs. Une mauvaise pièce jugée bonne est envoyé chez le client, alors qu'une bonne pièce jugé défectueuse qui est vérifié manuelle est moins couteux. 


- le piège de l'accuracy : le modèle « toujours OK » obtient 99,9 % ;
Quand une classe est très rare, l'accuracy est dominée par la classe majoritaire. Elle peut être excellente pour un modèle qui ne détecte rien. On regarde plutot : le rappel, le taux de faux rejets ou la précision.

- LF / CRLF
La fin de ligne est un caractère invisible : LF (\n) sous Linux et macOS, CRLF (\r\n) sous Windows.
Ton code tourne au final sous Linux : dans l'image Docker, dans la CI et sur Cloud Run. Un script shell en CRLF y plante, avec une erreur du genre /bin/sh^M: bad interpreter, où le ^M est le \r en trop. Tu ne le vois pas dans l'éditeur, donc tu peux chercher longtemps.
La solution : .gitattributes avec * text=auto eol=lf. La règle est écrite dans le repo, donc elle s'applique à tout le monde, au lieu de dépendre de la configuration Git de chaque machine.
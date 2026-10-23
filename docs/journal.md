# Séance 1 : 15 et 16 Octobre
- pourquoi du non supervisé plutôt qu'un classifieur ;
Parce ce que dans ce projet, les défauts sont beaucoup moins présent que les bons exemples. De plus, il existe plusieurs types/formes de défauts dont certains que l'on ne connait potentiellement pas encore donc nous n'allons nécessairement pas avoir assez d'exemple pour faire un entrainement supervisé tel que pour un classifieur. Un classifieur a besoin d'avoir vu un défaut (plusieurs fois) pour pouvoir le reconnaitre. Alors qu'avec le non supervisé, on lui apprend ce qui est normal, ce qui sera donc "pas normal" sera éloigné et identifiable.

- PatchCore en 3 lignes : mémoire de patchs, une seule époque, score par le max ;
Pour récupérer les informations d'une image, nous allons le passer dans un CNN où les couches sont gelés, on ne prend pas toutes les couches, car on ne veut pas classifier mais extraire l'information. Du au faite que ce soit gelé, le résultat est déterministe pour une image (Pas besoin de le faire plusieurs fois par image). Vu que l'on extrait le résultat sur une couche intermédiare, on obtient une grille de vecteur en 32x32 donc 1024 vecteurs, appelé ici patch. Tous ces vecteurs sont gardés en mémoire pour savoir à quoi ressemble une pièce normal. On préfère ne pas tout garder car la recherche du plus proche voisin est lente et lourde en mémoire donc on garde environ 10% tout en essayant de garder une diversité. 
Pour savoir si une image test est normal ou pas, nous extrayons de la meme facon les patchs puis pour chacun d'entre eux, on regarde le patch en mémoire qui est le plus proche avec celui ci. On calcul ensuite la distance entre ces deux patchs, distance faible -> ressemble à quelque chose que l'on connait, distance forte -> jamais vu donc suspect. Pour l'image test, nous prenons la distance maximum parmi les patchs testés.
Info sup -> la carte d'anomalie : la grille des distances localise le défaut, et c'est elle qui sert à l'AUROC pixel

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

# Séance 2 : 20 au 22 octobre 2026, premier entraînement

## Mise en route
- Lire une trace d'erreur :
  (Par où commencer ? Comment repérer si l'erreur vient de mon code ou d'une librairie ?)
  Méthode, sur l'exemple du 404 :
  1. Lire la dernière ligne : type et message de l'erreur.
     `urllib.error.HTTPError: HTTP Error 404: Not Found` = le serveur répond que l'URL n'existe pas.
  2. Repérer à qui appartient chaque ligne grâce au chemin :
     `notebooks\` = mon code, `.venv\...\site-packages\` = librairies, `Python311\Lib\` = Python standard.
  3. Remonter jusqu'à la dernière ligne de mon code : c'est l'appel qui a mené à l'erreur, pas forcément le coupable.
     Ici `first_run.py`, line 19 : `engine.fit(...)`, appel correct.
  4. Lire les noms de fonctions entre les deux pour avoir l'histoire :
     `fit` → `prepare_data` → `download_and_extract` → `urlretrieve` → 404.
     Conclusion : anomalib essaie de télécharger MVTec via un lien mort codé en dur. Le problème ne vient pas de mon code.

  Deuxième exemple, le `MemoryError` : la trace commence par `multiprocessing\spawn.py` puis `first_run.py, line 1` (un import).
  C'est un worker lancé par spawn qui réimporte mon script : signe que le problème vient des workers.

- Le lien de téléchargement mort :
  (Pourquoi une URL écrite en dur finit par casser ? Comment j'ai contourné ?)
  L'URL pointait vers un hébergeur de fichiers. MVTec a refait son site et changé d'hébergement, donc l'ancien lien est mort ;
  "anomalib 2.7.0" a été publiée avec ce lien figé dans le code. La librairie ne peut pas suivre les changements d'un site externe.
  J'ai donc cherché un autre lien ou un lien vers une copie puis je l'ai download pour ne plus dépendre de lien externe.
  Car le prepare_data regarde en premier ce qu'il y a sur le disque avant d'aller télécharger le dataset.

- Les workers sous Windows :
  (Pourquoi 8 workers ont fait planter la RAM ? Pourquoi num_workers=0 est le bon choix ici ? Dans quel cas j'en remettrais ?)
  Un worker sous Windows est assez couteux car il lance un Python neuf et vide par workers, qui doivent chacun réexecuter pour rattraper le processus principal, ca implique également les importations qui sont TRES lourd. Ce qui explique pourquoi ca plante.
  La raison du "num_workers=0" est donc juste au dessus, au lieu de lancer un worker lent au démarage et lourd, j'utilise juste le processus principal mais c'est donc le GPU qui attend le CPU.
  Je mettrais plus de workers uniquement si il nous avons énormement de données, des gros traitement CPU ou pas mal d'époques pour l'entrainement.

## Calculs
- Patchs par image :
  (Image 256x256, layer2 divise par 8 → grille de ? x ? = ? patchs. Corrige ton 28x28 de la séance 1.)
  256x256 - "/8" -> 32x32 = 1024 patchs

- Taille de la banque mémoire :
  (209 images x ? patchs = ? vecteurs. 10 % de ça doit donner le 21 400 du log.)
  (? vecteurs x 1536 valeurs x 4 octets = ? Go. Comparé à la montée observée dans nvidia-smi ?)
  209 images x 1024 patchs = 214 016 vecteurs
  214 016 x 10% ~= 21 400 vecteurs gardés pour coreset
  214 016 vecteurs x 1 536 valeurs x 4 octets = 1 314 914 304 o ~= 1.3 Go (Se rapproche des +1Go vu dans nvidia-smi)
  Donc coreset est à 0.13 Go

## Ce que montre nvidia-smi
  (Pourquoi le GPU est à ~5 % pendant l'extraction ? Quelle phase le met à 95 % ?)
  Pendant l'extraction, le GPU est seulement à 5% car c'est la periode où l'on utilise le CPU, on crée les batchs en faisant les traitements nécessaires pour ensuite les envoyer au GPU. Dans le cas actuel, seul un processus principal tourne avec num_workers=0. Le GPU passe donc son temps à attendre le CPU.
  Pendant cette même extraction, la mémoire monte par paliers. La raison est que c'est la banque mémoire qui se remplit des vecteurs extraits des différentes images, et vu que ca se passe par batch, ca se fait par palier. Pendant cette phase, la banque mémoire est situé dans la VRAM (mémoire GPU). 
  Le but est de créer un coreset qui représente au mieux ce qui est valide/correct donc il nous faut un sous ensemble de patchs qui sont le plus représentatif de l'ensemble des patchs sinon nous aurons des faux rejet. Pour pouvoir valider un patch potentiel, il faut savoir si il est déjà bien représenté ou non, il faut donc le comparer avec les patchs choisis dans le coreset en construction, si sa distance minimum parmi les patchs choisis est basse, c'est qu'il est déjà bien représenté. Pour éviter de refaire les calculs de distances avec un coreset qui ne cesse de grandir, l'algorithme met à jour la distance minimale avec le nouveau patch choisi, on ne refait donc pas toutes la matrice des distances [dist_min = min(ancienne dist_min, distance au nouveau patch)] . L'algorithme doit donc regarder tous les patchs potentiels et pour chacun d'entre eux, il doit regarder la distance minimale parmi les patchs choisis ensuite il sélectionne le patch potentiel avec la plus grande distance minimale et l'ajoute au coreset. C'est donc assez long et couteux. Parlons chiffre : si nous devons garder 10% du total, ca fait 21.4k patchs à choisir donc 21.4k itérations aussi; dans chacune des itérations, nous devons actualiser la distance minimale envers le coreset avec le nouveau patch, donc 214k calculs de distance; soit 21.4k itérations de 214k comparaisons soit plus de 4.5 milliard de comparaisons. 
  Pendant la phase de coreset, vu que la banque mémoire est complète et en mémoire dans le GPU, les CPU ne sont plus utilisés. Le GPU tourne à 95%, car il effectue les 214k comparaisons à chaque itération.

## Le coreset coûte cher, et quadratiquement
  (bottle ~60 s, screw ~150 s alors qu'il n'y a que 1,5x plus d'images. Pourquoi ? Que se passe-t-il si je double les données ?)
  La formule f du nombre de comparaison en fonction de N, le nombre de patchs, est f(N) = r x N^2 (r est le ratio du coreset). 
  Partons du plus simple, si il y a 2x plus d'images, est ce qu'il y a 2x de comparaisons ? 
  Regardons : f(2N) = r x (2N)^2 = r x 4N^2 = 4 x r x N^2; Ca ne double pas, ca quadruple, l'évolution est donc quadratique, càd 10x plus d'images implique que ce sera 100x plus lent.
  Regardons bottle vs screw : 
  - 320 (img) x 1024 (patchs/img) = 327 680 patchs
  - 327k / 214k ~= 1.53, il y a donc environ 1.53x plus de patchs pour screw
  - 1.53^2 ~= 2.34, il y a donc environ 2.34x plus de comparaison pour screw
  - 60 x 2.34 = 140.4, ce qui se rapproche de 150s (Il manque de petit détail comme le fait que ce n'est pas parfaitement N^2 ou que l'on a calculé la durée lues à vue d'oeil mais l'ordre de grandeur est là)
  Et si le ratio du coreset passe de 10 % à 1 %, avec le même N ? Si on regarde la formule, on passerai de f(N)=0.1N^2 à f(N)=0.01N^2, on divise par 10 mais je rappelle que l'on perd en diversité, et le risque de faux rejet augmente fortement.

## Résultats bottle vs screw
| | bottle | screw |
|---|---|---|
| image_AUROC | 1.000 | 0.970 |
| image_F1 | 0.992 | 0.950 |
| pixel_AUROC | 0.986 | 0.989 |
| pixel_F1 | 0.727 | 0.369 |

- Pourquoi screw est plus dur :
  (Regarde train/good : qu'est-ce qui varie d'une vis à l'autre ? Taille des défauts ?)
  Les vis sont un problème plus difficile pour plusieurs raisons dans ce cas, elles ne sont pas tout le temps dans la meme position et les erreurs/défauts ne semble pas être de la meme taille et sont plus difficile à repérer que pour bottle. La taille des défauts est un problème car on passe à une carte de features/anomalie en 32x32 où chaque case représente 8x8 pixels, on perd donc logiquement de l'information dans cette moyenne de 64 pixels. Pourquoi la position variable gêne-t-elle ? Car une vis qui change meme un peu d'angle ne donne pas les mêmes patchs et donc ne se retrouvera pas forcement dans train/good et créera un faux rejet.

- AUROC pixel haute mais F1 pixel bas :
  (Quel lien avec le piège de l'accuracy de la séance 1 ?)
  Prenons un exemple : Une image de vis fait 256 × 256 = 65 536 pixels. Le défaut est une petite rayure de 20 × 20 = 400 pixels. La carte d'anomalie fait 32 × 32 et chaque case couvre 8 × 8 pixels. Une fois agrandie, elle est floue : la tache rouge déborde du défaut. Disons que le modèle marque une zone de 40 × 40 = 1 600 pixels, qui contient tout le défaut.
  - Rappel : quelle part des 400 pixels défectueux est détectée ?
    > R=400/400=1
  - Précision : parmi les 1 600 pixels marqués, combien sont vraiment défectueux ?
    > P=400/1600=.25
  - F1 = 2 × P × R / (P + R).
    > F1=(2x1x.25)/(1+.25)=.5/1.25=.4
  - Taux de faux positifs : 1 200 pixels bons marqués, sur combien de pixels bons au total ?
    > 1 200 / (65 536 − 400) = 1 200 / 65 136 ≈ 1,8 %
  On peut voir que le F1 à .369 de screw correspond à peu près au F1 calculé avec l'exemple et permet de nous donner une idée d'où vient le problème.
  Nous pouvons aussi comprendre pourquoi nous avons un aussi gros score pour AUROC, cela s'explique par le fait que le taux de pixel FP est minime et donc impacte peu l'AUROC.
  Ce qui nous permet de remettre à nouveau en question le 99.9% de la séance 1 où nous avions une valeur qui manquait d'informations et masqué la pertinence du système. Nous pouvons le voir ici avec la précision et l'AUROC, où 1200 erreurs de pixels pèsent 75% pour l'un et 1.8% pour l'autre. Ca dépend de quel point de vue on regarde, car l'on peut facilement noyer l'information comme on le voit juste avant avec l'AUROC qui divise les erreurs par tous les pixels bons (65 000, surtout du fond facile), alors que le F1 ne regarde que les pixels marqués et les vrais défauts.

- AUROC image = 1.0 mais F1 < 1 sur bottle :
  (Qu'est-ce que ça dit sur la différence entre "séparable" et "bien seuillé" ?)
  Reprenons un exemple : Le test bottle compte 63 défectueuses et 20 bonnes (chiffre de bottle). Suppose une seule bonne bouteille rejetée à tort.
  P=63/(63+1)=.9843, R=63/(63+0)=1, F1=2x1x.9843/(1+.9843)~=.992
  On retrouve le résultat obtenu par bottle.
  Différence entre "séparable" et "bien seuillé" :
  - Séparable, ce traduit par le fait que l'on peut différencier des groupes car ils ne s'entremele pas (AUROC=1) ou peu (AUROC<1) dans un espace (à une ou plusieurs dimensions).
  - Bien seuillé serait plus les lignes, droites, bordures qui sépare les différents groupes, donc bien le placer permettrer d'avoir des groupes plus ou moins pure, la pureté des groupes dépendrait du seuil utilisé.  Ici, en 1D, la frontière est un point : la valeur du seuil.
  Ex:
  score :  0.1  0.2  0.3  0.4  |  0.6  0.7  0.9
           bonnes              |  défectueuses
    - Avec un seuil à 0,5, il n'y a aucune erreur.
    - Avec un seuil à 0,35, la bonne à 0,4 est rejetée. Une bonne est rejetée, c'est le même type d'erreur que sur bottle.
    - Dans les deux cas, l'AUROC vaut 1,0 : le classement n'a pas changé.
  Laquelle des deux propriétés compte au final pour la production?
  > Ce serait le seuillage, car c'est lui qui permet de dire en fonction du score si une pièce est défectueuse ou non. Mais si il n'y avait pas de séparabilité, nous ne pourrions pas trouver de bon seuil qui permettrait de séparer les bonnes pièces des mauvaises. C'est notre model qui permet de les séparer mais c'est à nous de décider du seuil en fonction des coûts associés.

## Pièges à corriger plus tard
- Fuite du seuil :
  (val_split_mode = same_as_test : sur quelles images le seuil est réglé ? Pourquoi c'est un problème ?)
  Ici, avec same_as_test, la validation est une copie du test : le seuil a été réglé sur les mêmes images que celles qui mesurent le F1.
  Le seuil est basé sur les données de validations. Basé le seuil sur le test, biaserait les résultats vers le test, ce qui nous permettrait pas de savoir si le seuil est si bien que ça, car les mesures pour juger de si le système marche se base sur les données de test.
  Si une personne connait les questions de l'examen d'histoire en avance et a une bonne note, peut on vraiment dire qu'il maitrise le sujet ?
  L'AUROC ne serait pas impacté car il ne choisit pas de seuil, alors que le F1 se basera sur les résultats données avec le seuil utilisé, il sera donc optimiste.
  C'est le travail de la 2e semaine où nous séparerons les données validation des données test.

- Reproductibilité :
  (pixel_AUROC différent à la 6e décimale entre deux runs : pourquoi ? Que faire ?)
  Cela s'explique par le premier patch choisi, car si le premier patch est différent alors l'évolution du coreset est différent donc banque légèrement différente, impacte donc les distances au plus proche voisin, ce qui explique donc les scores différents. Pour pouvoir avoir les mêmes résultats, il faut les mêmes premiers patchs, ce qui peut être un paramètre tel que le seed. 
  Il existe une deuxième cause possible, que la graine ne corrige pas. Sur GPU, les additions en float32 ne se font pas toujours dans le même ordre, et (a + b) + c ≠ a + (b + c) à la dernière décimale près

## Questions ouvertes
- MVTec AD n'a pas de dossier de validation. Il faudra donc découper une partie du test pour en faire une. Qu'est-ce que ça coûte ?
- Pourquoi le F1 de bottle n'est pas à 1, alors que le seuil a été réglé sur le test ?
- Que donne l'AUPRO, la métrique pensée pour les petits défauts, plutôt que l'AUROC pixel ?
- Est ce que l'impacte du premier patch varie en fonction des problèmes tel que bottle vs screw ? Et en fonction par exemple du ratio pour le coreset ?
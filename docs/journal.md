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
  Le seuil est basé sur les données de validations. Basé le seuil sur le test, biaiserait les résultats vers le test, ce qui nous permettrait pas de savoir si le seuil est si bien que ça, car les mesures pour juger de si le système marche se base sur les données de test.
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





# Séance 3 : 23 au 27 octobre 2026, train.py et les 8 runs

## Ligne de commande
- argparse :
  (À quoi ça sert ? `required` ou `default` : comment je choisis ? Que fait `choices` ?)
  "argparse" permet de pouvoir passer des arguments/paramètres dans un script python.
  "required" fait en sorte que l'on est obligé de donner une valeur pour un argument du script.
  "default" fait en sorte que si il n'y a pas de valeur pour un argument, on met celui qui a été mis par défaut dans argparse.
  "required" vs "default" : cela dépend de l'argument, par exemple, une graine peut être choisie par défaut car son importance n'est pas la même que pour le choix du model utilisé ou du dataset utilisé où l'on va plutot utilisé "required" 
  `choices` permet de restreindre les choix possibles d'un argument.
- `python -m inspection.train` contre `python src\...\train.py` :
  (Différence ? Pourquoi `-m` ici ? Pourquoi `src.inspection` ne marche pas ?)
  Avec `python -m`, on va chercher à executer un module python, alors que sans le `-m`, on l'execute le script.
  `src` est un simple dossier de rangement, pas un paquet Python. C'est là où l'on va chercher les scripts où se trouve le code. C'est un standard.
- `python3` sous Windows :
  (Qu'est-ce qui s'est passé ?)
  Pour pouvoir lancer les différents modèles, ils nous faut les librairies et donc l'environnement sauf que `python3` n'utilise pas l'environnement contruit mais un autre donc il faut utiliser `python`.

## Organisation du code
- Une fonction, un rôle :
  (Pourquoi `train()` reçoit des valeurs et pas `args` ? Pourquoi le JSON est écrit dans `main()` ?)
  `train()` reçoit des valeurs et pas `args`, car ce n'est pas le rôle de train, son objectif est de lancer l'entrainement d'un modèle avec le dataset donné. De plus, il devient plus difficile d'utiliser `train()` dans d'autres contextes avec `args`.
  Dans le meme principe que dit précédemment, ce n'est pas le rôle de `train()` mais de l'appelant de décider de ce qu'il fait du résultat.
- Imports :
  (Lesquels en haut du fichier, lesquels dans `train()`, et pourquoi ?)
  On choisit d'importer les standards et les imports léger en haut du script car ils sont léger à importer alors que les autres plus lourd, on préfère les importer si c'est vraiment nécessaire. L'autre raison est que certain import ne s'installe pas avec le CI, ce qui créerait des erreurs et ne lancerait rien.
- Code jamais exécuté :
  (L'histoire de `model.export`, et la règle que j'en tire.)
  Nous utilisions une méthode qui n'existait pas donc plutot que de la laisser et créer des erreurs si utilisé, nous l'avons enlevé.
  code jamais exécuté = code cassé jusqu'à preuve du contraire, on le teste ou on le retire

## Reproductibilité
  (Expérience : deux runs avec la même graine, résultat ? Quelle cause de la séance 2 est confirmée ?)
  Deux runs avec les mêmes paramètres ainsi que la meme seed donnent les mêmes résultats, ce qui coincide avec l'hypothèse de la scéance 2 (dans "Pièges à corriger plus tard") où l'on supposait que le choix du premier patch impactait les résultats, alors nous avions supposé que la seed permettrait de résoudre ce porblème car elle permettrait de garder le meme permier patch.

## PatchCore contre PaDiM
  (Tableau des 8 runs. Pourquoi PaDiM s'effondre sur screw ? Pourquoi égalité sur carpet ? Quels écarts sont dans le bruit de la graine ? Temps d'entraînement ?)
  La raison de pourquoi PaDim s'effondre sur screw est en raison de son mécanisme où il doit modéliser chaque position de la grille par une gaussienne où pour screw, une case de l'image peut être du fond, un filetage ou bien la tête car les vis n'ont pas toujours les mêmes positions, il n'y a donc pas de "forme normal" par case. Ce qui est à l'inverse plutot bien gérer par PatchCore où on lui demande si il a déjà vu ce patch dans sa banque mémoire, ce qui n'implique pas la position.
  Comment expliquer que le problème de carpet renvoie une quasi égalité ? Cela s'explique par la texture, où les motifs se répètent partout dans l'image, donc chaque case sera très similaire. Ici, la position ne porte pas d'informations particulières, ce qui permet à PaDiM de rivaliser avec PatchCore.
  Nous avons pu remarquer avec le peu de run que l'on a fait avec des graines différentes, que celle-ci peut faire bouger les résultats d'environ 8e-3. Ce qui nous permet de dire que l'on ne peut pas dire quel modèle est meilleur pour bottle car la différence est inférieur à l'écart créé par le bruit, alors que ce n'est pas le cas pour screw où la différence est réelle. (FAUX, voir Plusieurs graines)
  PaDiM prend environ 1min, et PatchCore de 2 à 4min.

## Plusieurs graines
| Model | Category | Type | Mean Image AUROC ± Std | Mean Pixel AUROC ± Std |
|-------|----------|------|------------------------|------------------------|
| padim | bottle | object | 0.997 ± 0.002 | 0.979 ± 0.001 |
| patchcore | bottle | object | 1.000 ± 0.000 | 0.986 ± 0.000 |
| padim | carpet | texture | 0.969 ± 0.013 | 0.989 ± 0.001 |
| patchcore | carpet | texture | 0.987 ± 0.001 | 0.991 ± 0.000 |
| padim | metal_nut | object | 0.956 ± 0.011 | 0.953 ± 0.008 |
| patchcore | metal_nut | object | 0.998 ± 0.001 | 0.987 ± 0.000 |
| padim | screw | object | 0.792 ± 0.038 | 0.977 ± 0.002 |
| patchcore | screw | object | 0.964 ± 0.003 | 0.989 ± 0.000 |

- Pourquoi refaire les runs avec plusieurs graines ?
  (Qu'est-ce qu'une seule paire de runs ne permettait pas de conclure ?)
  Refaire plusieurs runs nous permet de savoir l'impacte qu'à l'aléatoire. Une graine permet également de reproduire une run à l'identique, là où l'aléatoire a un impacte. Faire plusieurs runs nous permet également d'estimer le bruit et de connaitre de façon plus général les capacités du modèles.
- Ce que j'ai mesuré :
  (5 graines, quelles métriques, quel script ? Moyenne et écart-type.)
  Nous avons utilisé 5 graines, deux statistiques (Moyenne et Ecart-type) sur deux métriques AUROC image et pixel, les deux modèles ainsi que 4 datasets, généré les 40 runs via `train.py` et analyser le tout via `summarize.py`. 
  Nous pouvons remarquer en regardant le tableau que PatchCore est beaucoup plus stable vis-à-vis de l'aléatoire, ce qui n'est pas le cas de PaDiM (de 0 à 0.003 pour PatchCore et de 0.002 à 0.038 pour PaDiM). Meme PaDiM reste assez stable sur AUROC pixel (de 0 pour PatchCore et de 0.001 à 0.008 pour PaDiM).
  Nous pouvons également remarquer que selon le problème/dataset, l'aléatoire a une part plus où moins importante, ce qui peut paraitre logique avec la difficulté des problèmes et hypothétiquement la taille des datasets.
  Il est important de dire que les résultats que nous avions pour la graine 42 ne sont pas forcement les moyennes obtenues des différentes graines. 
- `stdev` contre `pstdev` :
  (Quelle différence ? Pourquoi n − 1 ici ?)
  `stdev` travaille sur un échantillon (c'est stdev qui utilise "n-1").
  `pstdev` travaille sur une population.
  La raison du n-1 est que la moyenne est calculé sur ces mêmes graines, les valeurs sont donc en moyenne plus proches de leur propre moyenne que de la vraie, et diviser par n sous-estime l'écart. Diviser par n − 1 corrige ce biais.
- Résultats :
  (Écart-type de PatchCore contre PaDiM ? Qu'est-ce qui a changé par rapport aux conclusions avec la seule graine 42 : carpet, estimation du bruit ?)
  Pour les textures tel que `carpet`, PatchCore est devant et plus stable pour image AUROC, PaDiM obtient des résultats qui varie plus. Alors que pour pixel AUROC, les deux modèles sont à peu près équivalent, les motifs qui se répètent au travers de l'images donc la position dans l'image n'a que peu d'influence.
  Ces résultats ne sont pas les mêmes qu'avec la gaine 42 seule où l'on concluait sur une quasi-égalité.
- Pourquoi PaDiM est-il instable ?
  (Ce que fait `torch.randperm` dans le code de PaDiM. Pourquoi PatchCore est-il stable ?)
  Pour PaDiM, 100 dimensions sont gardées au hasard parmi 448, sachant que certaines dimensions sont plus au moins informatives, leur influence varie et impacte les résultats. Quant à PatchCore, la seule vrai part de hasard est le premier patch choisi pour le coreset, mais son effet diminue en fonction du nombre de patchs gardés pour le coreset, par exemple, quelle est l'influence d'un patch parmi 20 000 d'entre eux (bottle). 
- Comparer deux modèles :
  (La méthode écart contre bruit, sur bottle et screw.)
  Changer la seed fait bouger les scores de 0 à 0.003 pour PatchCore et de 0.002 à 0.038 pour PaDiM, donc:
  - bottle (1.000 ± 0.000 vs 0.997 ± 0.002): la différence (0.003) est environ presque celle du bruit, donc aucun des modèles ne peuvent être différenciés;
  - screw (0.964 ± 0.003 vs 0.792 ± 0.038): la différence (0.17) est plus de 4 fois le bruit, donc la différence est réelle.
- Erreur de comparaison :
  (Les backbones par défaut sont différents. Qu'est-ce que ça change à mes conclusions ?)
  Les deux modèles n'utilisent pas la même base pour représenter les images donc nécessairement, il est plus difficile de comparer les méthodes à armes égales. On ne compare plus seulement la méthode pour retrouver l'erreur mais aussi la façon de représenter les images et leurs erreurs. Donc 2 problèmes et pas 1.

## Questions ouvertes
- Comparer PatchCore à PaDiM avec `backbone="wide_resnet50_2"`.
- 5 graines suffisent-elles ?





# Séance 4 : 29 octobre 2026, choisir le seuil

## Pourquoi un split de validation
- D'où viennent les images pour choisir le seuil ?
  (Que contient le train de MVTec ? Où sont les seules images défectueuses ?)
  Actuellement les images pour choisir le seuil viennent du dataset de test. 
  Les images défectueuses sont dans le dataset test et pas train, car le dataset train est utilisé pour définir ce qu'est une bonne pièce, en mettre des mauvaises ferait que des mauvaises pièces seraient jugées comme bonnes.
- Pourquoi les retirer de la mesure finale ?
  (Lien avec la fuite `same_as_test` de la séance 2.)
  Il faut retirer les images qui ont servi au choix du seuil car ce choix est biaisé et donnera un resultat optimiste.
  L'idée serait donc de créer un dataset de validation extrait du dataset de test. ("extrait" donc plus dans le dataset de test)
- Le prix à payer :
  (Qu'est-ce qu'on perd en taillant la validation dans le test ?)
  Le coût, c'est de réduire les données utilisées pour juger le modèle. Par exemple, avec 20 good, le taux de faux rejet évolue de 5% en fonction du nombre de faux rejet alors qu'avec 10 good, ce serait 10%, donc on aurait une analyse moins précise du modèle. Ce serait pareil pour les autres métriques.

## Stratifier
- Le risque d'un découpage au hasard :
  (Un défaut rare entièrement d'un côté : que se passe-t-il dans chacun des deux cas ?)
  Un découpage au hasard pourrait par exemple mettre de côté un défaut rare que ce soit dans le dataset de validation ou de test, ce qui causerait un mauvais choix de seuil ou une mesure imprécise du modèle. 
- Stratifier sur quoi ?
  (Quels groupes ? Quelle fonction en Python ?)
  Pour éviter d'oublier des défauts dans un dataset, il faudrait s'assurer de une moitié de chaque groupe pour chaque dataset où un groupe serait soit un type de défauts soit les bonnes pièces.
  La fonction utilisée serait `sklearn.model_selection.train_test_split(..., stratify=types, random_state=...)`
- Vocabulaire :
  (Positif et négatif dans ce projet.)
  Dans ce projet, nous cherchons à détecter les anomalies, donc les positifs sont les pièces défectueuses et les négatifs sont les bonnes pièces.

## Petits effectifs
- Que vaut une bonne pièce rejetée dans le taux de faux rejets ?
  (Avec 20 good, puis avec 10 good.)
  Un faux rejet couterait 10% sur le taux de faux rejet avec 10 good, contre 5% avec 20 good.
- Ce que j'en tire pour mes résultats :
  (Paliers, comptes avec les taux, pareil pour le rappel ?)
  Moins de données rend moins précis l'analyse du modèle. Plus y a de bonnes données variées, plus l'analyse sera précise.
  Avec 10 good, le taux de faux rejets évolue de 10% par paliers, donc si on a 0 faux rejets dans notre analyse, on ne sait pas différencier 0% de 8%.
  Avec le rappel, le problème est plus fin car nous avons plus de données sur les pièces défectueuses dans le dataset de test. Avec environ 31 mauvaises pièces, on aurait des paliers de 3%.


## Le seuil par le coût
- La règle de décision :
  (Quand une pièce est-elle KO ? Que fait baisser le seuil ?)
  Une pièce est dite KO quand son score est supérieur ou égal au seuil choisi, car plus on se rapproche de 0 plus on ressemble à une bonne pièce.
  Baisser le seuil impliquerait d'accepter plus de faux rejet (rejet de bonnes pièces) mais rejeter aussi plus de mauvaises pièces.
- `cost_fn=10` contre `cost_fn=1` :
  (Le seuil monte ou descend ? Pourquoi ?)
  Avec cost_fn=10 contre cost_fn=1, le seuil va privilégier le fait de corriger les défauts non détectés par rapport aux bonnes pièces rejetées, le seuil descend donc.
- Pourquoi pas le seuil qui maximise le F1 ?
  (Que suppose le F1 sur le coût des deux erreurs ?)
  `F1 = 2·TP / (2·TP + FP + FN)`, où FP sont les bonnes pièces rejetées et les FN sont les défauts ratés.
  Dans le dénominateur de `F1`, `FP` et `FN` comptent pour 1 et `TP` 2, alors `FP` et `FN` ont le même poids, 1, donc cela implique que rater un défaut ou rejeter une bonne pièce fait baisser le F1 exactement autant. Cela suppose que les deux types d'erreurs se valent alors qu'en production, on préfère éviter plus les défauts ratés que les bonnes pièces rejetées.
  En conclusion, le F1 suppose `cost_fn = cost_fp`, alors que la règle par le coût nous laisse fixer ce rapport (10, 100…).
  Autre limite : L'immense majorité des pièces sont bonnes (les TN). F1 ne le prend pas en compte. Faire passer plus de bonnes pièces ne change rien au F1.

## Le seuil collé au défaut
- Le bug :
  (Quels étaient les seuils candidats ? Quel seuil était choisi sur bottle ? Que devient un nouveau défaut à 37 ?)
  Les seuils candidats précedemment étaient les scores uniques des différentes pièces.
  Le seuil qui était choisi était le score minimum des pièces défectueuses (38), cela impliquait qu'une pièce défectueuse avec un score légèrement plus faible (exemple 37), meme minime, était comptée comme non défectueuse. 
- Pourquoi c'est le pire côté :
  (Avec `cost_fn=10`, de quel côté faut-il garder de la marge ?)
  En production, on préfère rejeter des bonnes pièces qu'accepter des pièces défectueuses mais le choix de seuil actuel faisait l'inverse (score minimum des pièces défectueuses). Il faut plutot un seuil qui s'approche du score maximum des bonnes pièces.
- Pourquoi les tests ne l'ont pas vu :
  (Qu'est-ce qu'ils vérifiaient ? Qu'est-ce qu'ils ne vérifiaient pas ?)
  Ils vérifiaient que les seuils fonctionnaient mais ils ne regardaient pas la cohérence du seuil. Les tests actuellement regardaient seulement le coût, le rappel et les faux rejets sur les données utilisées pour choisir le seuil. Sauf que dans l'intervalle de bottle par exemple, toutes les valeurs entre 28.4 et 38 nous donnaient un même cout, il ne prenait pas en compte l'existance d'une marge en cas de nouvelles pièces. 
- La correction :
  (Milieux entre scores uniques, `-inf` et `+inf` : à quoi sert chacun ?)
  Ca nous permet d'avoir les cas où tout est refusé (`-inf`) ou tout est accepté (`+inf`). 
  Nous avons pris plutot le milieu entre les différents candidats pour pouvoir avoir une marge dans le cas où il y aurait une nouvelle pièce.
- Écrire le test avant :
  (Pourquoi vérifier qu'il échoue avec l'ancien code ?)
  Pour prouver que l'ancienne méthode ne marchait pas dans ce cas, donc également que le test est capable de trouver le bug, et pour vérifier que notre nouvelle méthode marche. 
- Pièges numpy rencontrés :
  (`np.append` avec 3 valeurs)
  `np.append(arr, values, axis)` ne prend que 2 valeurs à coller : la 3e (`np.inf`) était prise comme axe, d'où l'erreur. On utilise `np.concatenate([[-np.inf], scores, [np.inf]])`.
  `a[:-1] + a[1:] * 0.5` calcule d'abord la multiplication : il faut des parenthèses, `(a[:-1] + a[1:]) * 0.5`.

## Proportion de défauts : validation contre ligne
- Sur la validation :
  (Combien de défauts pour 10 bonnes ? Un défaut pèse combien de bonnes, en nombre d'exemples ?)
  Il y a 31~32 défauts pour 10 bonnes. On a donc 3 défauts pour 1 bonnes, soit les défauts sont trois plus nombreux.
- Sur une ligne :
  (Mêmes questions, avec 1 défaut pour 100 pièces.)
  On aurait 1 défaut pour 99 bonnes, donc à l'inverse ici, les bonnes sont 99 fois plus nombreuses.
- Conséquence sur le seuil :
  (`cost_fn × FN + cost_fp × FP` compte les erreurs de la validation. Le seuil choisi est-il trop bas ou trop haut pour la ligne ? Pourquoi ?)
  Le problème de ce deséquilibre est qu'il impacte la fonction de cout, car la part de bonnes pièces est beaucoup plus élevé et potentiellement il y aura plus de FP, ce qui rend cela moins comparable. Donc si il y a trop de pièces défectueuses, le seuil serait potentiellement trop bas, et à l'inverse, si il y a trop de bonnes pièces, le seuil serait trop haut, car le déséquilibre des classes n'est pas pris en compte.
  Exemple: Le modèle rate 10 % des défauts et rejette 10 % des bonnes, avec cost_fn=10 et cost_fp=1 :
  |                   | Validation (31 défauts, 10 bonnes) | Ligne (10 défauts, 990 bonnes) |
  |-------------------|------------------------------------|--------------------------------|
  | Coût des FN       | 3*10=30                            | 1*10=10                        |
  | Coût des FP       | 1*1=1                              | 99*1=99                        |
  | Erreur qui domine | FN très largement                  | FP assez largement             |
  Sur la validation, les FN dominent donc le seuil descend ; sur la ligne, ce sont les FP. Le seuil choisi sur la validation est donc trop bas pour la ligne.
- Comment corriger :
  (Idée à trouver : remplacer les comptes par des taux ? Où intervient la proportion de défauts ?)
  L'idée potentielle serait d'utiliser le taux de pièces défectueuses acceptés pour FN et le taux de bonnes pièces refusées pour FP. Ce qui balencerait un peu plus même si, le nombre de pièces défectueuses comparées aux bonnes (et inversement) aura quand meme un impact. Pareil que sur un autre problème, la précision, 10 pièces veut dire des paliers de 10%.
  Cela était pour corriger le déséquilibre des données, ce qu'il manque c'est l'information de la ligne.
  Pour 1000 pièces : 1000·p défauts, dont 1000·p·taux_FN ratés ; 1000·(1−p) bonnes, dont 1000·(1−p)·taux_FP rejetées.
  Coût par pièce : `cost_fn × p × taux_FN + cost_fp × (1 − p) × taux_FP`.
  Les taux retirent le déséquilibre du test, p remet celui de la ligne. p est une donnée de production, pas un réglage du modèle.
  Avec p = 0.5, on retrouve les taux seuls : ça suppose 50 % de défauts.
  Avec p = 0.01, cost_fn = 10 : poids 100 (FN) contre 990 (FP), donc le seuil monte.
  Les deux erreurs pèsent autant quand cost_fn / cost_fp = (1 − p) / p, soit 99 pour p = 0.01.
  Limite : avec 10 bonnes pièces, le taux de faux rejets avance par paliers de 10 %.

## Récupérer les scores
- Le checkpoint :
  (Où anomalib l'écrit-il ? Comment le recharger ?)
  anomalib l'écrit dans `results/<Modèle>/MVTecAD/<catégorie>/vN/weights/lightning/model.ckpt`, mais rien ne relie un vN à sa graine. On réentraîne donc dans le script (~1 min sur bottle)
- Ce que renvoie `engine.predict` :
  (Type, champs utiles : score, label, chemin. Comment je retrouve le type de défaut ?)
  Une liste d'`ImageBatch` (3 batches pour les 83 images de bottle). Champs utiles : `image_path` (liste de str), `gt_label` (tensor de booléens, converti avec `.int().tolist()`), `pred_score` (tensor de float). Le type de défaut est le dossier parent : `Path(path).parent.name`. `pred_label` est à ignorer : il vient du seuil fuité.
- Normalisation d'anomalib
  Par défaut, le score est centré sur le seuil fuité (0.5 = son seuil) et coupé à [0, 1], d'où les 1.0000. On la désactive avec `PostProcessor(enable_normalization=False)`. Scores bruts sur bottle : bonnes au max à 28.4, défauts au min à 38.0, donc parfaitement séparable.

## Questions ouvertes
- Est ce que stratifier sur les positions des bonnes pièces quand nécessaire serait une bonne idée ? (tel que `screw`)
- Décaler le seuil selon le coût en modélisant les distributions de scores.
- Comment connaître p en pratique, et que faire s'il change ?





# Séance 5 : 4 au 6 novembre 2026, un seuil mesuré honnêtement

## `defect_rate` dans threshold.py
- Valeur par défaut :
  (`None`, 0.5 ou 0.01 ? Que suppose `None` ?)
  `None` par défaut, car p est là comme paramètre pour ajouter du réalisme : c'est une option pour confronter le laboratoire à la réalité terrain. Avec `None`, le coût reste en comptes, ce qui suppose que la ligne a la même proportion de défauts que la validation (environ 3 défauts pour 1 bonne). `None` n'est donc pas neutre.
- p = 0 ou p = 1 :
  (Alerte ou erreur ? Que se passe-t-il à p = 0 ?)
  Mon premier réflexe était d'autoriser avec une alerte. Mais à p = 0, le coût se réduit à `cost_fp × taux_FP`, donc le seuil optimal est `+inf` : on ne rejette rien. En batch, personne ne lit l'alerte, `+inf` est écrit dans `threshold.json` et la production accepte tout. On lève donc un `ValueError` si p n'est pas strictement entre 0 et 1.
- Ce que garde `threshold.json` :
  (Pourquoi stocker `defect_rate`, même à `None` ? Pourquoi sans valeur par défaut dans `save_threshold` ?)
  Mieux vaut plus de détails que pas assez : on garde tout ce qui a servi à choisir le seuil (`cost_fn`, `cost_fp`, `defect_rate`). `null` veut dire « coût en comptes ». Le paramètre est obligatoire pour qu'un oubli donne une erreur, au lieu d'un fichier qui mentirait.
- Le calcul à la main :
  (Données du test, `cost_fn=10`. Quel seuil gagne à p = 0.5 ? À p = 0.01 ?)
  | Seuil | taux_FN | taux_FP | coût p = 0.5 | coût p = 0.01 |
  |-------|---------|---------|--------------|---------------|
  | 0.225 | 0       | 0.5     | 0.25         | 0.495         |
  | 0.35  | 0.5     | 0.25    | 2.625        | 0.2975        |
  | 0.65  | 0.5     | 0       | 2.5          | 0.05          |
  À p = 0.5, 0.225 gagne avec un rappel de 1. À p = 0.01, 0.65 gagne avec un rappel de 0.5 : rater la moitié des défauts coûte moins cher que rejeter 25 % des bonnes pièces sur une ligne à 99 % de bonnes.
- Un test vert qui ne prouve rien :
  (Pourquoi `default.cost != with_defect_rate.cost` ne suffisait pas ?)
  Le test passait avec et sans le bug. Coût de `None` pour le seuil choisi (FN = 0, FP = 2 sur 4) : 2.0 avec le code correct (comptes), 0.5 avec le code buggé (taux). Avec `defect_rate=0.5`, c'est 0.25. Les deux versions sont différentes de 0.25, donc `!=` ne distinguait rien. On vérifie maintenant la valeur exacte, `default.cost == 2.0` : avec le bug, le test échoue. Un test doit échouer si le code est faux, d'où l'intérêt de le voir échouer au moins une fois.
- Pièges Python :
  (`== None`, `payload.get`, `==` sur des flottants)
  Le standard python pour vérifier l'égalité avec `None` est `... is None` et pas `... == None` (ce que j'avais mis à la base).
  Quand on voulait récupérer une valeur, si elle était manquante, on voulait une erreur, ce qui n'est pas le cas avec `payload.get` qui renvoie `None` quand la valeur est absente.
  Un float est stocké en binaire sur 53 bits. 1/3 a une écriture binaire infinie, donc il est arrondi. Selon le calcul, l'arrondi tombe différemment : `2/3` donne 0.6666666666666666 et `1 - 1/3` donne 0.6666666666666667. `==` les dit différents. `pytest.approx` compare avec une tolérance. Pour des valeurs exactes en binaire (1/4, 1/2) ou des entiers, `==` suffit.

## Le découpage validation / test
- Le premier bug :
  (Qu'est-ce que je découpais ? Pourquoi chaque moitié contenait les 160 images ?)
  Je voulais séparer le dataset test initial en deux autres datasets, un de validation et un de test. Pour cela, je prenais la liste des défauts pour les séparer en deux sauf que cela ne permet pas de les séparer car chaque sub list contenait ces mêmes défauts, donc les deux datasets prenaient tous les éléments du dataset initial. Les séparer seulement par les types de défauts étaient donc une erreur bête.
- Une graine séparée pour le découpage :
  (Pourquoi pas la même que celle du modèle ?)
  Avec deux graines séparées, je peux garder le modèle fixe et ne faire varier que le découpage, pour mesurer à quel point le seuil dépend des images tombées en validation.
- Les groupes impairs :
  (41, 25, 23 images : comment se répartissent-elles ?)
  Une moitié prend une image de plus : 41 good → 20/21, 25 scratch_neck → 13/12, 23 → 11/12. Un écart d'une image sur 12 (environ 4 %), sans conséquence.

## `evaluate_threshold`
- Pourquoi dans threshold.py :
  (Que risque-t-on avec deux copies de la règle `>=` ?)
  Si l'on fait deux fois la même chose dans deux fichiers différents et que l'on veut modifier la méthode, on peut le corriger dans un fichier mais oublier de le faire dans l'autre, ce qui peut causer différents problèmes tel que de faux tests ou mauvais résultats.
- Les tests :
  (Pourquoi « FN + TP = nombre de défauts » est un test faible ? Pourquoi FN et FP doivent être différents ? Le cas limite ?)
  « FN + TP = nombre de défauts » est vrai pour n'importe quel seuil, et même avec une règle fausse (`>` au lieu de `>=`, ou FN et FP inversés) : il ne peut donc pas détecter ces bugs. On vérifie plutôt des comptes exacts calculés à la main. FN et FP doivent avoir des valeurs différentes (1 et 2) : sinon, une inversion donnerait le même résultat et passerait inaperçue. Le cas limite (un score exactement égal au seuil doit être KO) est le seul test qui distingue `>=` de `>`.

## Premier résultat honnête (split_seed 42)
- Les chiffres :
  | | Seuil | Rappel val → test | Faux rejets val → test | Coût val → test |
  |---|---|---|---|---|
  | `None` | 30.57 | 1.000 → 0.932 (FN 0 → 4) | 25 % → 14 % (FP 5 → 3) | 5 → 43 |
  | `0.01` | 32.90 | 0.850 → 0.797 (FN 9 → 12) | 0 % → 9.5 % (FP 0 → 2) | 0.015 → 0.115 |
- Mes prédictions et ce qui s'est passé :
  (La pièce à 35.78 ? Faux rejets plus élevés sur le test ?)
  La bonne pièce à 35.78 risquait de devenir un faux rejet : confirmé avec 0.01, le seuil est collé au good max de la validation (32.83). Je pensais que les faux rejets seraient plus élevés sur le test, car good max y est plus haut et un faux rejet coûte cher avec peu de valeurs. Vrai avec 0.01 (0 → 2), faux avec `None` (5 → 3) : le seuil 30.57 est bas, en plein dans la masse des bonnes pièces (moyenne ~29). Le nombre de bonnes rejetées dépend alors du hasard du découpage, et 2 pièces d'écart suffisent à passer de 25 % à 14 %.
- Ce que ça montre :
  (Combien de fois plus cher sur le test ?)
  Nous avons pu donc confirmer que le coût mesuré sur la validation était beaucoup trop optimiste, car nous pouvons le voir sur le tableau ci dessus, le coût est au moins 7 fois plus élevé pour le dataset de test que pour le dataset de validation quand on choisit le seuil sur le dataset de validation.

## 20 découpages
- Les chiffres (moyenne, min–max sur le test) :
  | cost_fn | p | Seuil | Rappel | Faux rejets |
  |---|---|---|---|---|
  | 10 | None | 29.73 (29.46–30.38) | 0.987 (0.949–1.000) | 27.5 % (19.0–45.0) |
  | 10 | 0.01 | 34.50 (31.98–35.97) | 0.737 (0.550–0.898) | 6.6 % (0.0–25.0) |
  | 100 | None | 29.73 (29.46–30.38) | 0.987 (0.949–1.000) | 27.5 % (19.0–45.0) |
  | 100 | 0.01 | 31.96 (29.94–33.14) | 0.875 (0.797–0.983) | 14.1 % (0.0–28.6) |
- Mes prédictions :
  (Quel seuil varie le plus ? Que fait `cost_fn=100` ?)
  Le seuil varie plus avec 0.01, car il est collé au good max : confirmé (écart-type 1.50 contre 0.26). La vraie raison : c'est l'extrême de 20 bonnes pièces contre celui de 60 défauts. Avec `cost_fn=100`, le seuil baisse et les faux rejets montent, car on accepte moins de pièces défectueuses et on compense par des faux rejets : confirmé (34.5 → 32.0, 6.6 % → 14.1 %).
- Le découpage 42 :
  (Était-il représentatif ?)
  Le test avec une seule seed (42) n'était pas du tout représentatif, car nous avions eu 30.567 comme seuil avec `None`, ce qui est plus élevé que le maximum trouvé sur les 20 seeds utilisées juste après. Et avec `0.01`, il nous donnait 32.9 alors que la moyenne des 20 seeds est 34.5. Si nous en étions resté là, nous aurions fait des conclusions hâtives et basées sur un tirage atypique
- `None` ne bouge pas avec `cost_fn` :
  (Pourquoi le même seuil avec 10 et 100 ?)
  Avec `cost_fn=10`, le rappel de validation vaut déjà 1.0 : FN = 0. Le seuil est juste sous le défaut le plus bas de la validation. Augmenter `cost_fn` multiplie un FN qui vaut 0, donc le coût de ce seuil ne change pas. Et le descendre encore ne rattraperait aucun défaut, il ajouterait seulement des faux rejets. Le seuil reste donc le même pour tout cost_fn assez grand. 

## Petits effectifs, en vrai
- Ce qui est fiable :
  (Paliers du rappel et des faux rejets)
  On a plus de précision sur les pièces défectueuses, car les paliers sont plus petits : 1 défaut = 1.7 %, 1 bonne pièce = 5 %. Il est donc assez dure de tirer des conclusions sur les performances du modèle, notamment surtout en ce qui concerne les infos sur les bonnes pièces, en prenant en compte les cas atypiques et le palier élevé, les résultats bougent énormement. Pareil pour les infos sur les défauts même si le palier est plus faible. 
- La règle de trois :
  (Que veut dire 0 faux rejet sur 20 ?)
  Avec 0 erreur sur n pièces, on peut seulement dire que le vrai taux est inférieur à 3/n, avec 95 % de confiance. 0 faux rejet sur 20 veut donc dire « moins de 15 % », pas « 0 % ». D'où vient le 3 : si le vrai taux est p, la probabilité de ne voir aucune erreur sur n pièces est (1 − p)^n ≈ e^(−pn). Elle tombe à 5 % quand pn ≈ 3, car ln(0.05) ≈ −3.

## Questions ouvertes
- Peut-on déployer PatchCore sur screw avec cost_fn=100 et p=0.01 (12.5 % de défauts ratés, 14 % de faux rejets) ?
- Mettre `cost_fn`, `cost_fp` et `defect_rate` dans `OperatingPoint`, et les retirer des paramètres de `save_threshold`.
- Sauvegarder les scores pour ne plus réentraîner à chaque lancement.





# Séance 6 : 10 au 12 novembre 2026, evaluate.py

## Le cache des scores
- Pourquoi un cache :
  (Combien coûte un recalcul des scores ? Qu'est-ce qui change d'un appel à l'autre, et qu'est-ce qui ne change pas ?)
  En fonction des modèles et données utilisées, lancer le calcul du score peut etre plusieurs minutes. En gardant les mêmes paramètres et en utilisant une meme seed, on peut relancer exactement le même calcul et avoir les mêmes scores/résultats, donc pour éviter de le recalculer, pourquoi ne pas sauvegarder le score, c'est le cache.  
- Où ranger les fichiers :
  (Dans results/ directement, ou dans un sous-dossier ? Dans git ou pas ?)
  Il ne faut pas les mettre directement dans results/, où sont déjà les *_results.json de la semaine 1. J'avais proposé results/saved_runs/ : on a pris results/scores/. Ces sauvegardes ne vont évidemment pas sur git (elles contiennent en plus des chemins absolus vers mon disque).
- Le nom du fichier :
  (Que doit contenir le nom ?)
  Tout ce qui fait varier les scores : modèle, catégorie, graine du modèle. Sinon, un second lancement relit les scores du premier sans rien dire. split_seed n'y est pas, car il ne change pas les scores, seulement leur découpage. La normalisation est fixée à `false`, notée dans le fichier, et marqué `_raw` dans le nom du fichier (car lui aussi change les valeurs).
- Le piège JSON :
  (JSON relit les tuples comme des listes, et `set` refuse les listes. Comment corriger ?)
  J'avais proposé d'enregistrer {validation: [...], test: [...]}. Mais le cache garde les 160 scores avant découpage, sinon il faudrait un fichier par split_seed, et la séparation en validation et test est suffisamment rapide. On reconstruit donc les tuples à la relecture, dans `load_scores` qui lit le JSON, et récupère les 160 scores qui sont représentés par une liste de dictionnaires (nécessaire pour JSON), où chaque tuple est recréé via le dictionnaire correspondant.
- Format :
  (Pourquoi du JSON suffit pour des floats ?)
  Python écrit un float en JSON avec assez de chiffres pour retrouver exactement la même valeur à la relecture : les chiffres sont identiques au chiffre près, pas seulement proches. Un CSV en `:.3f` aurait arrondi les scores, et le seuil aurait pu changer.
- Ce qu'on enregistre en plus des scores :
  (Pourquoi `normalize: false` et la version d'anomalib ? Pourquoi `importlib.metadata` plutôt que `anomalib.__version__` ?)
  Il nous faut d'autres informations pour être sûr que le score ne change pas :
  - `normalize: false` : les scores changent en fonction des valeurs utilisées dans la normalisation, si celles ci sont différentes, elle rend les scores incomparables.
  - la version d'anomalib : nous ne controlons pas les changements de code pour une librairie, si celui ci change, cela peut potentiellement affecter le score.
  Nous avons utilisé `importlib.metadata` plutot que `anomalib.__version__` car l'import de la librairie `anomalib` est inutilement lourd et la CI n'installe pas cette librairie.
- Le piège `path` :
  (Quelle variable écrasait laquelle ? Qu'aurait-on perdu ?)
  Dans `save_scores`, `path` était à l'origine le fichier JSON de sortie, mais la boucle réutilisait ce même nom pour le chemin de chaque image. À la fin de la boucle, `path` valait le chemin de la dernière image MVTec, et le JSON aurait écrasé cette image du dataset. Résolu en renommant la variable de la boucle `image_path`. `path` est un nom trop générique pour être réutilisé sans risque.

## Les coûts dans OperatingPoint
- L'idée :
  (Pourquoi mettre `defect_rate`, `cost_fn` et `cost_fp` dans le point ?)
  C'est moi qui ai posé la question avant de lancer l'étape 3 : ne fallait-il pas mettre defect_rate dans OperatingPoint ? Un point transporte maintenant les paramètres qui ont produit son `cost`, et `save_threshold` les lit dans le point au lieu de les recevoir à part. Un fichier qui mentirait devient impossible.
- Les tests d'abord :
  (Quels tests écrire ou modifier ?)
  Modifier `test_save_and_load_roundtrip`, pour vérifier que les nouvelles données sont bien sauvegardées et rechargeables. On a ajouté `test_point_carries_its_cost_parameters`.
- `save_threshold` :
  (Que devient sa signature ? `{**asdict(point), "cost_fn": ...}` sert-il encore ?)
  Non, `asdict` suffit, puisque tout est dans le point. La signature devient donc `save_threshold(point, path)` car comme dit précedemment les valeurs sont déjà dans point et les remettre pourrait mener à une erreur.

## evaluate.py
- Quel seuil pour la production :
  (20 découpages, 20 seuils. Lequel écrire dans models/threshold.json ?)
  Aucun des 20 : on peut prendre toutes les images. Les 20 découpages servent à mesurer honnêtement la performance. Une fois la mesure faite, plus besoin de garder la moitié des images de côté : le seuil final est choisi sur les 160.
- scikit-learn et la CI :
  (La CI n'installe ni anomalib ni scikit-learn. Que faire ?)
  Ajouter scikit-learn à l'extra `dev` me semblait moins bricolé que l'importer dans la fonction et sauter les tests. Il est léger, et les tests du découpage tournent en CI. anomalib, lui, reste importé dans les fonctions.
- Le découpage en fonctions :
  (Rôle de `load_or_collect_scores`, `split_results`, `evaluate_split`, `summarize`, `main`.)
  `load_or_collect_scores` vérifie que le calcul des scores concernant certains paramètres n'a pas été déjà fait pour éviter de relancer inutilement, il charge donc les scores si déjà produits sinon il lance le calcul, récupère puis sauvegarde les scores. 
  `split_results` permet de découper en deux moitiés stratifiées (validation et test), où chaque type de défaut garde la même proportion dans les deux moitiés.
  `evaluate_split` prend les scores pour les séparer en deux datasets avec `split_results`, puis il choisit en seuil en fonction de validation et ensuite évalue ce seuil sur validation et test, pour ainsi renvoyer deux `OperatingPoint` qui nous donnent les informations du seuil choisi (sur validation) pour le dataset de validation et de test.
  `summarize` récupère une liste de points avec leurs valeurs, et renvoie la moyenne, le minimum, le maximum et l'écart-type de ces points.
  `main` prend les arguments données (modèles, catégorie, seed, costs fn/fp, defect_rate, nombre de génération), et renvoie un résumé sur `n_splits` découpages (20 par défaut) avec `summarize` et les différentes fonctions si dessus, il choisit ensuite le seuil final parmi les 160 images et l'écrit dans `models/threshold.json`.
- Les tests :
  (Que prouve `test_logic_split_results` ? Pourquoi trois points construits à la main pour `summarize` ?)
  Avec `test_logic_split_results`, je voulais vérifier que la répartion de `split_results` serait bien cohérentes et qu'elle prendrait bien en compte le type de défauts, ainsi que m'assurer que les répartitions sont distincts et qu'elles contiennent ensemble les scores en entrée.
  Pour `summarize`, je voulais créer une liste de points et vérifier qu'on retrouve les valeurs exactes.
  Avec `test_save_and_load`, on veut s'assurer que les scores soient bien sauvegardés et peuvent bien être chargés.
- Le nom du rapport :
  (Que se passait-il quand deux runs avec des paramètres différents écrivaient dans le même fichier ?)
  Avant, un run avec d'autres coûts ou une autre graine écrasait le rapport précédent. Maintenant, tout ce qui change le résultat est dans le nom du fichier (modèle, catégorie, graine, `cost_fn`, `cost_fp`, `defect_rate`) : des paramètres différents donnent un rapport différent. Seul `n_splits`, qui joue sur la précision de la mesure plus que sur le résultat, est uniquement dans le rapport. Un run avec un autre `--n-splits` écrase donc encore le rapport, c'est un choix assumé.

## ddof=1
  (Écart-type de population ou d'échantillon ? Pourquoi le second ici ? De combien les chiffres ont-ils bougé par rapport à la séance 5 ?)
  Il y a l'écart-type de population ou d'échantillon, le premier est plus utilisé quand nous avons une population bien définie et fini alors que le second, nous travaillons sur part représentative de celle ci. C'est pour cela que l'on va plutot utilisé la 2e pour plusieurs raisons, nous travaillons sur un échantillon représentatif d'un problème, et nous travaillons également sur plusieurs générations de split mais pas toutes celles possibles.
  L'écart-type est multiplié par √(20/19) ≈ 1.026, soit +2.6 %. La raison précise : la moyenne est calculée sur les mêmes 20 valeurs, ce qui fait paraître l'écart à la moyenne trop petit. Diviser par n−1 corrige ce biais.

## Le seuil final
- Ma prédiction :
  (Près du min, de la moyenne ou du max des 20 seuils ?)
  Plus haut, car le max des bonnes pièces sera possiblement plus haut avec 41 bonnes pièces qu'avec 20.
- Le résultat :
  (32.20, contre une moyenne de 31.96 et un max de 33.14. Pourquoi ma prédiction n'est qu'à moitié juste ?)
  Avec `cost_fn = 100` et `defect_rate = 0.01`, un défaut raté a un coefficient de 1 (= 100 x 0.01) alors qu'un faux rejet a un coefficient de 0.99. Les deux erreurs coûtent presque autant, donc le seuil cherche un compromis dans la zone où bonnes et mauvaises pièces se chevauchent (il rejette encore 3 bonnes pièces), il n'est pas collé au max des bonnes pièces. C'est pour cela qu'il est proche de la moyenne des 20 seuils. Avoir plus d'images rend le seuil plus stable, pas forcément plus haut. Mon raisonnement aurait été juste avec `cost_fn = 10`, où les faux rejets dominent.
- Le piège de `threshold.json` :
  (Pourquoi ne pas citer son rappel de 0.857 et ses 7.3 % de faux rejets ? Quels chiffres citer à la place ?)
  Le seuil final est choisi et mesuré sur les mêmes 160 images : son rappel (0.857) et ses faux rejets (7.3 %) sont optimistes, comme la validation en séance 5. Les chiffres à citer sont ceux de reports/, mesurés sur des tests jamais vus : rappel 0.875 (0.797 à 0.983), faux rejets 14.1 % (0 à 28.6 %).

## La CI rouge
  (Qu'est-ce qui a cassé sur de3f4ac ? Comment l'éviter avant chaque commit ?)
  J'avais écrit `name +=f"..."` sans espace après `+=`. `ruff format --check` le refuse, ce qui a fait échouer la CI (corrigé dans e6cedbf). Pour éviter cela, soit faire `ruff format .` avant chaque commit, soit installer un hook pre-commit.

## Questions ouvertes
- L'export ONNX sort-il des scores bruts, dans les mêmes unités que le seuil ?
- `threshold.json` ne dit pas pour quel modèle ni quelle catégorie il a été calculé : faut-il l'ajouter ?
- Déploierais-tu PatchCore sur screw avec 12.5 % de défauts ratés et 14 % de faux rejets ?
- Faut-il ranger les *_results.json de la semaine 1 dans un sous-dossier de results/ ?





# Séance 7 : 16 et 17 novembre 2026, export ONNX

## Ce que contient le fichier ONNX
- Le graphe :
  (Que trace l'export ? Pourquoi exporter avec enable_normalization=False ? Que faire de pred_label ?)
  L'export trace ce que l'on appelle forward, càd le pre processing, le modèle puis le post processing. 
  La raison du `enable_normalization=False` est que les scores normalisés sont écrêtés dans [0, 1] : un gros défaut et un défaut moyen peuvent tous les deux valoir 1, également le seuil a été mis à 0.5, ce qui rend la notion de distance encore un peu plus compliqué. La raison principale est que le seuil est choisi sur un score brut, donc avec les scores entre 0 et 1, le modèle dirait que tout est sans anomalie.
  `pred_label` utilise le seuil d'anomalib (avec les scores normalisés), donc ca nous est pas utile. 
- La taille :
  (Ma prédiction, le calcul de la banque de mémoire, la taille réelle, d'où vient l'écart ?)
  327 680 vecteurs, 10% des vecteurs pour le coreset, 1536 float/vecteur, 4 o/float, 1 Mo = 1e6 o
  J'avais commencé par une erreur de compréhension et prédis un modèle de 0.13 Mo, ce qui n'est pas très réaliste, le problème venait de l'oublie qu'un vecteur n'est pas seulement un float de 4o mais 1536 float de 4o.
  Sans mon oublie, la prévision était de 201Mo et la taille réelle était de 301Mo, l'écart vient du modèle utilisé pour extraire les vecteurs d'une image (ResNet).

## Les pièges de l'export
- onnxscript :
  (Pourquoi l'extra train et pas serve ?)
  La partie export ne se fait que sur la machine, il utilise des librairies comme torch et onnxscript qui en est une dépendance. Alors que l'API dans Docker, ne fait que lire le fichier .onnx avec onnxruntime. Et serve est gardé léger exprès, sans torch, on laisse le plus lourd sur la machine avec train.
- Le fichier .data :
  (Pourquoi le move a cassé le modèle ? Pourquoi fusionner plutôt que déplacer deux fichiers ?)
  Pour pouvoir utiliser le modèle sous onnx, quand ce n'est pas fusionné, il faut les deux, sauf que le move ne déplacait que le graphe et pas le .data.
  Fusionner nous permet d'oublier la gestion des deux fichiers à la fois, et évite les oublis.
  Le nom du .data est écrit dans le graphe, donc renommer model.onnx casse le lien
- La taille de lot figée à 2 :
  (Pourquoi regarder les formes n'a pas suffi ? Pourquoi `dynamic_shapes=None` ?)
  Regarder les formes ne suffisait pas : le 2 était bien affiché, mais rien ne plante tant qu'on n'appelle pas le modèle. Seul un vrai appel avec une image (`Got: 1 Expected: 2`) a révélé le problème. L'API et le benchmark envoient une image à la fois, ils auraient planté. `dynamic_shapes=None` dit à l'exporteur qu'aucune dimension n'est dynamique : il trace avec un exemple `[1, 3, 256, 256]`, donc la taille de lot vaut 1 par construction, et non 2 par accident.
- L'entrée fixe en 256 :
  (Que se passe-t-il maintenant si on envoie une image de 1024 ? Pourquoi c'est mieux qu'une entrée dynamique ?)
  Mettre en autre chose que 256 causera une erreur.
  C'est mieux de mettre une entrée statique pour éviter des erreurs.
  Avant, une image de 1024 donnait un score faux de 8 points sans aucune erreur ; maintenant, elle plante tout de suite.

## Les scores : brut, mais identiques ?
- Ma prédiction :
  (Écart nul ?)
  Je ne connaissais pas les méthodes utilisées par anomalib donc je pensais qu'en utilisant les standards, l'écart serait probablement nul.
- Les quatre prétraitements :
  | Prétraitement | Écart max | Décisions changées |
  |---|---|---|
  | aucun (Resize du graphe) | 8.62 | 33 |
  | PIL uint8 | 2.20 | 3 |
  | torchvision float | 0.119 | 0 |
  | PIL "F" (retenu)| 0.119 | 0 |
- Pourquoi le Resize du graphe était faux :
  (Antialiasing : qu'est-ce que c'est, pourquoi ça compte sur une réduction de 1024 à 256 ?)
  En réduisant d'un facteur 4, chaque pixel de sortie représente 4 × 4 = 16 pixels d'origine : sans antialiasing, le bilinéaire n'en lit que 4 et ignore les 12 autres. Pour une réduction de 1.1, presque aucun pixel n'est ignoré, donc l'effet est faible. Les filets de la vis sont des motifs fins et répétés, exactement ce qui se transforme en faux motifs (crénelage). Résultat : 8.6 points d'écart et 33 décisions changées.
- Pourquoi PIL uint8 ne suffisait pas :
  (Arrondi, et pourquoi PatchCore y est sensible.)
  Le problème, c'est qu'il faut faire pareil qu'à l'entrainement pour obtenir les mêmes résultats, anomalib a redimensionné en float alors que PIL arrondit à l'entier.
  PatchCore y est sensible car son score est un maximum sur les patchs, il suffit qu'un seul patch bouge, les scores bougent nécessairement. PIL "F" utilise le même filtre, mais sans arrondi, et retrouve le résultat de torchvision. Donc la cause était l'arrondi.
- Les 0.12 restants :
  (Biais ou bruit ? D'où viennent-ils probablement ? Pourquoi c'est acceptable ?)
  Le 0.12 est du bruit, il provient probablement du traitement GPU car avec le pré traitement nous avions noté une différence d'environ 1e-5 ce qui le met hors de cause.
  Tant que les résultats ne différent pas, c'est acceptable, je parle ici des décisions/détections.
  Le problème serait qu'un score soit à 0.12 du seuil et avec le bruit, ca pourrait changer la décision.
- Le piège resize :
  (Pourquoi des chiffres identiques au millionième étaient un signal ?)
  C'était un signal, car ca impliquait qu'il n'y avait pas de modification, meme minime, car nous avons des scores qui sont plutôt précis donc ca se verait. Quand une modification ne change rien au chiffre près, vérifier d'abord qu'elle s'exécute vraiment.

## preprocess et IMAGE_SIZE
  (Pourquoi preprocess prend une image et pas un chemin ? Pourquoi dans inference.py ? Pourquoi 256 est une propriété du modèle et pas des images ?)
  La fonction `preprocess` prend plutot une image car l'API va nous envoyer une et pas un chemin, c'est donc plus logique. Séparer `load_image` et la partie de `preprocess` permet d'éviter d'avoir deux mêmes copies du code, de plus, ils ont deux objectifs différents, c'est donc aussi logique de les séparer. 
  Il est écrit dans `inference.py` car comme le dit son `preprocess` est une étape avant les traitements principaux, et va être principalement utilisé pour les images durant l'inférence, car dans train par exemple, on utilise anomalib et les datasets directement. Vu que la fonction aurait été la même que celle utilisé dans `inference.py`, on l'y a placé, aussi `inference.py` n'importe pas de lourdes librairies
  256 est une propriété du modèle entraîné : pendant le fit, toutes les images ont été réduites à 256, et la banque de mémoire contient des patchs vus à cette échelle. À une autre échelle, chaque patch couvre une autre zone de la vis, les caractéristiques changent, donc les distances et les scores aussi. La taille des images d'entrée (700, 900 ou 1024 selon la catégorie) n'a pas d'importance, puisqu'elles sont toutes ramenées à 256. Revers : un défaut de quelques pixels en 1024 peut disparaître en 256.

## Questions ouvertes
- threshold.json ne dit pas pour quel modèle ni quelle catégorie il a été calculé ;
- déploierais-tu PatchCore sur screw ?
- faut-il ranger les *_results.json de la semaine 1 ?
- AnomalyModel devrait-il lire la taille d'entrée directement dans le .onnx ?
- les 0.12 viennent-ils vraiment du TF32 ? (vérifiable en désactivant TF32 et en recalculant le cache.)
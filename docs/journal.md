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
  On choisit d'importer les standarts et les imports léger en haut du script car ils sont léger à importer alors que les autres plus lourd, on préfère les importer si c'est vraiment nécessaire. L'autre raison est que certain import ne s'installe pas avec le CI, ce qui créerait des erreurs et ne lancerait rien.
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
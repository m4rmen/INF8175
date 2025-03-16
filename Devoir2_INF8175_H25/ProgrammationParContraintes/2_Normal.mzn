% Alexandre Marmen (2045362)
% Adam Kassi Lahlou (2195278)


include "globals.mzn"; 

%-----------------------------------------------------------------------------%
% Données
%-----------------------------------------------------------------------------%

% Nombre de tâches
int: n;
% Durée maximal du projet
int: m;
% Limite de ressources pouvant être partagées entre les tâches en cours
int: limite;

% Durée des tâches
array[1..n] of 1..m: duree_tache;
% Ressources requises par les tâches
array[1..n] of 1..m: ressource_tache;

%-----------------------------------------------------------------------------%
% Variables de décisions
%-----------------------------------------------------------------------------%

array[1..n] of var 1..m: debut_tache; % Début des tâches
array[1..n] of var 1..m: fin_tache; % Fin des tâches

%-----------------------------------------------------------------------------%
% Objectif
%-----------------------------------------------------------------------------%

% Fin du projet
var int: fin_projet;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%

constraint forall(i in 1..n)(
   fin_tache[i] = debut_tache[i] + duree_tache[i]
);

constraint fin_projet = max(i in 1..n)(fin_tache[i]);

constraint cumulative(debut_tache, duree_tache, ressource_tache, limite);

array[1..m] of var 0..n: occ;
constraint global_cardinality(debut_tache, [t | t in 1..m], occ);
constraint forall(t in 1..m)( occ[t] <= 2 );

solve minimize fin_projet;

%-----------------------------------------------------------------------------%

output [
  "Envergure du projet  : ", show(fin_projet), "\n",
  "Début des tâches     : ", show(debut_tache), "\n",
  "Fin des tâches       : ", show(fin_tache), "\n",
];
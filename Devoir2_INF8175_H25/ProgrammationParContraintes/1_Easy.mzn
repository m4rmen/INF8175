% Alexandre Marmen (2045362)
% Adam Kassi Lahlou (2195278)

include "globals.mzn";

%-----------------------------------------------------------------------------%
% Données
%-----------------------------------------------------------------------------%

% Nombre d'acteurs et de rôles
int: n;
% Couleurs des costumes
enum costumes; 
% Costume démandé pour un rôle i (première dimension) par un acteur j (seconde dimension)
array[1..n,1..n] of costumes: choix_costume;

%-----------------------------------------------------------------------------%
% Variables de décisions
%-----------------------------------------------------------------------------%

% Variables de décision pour assigner un acteur à un rôle
array [1..n] of var 1..n: assignation_role;

array[1..n] of var costumes: chosen_costume;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%
constraint alldifferent(assignation_role);

constraint
  forall(i in 1..n) (
    chosen_costume[i] = choix_costume[assignation_role[i], i]
  );
  
constraint alldifferent(chosen_costume);

constraint
  forall(i in 1..n-1) (
    abs(assignation_role[i] - assignation_role[i+1]) != 1
  );
solve satisfy;

%-----------------------------------------------------------------------------%

output	
  ["L'acteur " ++ format(i) ++ 
   " prendra le rôle " ++ format(fix(assignation_role[i])) ++ 
   " (costume " ++ format(choix_costume[fix(assignation_role[i]), i]) ++ ")" ++"\n" 
  | i in 1..n];
from schedule import Schedule
import random
import math
import copy

def greedy_coloring(schedule):
    courses = list(schedule.course_list)
    courses.sort(key=lambda c: schedule.conflict_graph.degree(c), reverse=True)
    solution = {}
    for c in courses:
        neighbor_colors = {solution[n] for n in schedule.get_node_conflicts(c) if n in solution}
        color = 1
        while color in neighbor_colors:
            color += 1
        solution[c] = color
    return solution

def solve(schedule):
    T0 = 1500.0         
    T_min = 0.5e-3        
    alpha =  0.997       
    M = 1000            

    n_restarts = 15   

    courses = list(schedule.course_list)
    conflicts = schedule.conflict_list  

    def compute_cost(solution):
        conflict_count = 0
        for (a, b) in conflicts:
            if solution[a] == solution[b]:
                conflict_count += 1
        num_slots = len(set(solution.values()))
        return conflict_count * M + num_slots

    def is_conflict_free(sol):
        for (a, b) in conflicts:
            if sol[a] == sol[b]:
                return False
        return True

    best_overall_solution = None
    best_overall_cost = float('inf')

    for restart in range(n_restarts):
        current_solution = greedy_coloring(schedule)
        current_cost = compute_cost(current_solution)
        T = T0

        while T > T_min:
            new_solution = copy.deepcopy(current_solution)
            if random.random() < 0.5:
                course = random.choice(courses)
                max_slot = max(new_solution.values())
                new_slot = random.randint(1, max_slot)
                new_solution[course] = new_slot
            else:
                course1, course2 = random.sample(courses, 2)
                new_solution[course1], new_solution[course2] = new_solution[course2], new_solution[course1]

            new_cost = compute_cost(new_solution)
            delta = new_cost - current_cost

            if delta < 0 or random.random() < math.exp(-delta / T):
                current_solution = new_solution
                current_cost = new_cost

            T *= alpha

        if is_conflict_free(current_solution):
            used_slots = sorted(set(current_solution.values()))
            slot_map = {old: new for new, old in enumerate(used_slots, start=1)}
            for c in current_solution:
                current_solution[c] = slot_map[current_solution[c]]

        if current_cost < best_overall_cost and is_conflict_free(current_solution):
            best_overall_solution = copy.deepcopy(current_solution)
            best_overall_cost = current_cost

    if best_overall_solution is None:
        best_overall_solution = greedy_coloring(schedule)

    return best_overall_solution

import numpy as np
import random
from collections import defaultdict

# Определение параметров задачи
DAYS = 5
CLASSES_PER_DAY = 4
COURSE_NUM = 7
TEACHER_NUM = 7
ROOM_NUM = 4
GROUP_NUM = 4

# Курсы и преподаватели
courses = [
   ("Data Structure and Algorithms", "Askarov K.R"),
   ("English", "Абакирова Э.А"),
   ("Introduction to AI", "Beishenalieva A."),
   ("Advanced Python", "Prof. Daechul Park"),
   ("География Кыргызстана", "Жумалиев Н.Э."),
   ("История Кыргызстана", "Молошев А.И."),
   ("Манасоведение", "Бегалиев Э.С."),
]

# Ограничения по количеству занятий
course_limits = {
   "Data Structure and Algorithms": (1, 2),
   "English": (3, 5),
   "Introduction to AI": (1, 2),
   "Advanced Python": (1, 2),
   "География Кыргызстана": (1, 2),
   "История Кыргызстана": (1, 2),
   "Манасоведение": (1, 2)
}

# PSO параметры
w = 0.5  # Инерция
c1 = 1.5  # Личное обучение
c2 = 1.5  # Социальное обучение
num_particles = 50  # Количество частиц
iterations = 200  # Количество итераций

# Функция приспособленности (fitness function)
def fitness(schedule):
   penalty = 0
   course_counts = defaultdict(lambda: [0] * GROUP_NUM)
   for day in range(DAYS):
       for cls in range(CLASSES_PER_DAY):
           used_teachers = set()
           used_rooms = set()
           for group in range(GROUP_NUM):
               course, teacher, room = schedule[day][cls][group]
               course_counts[course][group] += 1
               if teacher in used_teachers:
                   penalty += 5  # Штраф за конфликт преподавателя
               if room in used_rooms:
                   penalty += 3  # Штраф за конфликт аудитории
               used_teachers.add(teacher)
               used_rooms.add(room)
   # Проверка выполнения ограничений по курсам
   for course, counts in course_counts.items():
       min_req, max_req = course_limits.get(course, (0, float('inf')))
       for group_count in counts:
           if group_count < min_req:
               penalty += (min_req - group_count) * 3
           elif group_count > max_req:
               penalty += (group_count - max_req) * 3
   return 1 / (1 + penalty)

# Генерация начальных частиц (расписаний)
def generate_random_schedule():
   schedule = [[[] for _ in range(CLASSES_PER_DAY)] for _ in range(DAYS)]
   for day in range(DAYS):
       for cls in range(CLASSES_PER_DAY):
           used_teachers = set()
           for group in range(GROUP_NUM):
               available_courses = [c for c in courses if c[1] not in used_teachers]
               if not available_courses:
                   available_courses = courses
               course, teacher = random.choice(available_courses)
               room = random.randint(1, ROOM_NUM)
               schedule[day][cls].append((course, teacher, room))
               used_teachers.add(teacher)
   return schedule

# PSO Алгоритм
particles = [generate_random_schedule() for _ in range(num_particles)]
velocities = [np.zeros((DAYS, CLASSES_PER_DAY, GROUP_NUM)) for _ in range(num_particles)]
pbest = particles[:]
pbest_scores = [fitness(p) for p in pbest]
gbest = pbest[np.argmax(pbest_scores)]

def update_velocity(velocity, particle, pbest, gbest):
   for day in range(DAYS):
       for cls in range(CLASSES_PER_DAY):
           for group in range(GROUP_NUM):
               r1, r2 = random.random(), random.random()
               inertia = w * velocity[day][cls][group]
               cognitive = c1 * r1 * (pbest[day][cls][group] != particle[day][cls][group])
               social = c2 * r2 * (gbest[day][cls][group] != particle[day][cls][group])
               velocity[day][cls][group] = inertia + cognitive + social
   return velocity

# Основной цикл оптимизации
for _ in range(iterations):
   for i in range(num_particles):
       velocities[i] = update_velocity(velocities[i], particles[i], pbest[i], gbest)
       new_schedule = generate_random_schedule()
       if fitness(new_schedule) > pbest_scores[i]:
           pbest[i] = new_schedule
           pbest_scores[i] = fitness(new_schedule)
       if fitness(new_schedule) > fitness(gbest):
           gbest = new_schedule

# Вывод оптимизированного расписания
for day in range(DAYS):
   print(f"Day {day + 1}:")
   for cls in range(CLASSES_PER_DAY):
       print(f"  Class {cls + 1}:")
       used_rooms = set()
       for group in range(GROUP_NUM):
           course, teacher, room = gbest[day][cls][group]
           while room in used_rooms:
               room = random.randint(1, ROOM_NUM)
           used_rooms.add(room)
           print(f"    Group {group + 1}: {course} by {teacher} in Room {room}")
   print("\n")

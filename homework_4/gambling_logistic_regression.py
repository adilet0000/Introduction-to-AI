import numpy as np
import pandas as pd
import random
from itertools import combinations
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ------------------------------
# 1. Настройка карт и функций
# ------------------------------

# Масти и ранги (для парсинга)
suits = ['♠', '♥', '♦', '♣']
suits_map = {'♠': 0, '♥': 1, '♦': 2, '♣': 3}
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
rank_to_value = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
}

# Формируем колоду (52 карты)
deck = [rank + suit for suit in suits for rank in ranks]

def parse_card(card_str):
    """
    Преобразует строку вида '10♣' или 'A♥' в (rank_value, suit_value).
    """
    suit_symbol = card_str[-1]          # Последний символ — масть
    rank_str = card_str[:-1]           # Всё, кроме последнего символа — ранг (учтёт '10')
    rank_val = rank_to_value[rank_str]
    suit_val = suits_map[suit_symbol]
    return (rank_val, suit_val)

def is_straight(ranks_sorted_desc):
    """
    Проверяем, является ли 5 отсортированных по убыванию рангов подряд идущими.
    Учитываем особый случай (Ace low): [14, 5, 4, 3, 2].
    """
    # Сначала проверяем обычную последовательность
    for i in range(4):
        if ranks_sorted_desc[i] != ranks_sorted_desc[i+1] + 1:
            break
    else:
        # Если не прервало цикл, значит все подряд
        return True
    
    # Проверка Ace low (5,4,3,2,A)
    # Если множество рангов = {14,5,4,3,2}, то это "колесо" (5-high straight)
    if set(ranks_sorted_desc) == {14, 5, 4, 3, 2}:
        return True
    
    return False

def evaluate_5_cards(cards_5):
    """
    Оцениваем 5-карточную руку (cards_5 — список из 5 кортежей (rank, suit)).
    Возвращаем категорию (0…8):
      8 = Straight Flush
      7 = Four of a Kind
      6 = Full House
      5 = Flush
      4 = Straight
      3 = Three of a Kind
      2 = Two Pair
      1 = One Pair
      0 = High Card
    (Без учёта тайбрейков.)
    """
    ranks_ = sorted([c[0] for c in cards_5], reverse=True)
    suits_ = [c[1] for c in cards_5]
    
    # Flush?
    flush = (len(set(suits_)) == 1)
    
    # Straight?
    straight = is_straight(ranks_)
    
    # Подсчёт кратности рангов (для определения пар, тройки, каре и т.д.)
    rank_counts = Counter(ranks_)
    freq = sorted(rank_counts.values(), reverse=True)  # напр. [4,1], [3,2], [3,1,1], [2,2,1], [2,1,1,1], [1,1,1,1,1]
    
    # Логика определения категории
    if straight and flush:
        return 8  # Straight Flush
    elif 4 in freq:
        return 7  # Four of a Kind
    elif 3 in freq and 2 in freq:
        return 6  # Full House
    elif flush:
        return 5  # Flush
    elif straight:
        return 4  # Straight
    elif 3 in freq:
        return 3  # Three of a Kind
    elif freq.count(2) == 2:
        return 2  # Two Pair
    elif 2 in freq:
        return 1  # One Pair
    else:
        return 0  # High Card

def best_5_from_7(cards_7):
    """
    Возвращаем лучший (максимальный) "рейтинг" руки из 7 карт,
    перебирая все комбинации по 5.
    """
    best_rank = 0
    for combo in combinations(cards_7, 5):
        rank_5 = evaluate_5_cards(combo)
        if rank_5 > best_rank:
            best_rank = rank_5
    return best_rank

# ------------------------------
# 2. Генерация синтетических раздач
# ------------------------------

def simulate_game():
    """
    Симулируем одну раздачу:
      - Перемешиваем колоду и раздаем 2 карты игроку + 5 общих карт.
      - Генерируем случайный размер ставки (10…100).
      - Считаем лучшую 5-карточную комбинацию из 7 карт (hand_rank).
      - На основе hand_rank и bet (с шумом) генерируем вероятность выигрыша.
      - Случайно определяем win=1 или 0.
    """
    cards_copy = deck.copy()
    random.shuffle(cards_copy)
    
    # 2 карманные + 5 общих
    hand = cards_copy[:2]
    board = cards_copy[2:7]
    
    # Парсим в (rank, suit) и ищем лучший рейтинг 5 из 7
    all_7_cards = [parse_card(c) for c in (hand + board)]
    hand_rank = best_5_from_7(all_7_cards)
    
    # Случайный размер ставки
    bet = random.uniform(10, 100)
    
    # "Сила" руки в терминах нашей логистической функции
    # Допустим, средний рейтинг ~ 3-4, возьмём как центр
    avg_rank = 3.5
    avg_bet = 55
    bias = 0.0
    alpha = 0.8      # влияние рейтинга руки
    beta_coeff = 0.01  # влияние ставки
    noise = np.random.normal(0, 0.5)
    
    # Линейная комбинация признаков + шум
    score = bias \
            + alpha * (hand_rank - avg_rank) \
            + beta_coeff * (bet - avg_bet) \
            + noise
    
    # Преобразуем score в вероятность (сигмоида)
    prob = 1 / (1 + np.exp(-score))
    
    # Генерируем исход (1 = выигрыш, 0 = проигрыш)
    win = 1 if random.random() < prob else 0
    
    return {
        'hand': hand,
        'board': board,
        'hand_rank': hand_rank,
        'bet': bet,
        'win': win
    }

# Генерируем датасет
n_games = 1000
data = [simulate_game() for _ in range(n_games)]
df = pd.DataFrame(data)

# ------------------------------
# 3. Обучение модели
# ------------------------------

X = df[['hand_rank', 'bet']]
y = df['win']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ------------------------------
# 4. Оценка модели и вывод
# ------------------------------

accuracy = accuracy_score(y_test, y_pred)
print(f"Оценка точности (Accuracy): {accuracy:.2%}")
print(f"\nОтчёт о классификации:\n{classification_report(y_test, y_pred)}")

print(f"Коэффициенты модели: {model.coef_}")
print(f"Свободный член (Intercept): {model.intercept_}")

# ------------------------------
# 5. Визуализация
# ------------------------------

# (A) Зависимость шанса на выигрыш от hand_rank
df_grouped_rank = df.groupby('hand_rank', as_index=False)['win'].mean()

plt.figure(figsize=(8, 5))
plt.plot(df_grouped_rank['hand_rank'], df_grouped_rank['win'], marker='o',
         label='Средняя вероятность (реальные данные)')

# Предсказания при фиксированной средней ставке
hand_rank_range = range(0, 9)  # 0…8
avg_bet = df['bet'].mean()
X_line_rank = pd.DataFrame({
    'hand_rank': hand_rank_range,
    'bet': [avg_bet]*len(hand_rank_range)
})
pred_probs_rank = model.predict_proba(X_line_rank)[:, 1]

plt.plot(hand_rank_range, pred_probs_rank, marker='x', color='red',
         label='Предсказанная вероятность (модель)')

plt.title('Зависимость вероятности выигрыша от рейтинга руки (0–8)')
plt.xlabel('Рейтинг руки (hand_rank)')
plt.ylabel('Вероятность выигрыша')
plt.legend()
plt.grid(True)
plt.show()

# (B) Зависимость шанса на выигрыш от ставки (bet)
df['bet_bin'] = pd.cut(df['bet'], bins=10, include_lowest=True)
bet_grouped = df.groupby('bet_bin', as_index=False)['win'].mean()

plt.figure(figsize=(8, 5))
plt.bar(bet_grouped['bet_bin'].astype(str), bet_grouped['win'], alpha=0.7,
        label='Средняя вероятность (реальные данные)')

# Линия предсказаний (варьируем bet, фиксируем средний рейтинг)
bet_range = np.linspace(df['bet'].min(), df['bet'].max(), 10)
avg_rank_val = df['hand_rank'].mean()

X_line_bet = pd.DataFrame({
    'hand_rank': [avg_rank_val]*len(bet_range),
    'bet': bet_range
})
pred_probs_bet = model.predict_proba(X_line_bet)[:, 1]

plt.plot(range(len(bet_range)), pred_probs_bet, color='red', marker='o',
         label='Предсказанная вероятность (модель)')

plt.xticks(range(len(bet_grouped)), bet_grouped['bet_bin'].astype(str), rotation=45)
plt.title('Зависимость вероятности выигрыша от размера ставки')
plt.xlabel('Интервалы ставки')
plt.ylabel('Вероятность выигрыша')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------
# 6. Итоговые комментарии
# ------------------------------
print(f"\n--- Итоговый анализ ---")
print(f"1) Теперь мы действительно учитываем покерные комбинации (лучшая 5-карточная из 7).")
print(f"2) Рейтинг руки (0–8) соответствует классическим категориям: High Card, One Pair, ... Straight Flush.")
print(f"3) Логистическая регрессия улавливает рост вероятности выигрыша при увеличении hand_rank.")
print(f"4) Точность модели: {accuracy:.2%} (учитывайте, что данные мы сгенерировали случайно).")
print(f"5) В реальном покере важны тайбрейки (старшинство в одной категории) и дополнительные факторы.")

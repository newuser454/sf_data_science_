import game_v2

def game_core_v3(number: int = 1) -> int:

    # фиксируем пороговые значения
    lower_bound = 1
    upper_bound = 100
    attempts = 0

    # используем алгоритм бинарного поиска для нахождения числа
    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        attempts += 1

        if mid == number:
            return attempts
        elif mid < number:
            lower_bound = mid + 1
        else:
            upper_bound = mid - 1

game_v2.score_game(game_core_v3)
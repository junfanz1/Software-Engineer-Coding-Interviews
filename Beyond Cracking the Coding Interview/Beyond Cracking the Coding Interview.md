Beyond Cracking the Coding Interview - by Gayle Laakmann McDowell, Mike Mroczka, Aline Lerner, Nil Mamano, 2025

> [Book Link](https://www.amazon.com/Beyond-Cracking-Coding-Interview-Successfully/dp/195570600X)

 <img src="https://github.com/user-attachments/assets/62a0e128-7ca0-49b4-a53f-fed51456f06b" width="35%" height="35%">

# Contents

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [20. Anatomy of Coding Interview](#20-anatomy-of-coding-interview)
- [25. Dynamic Arrays](#25-dynamic-arrays)
- [26. String Manipulation](#26-string-manipulation)
- [27. Two Pointers](#27-two-pointers)
- [28. Grids & Matrices](#28-grids-matrices)
- [29. Binary Search](#29-binary-search)
- [30. Set & Maps](#30-set-maps)
- [31. Sorting](#31-sorting)
- [32. Stacks & Queues](#32-stacks-queues)
- [33. Recursion](#33-recursion)
- [34. Linked Lists](#34-linked-lists)
- [35. Trees ](#35-trees)
- [36. Graphs](#36-graphs)
- [37. Heaps ](#37-heaps)
- [38. Sliding Windows ](#38-sliding-windows)
- [39. Backtracking](#39-backtracking)
- [40. Dynamic Programming](#40-dynamic-programming)
- [41. Greedy Algorithms](#41-greedy-algorithms)
- [42. Topological Sort ](#42-topological-sort)
- [43. Prefix Sums ](#43-prefix-sums)

<!-- TOC end -->

<!-- TOC --><a name="20-anatomy-of-coding-interview"></a>
## 20. Anatomy of Coding Interview

Get buy-in with magic question: I'd like to use A algorithm with B time and C space to solve problem. Should I code this now, or should I keep thinking?

<!-- TOC --><a name="25-dynamic-arrays"></a>
## 25. Dynamic Arrays

```py
class DynamicArray:
    def __init__(self):
        self.capacity = 10 
        self._size = 0
        self.fixed_array = [None] * self.capacity

    def get(self, i):
        if i < 0 or i >= self._size:
            raise IndexError('index out of bounds')
        return self.fixed_array[i]

    def set(self, i, x):
        if i < 0 or i >= self._size:
            raise IndexError('index out of bounds')
        self.fixed_array[i] = x

    def size(self):
        return self._size

    def append(self, x):
        if self._size == self.capacity:
            self.resize(self.capacity * 2)
        self.fixed_array[self._size] = x 
        self._size += 1

    def resize(self, new_capacity):
        new_fixed_size_arr = [None] * new_capacity
        for i in range(self._size):
            new_fixed_size_arr[i] = self.fixed_array[i]
        self.fixed_array = new_fixed_size_arr
        self.capacity = new_capacity

    def pop_back(self):
        if self._size == 0:
            raise IndexError('pop from empty array')
        self._size -= 1 
        if self._size / self.capacity < 0.25 and self.capacity > 10:
            self.resize(self.capacity // 2)

    def pop(self, i):
        if i < 0 or i >= self._size:
            raise IndexError('Index out of bounds')
        saved_element = self.fixed_array[i]
        for index in range(i, self._size - 1):
            self.fixed_array[index] = self.fixed_array[index + 1]
        self.pop_back()
        return saved_element
```
<!-- TOC --><a name="26-string-manipulation"></a>
## 26. String Manipulation

```py
def is_uppercase(c):
    return ord(c) >= ord('A') and ord(c) <= ord('Z')

def is_digit(c):
    return ord(c) >= ord('0') and ord(c) <= ord('9')

def is_alphanumeric(c):
    return is_lowercase(c) or is_uppercase(c) or is_digit(c)

def to_uppercase(c):
    if not is_lowercase(c):
        return c 
    return chr(ord(c) - ord('a') + ord('A'))

def split(s, c):
    if not s:
        return []
    res = []
    current = []
    for char in s:
        if char == c:
            res.append(''.join(current))
            current = []
        else:
            current.append(char)
    res.append(''.join(current))
    return res 

def join(arr, s):
    res = []
    for i in range(len(arr)):
        if i != 0:
            for c in s:
                res.append(c)
        for c in arr[i]:
            res.append(c)
    return array_to_string(res)
```

<!-- TOC --><a name="27-two-pointers"></a>
## 27. Two Pointers

```py
def palindrome(s):
    l, r = 0, len(s) - 1 
    while l < r:
        if s[l] != s[r]:
            return False 
        l += 1 
        r -= 1
    return True 

def smaller_prefixes(arr):
    sp, fp = 0, 0
    slow_sum, fast_sum = 0, 0 
    while fp < len(arr):
        slow_sum += arr[sp]
        fast_sum += arr[fp] + arr[fp+1]
        if slow_sum >= fast_sum:
            return False 
        sp += 1 
        fp += 2 
    return True 

def common_elements(arr1, arr2):
    p1, p2 = 0, 0 
    res = []
    while p1 < len(arr1) and p2 < len(arr2):
        if arr1[p1] == arr2[p2]:
            res.append(arr1[p1])
            p1 += 1 
            p2 += 1 
        elif arr1[p1] < arr2[p2]:
            p1 += 1 
        else:
            p2 += 1 
    return res 

def palindrome_sentence(s):
    l, r = 0, len(s) - 1 
    while l < r: 
        if not s[l].isalpha():
            l += 1 
        elif not s[r].isalpha():
            r -= 1 
        else:
            if s[l].lower() != s[r].lower():
                return False 
            l += 1 
            r -= 1 
    return True 

def reverse_case_match(s):
    l, r = 0, len(s) - 1 
    while l < len(s) and r >= 0:
        if not s[l].islower():
            l += 1 
        elif not s[r].isupper():
            r -= 1 
        else:
            if s[l] != s[r].lower():
                return False 
            l += 1 
            r -= 1 
    return True

def merge(arr1, arr2):
    p1, p2 = 0, 0
    res = []
    while p1 < len(arr1) and p2 < len(arr2):
        if arr1[p1] < arr2[p2]:
            res.append(arr1[p1])
            p1 += 1 
        else:
            res.append(arr2[p2])
            p2 += 1 
    while p1 < len(arr1):
        res.append(arr1[p1])
        p1 += 1 
    while p2 < len(arr2):
        res.append(arr2[p2])
        p2 += 1 
    return res 

def two_sum(arr):
    l, r = 0, len(arr) - 1 
    while l < r:
        if arr[l] + arr[r] > 0:
            r -= 1 
        elif arr[l] + arr[r] < 0:
            l += 1 
        else:
            return True 
    return False 

def sort_valley_array(arr):
    if len(arr) == 0:
        return []
    l, r = 0, len(arr) - 1 
    res = [0] * len(arr)
    i = len(arr) - 1 
    while l < r:
        if arr[l] >= arr[r]:
            res[i] = arr[l]
            l += 1 
            i -= 1 
        else:
            res[i] = arr[r]
            r -= 1 
            i -= 1 
    res[0] = arr[l]
    return res 

def intersection(int1, int2):
    overlap_start = max(int1[0], int2[0])
    overlap_end = min(int1[1], int2[1])
    return [overlap_start, overlap_end]
def interval_intersection(arr1, arr2):
    p1, p2 = 0, 0
    n1, n2 = len(arr1), len(arr2)
    res = []
    while p1 < n1 and p2 < n2:
        int1, int2 = arr1[p1], arr2[p2]
        if int1[1] < int2[0]:
            p1 += 1 
        elif int2[1] < int1[0]:
            p2 += 1 
        else:
            res.append(intersection(int1, int2))
            if int1[1] < int2[1]:
                p1 += 1 
            else:
                p2 += 1 
    return res 

def reverse(arr):
    l, r = 0, len(arr) - 1 
    while l < r:
        arr[l], arr[r] = arr[r], arr[l]
        l += 1
        r -= 1 

def sort_even(arr):
    l, r = 0, len(arr) - 1 
    while l < r:
        if arr[l] % 2 == 0:
            l += 1 
        elif arr[r] % 2 == 1:
            r -= 1 
        else:
            arr[l], arr[r] = arr[r], arr[l]
            l += 1 
            r -= 1

def remove_duplicates(arr):
    s, w = 0, 0
    while s < len(arr):
        must_keep = s == 0 or arr[s] != arr[s-1]
        if must_keep:
            arr[w] = arr[s]
            w += 1 
        s += 1 
    return w 

def move_word(arr, word):
    seeker, writer = 0, 0
    i = 0
    while seeker < len(arr):
        if i < len(word) and arr[seeker] == word[i]:
            seeker += 1 
            i += 1 
        else:
            arr[writer] = arr[seeker]
            seeker += 1 
            writer += 1 
    for c in word:
        arr[writer] = c 
        writer += 1 
```

<!-- TOC --><a name="28-grids-matrices"></a>
## 28. Grids & Matrices

```py
def is_valid(room, r, c):
    return 0 <= r < len(room) and 0 <= c < len(room[0]) and room[r][c] != 1 
def valid_moves(room, r, c):
    moves = []
    directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    for dir_r, dir_c in directions:
        new_r = r + dir_r 
        new_c = c + dir_c 
        if is_valid(room, new_r, new_c):
            moves.append([new_r, new_c])
    return moves 

def queen_valid_moves(board, piece, r, c):
    moves = []
    king_directions = [
        [-1, 0], [1, 0], [0, -1], [0, 1] # vertical and horizontal
        [-1, -1], [-1, 1], [1, -1], [1, 1] # diagonal
    ]
    knight_directions = [[-2, 1], [-1, 2], [1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1]]
    if piece == "knight":
        directions = knight_directions
    else: # king and queen
        directions = king_directions
    for dir_r, dir_c in directions:
        new_r, new_c = r + dir_r, c + dir_c 
        if piece == "queen":
            while is_valid(board, new_r, new_c):
                new_r += dir_r 
                new_c += dir_c 
        elif is_valid(board, new_r, new_c):
            moves.append([new_r, new_c])
    return moves 

def safe_cells(board):
    n = len(board)
    res = [[0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            if board[r][c] == 1:
                res[r][c] = 1 
                mark_reachable_cells(board, r, c, res)
    return res 
def mark_reachable_cells(board, r, c, res):
    directions = [
        [-1, 0], [1, 0], [0, -1], [0, 1] # vertical and horizontal
        [-1, -1], [-1, 1], [1, -1], [1, 1] # diagonal
    ]
    for dir_r, dir_c in directions:
        new_r, new_c = r + dir_r, c + dir_c 
        while is_valid(board, new_r, new_c):
            res[new_r][new_c] = 1 
            new_r += dir_r 
            new_c += dir_c 

def is_valid(grid, r, c):
    return 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == 0 
def spiral(n):
    val = n * n - 1 
    res = [[0] * n for _ in range(n)]
    r, c = n - 1, n - 1 
    directions = [[-1, 0], [0, -1], [1, 0], [0, 1]] # counterclockwise
    dir = 0 # start going up 
    while val > 0:
        res[r][c] = val 
        val -= 1 
        if not is_valid(res, r + directions[dir][0], c + directions[dir][1]):
            dir = (dir + 1) % 4 # change directions counterclockwise 
        r, c = r + directions[dir][0], c + directions[dir][1]
    return res 

def distance_to_river(field):
    R, C = len(field), len(field[0])
    def has_footprints(r, c):
        return 0 <= r < R and 0 <= c < C and field[r][c] == 1
    r, c = 0, 0 
    while field[r][c] != 1:
        r += 1 
    closest = r 
    directions_row = [-1, 0, 1]
    while c < C:
        for dir_r in directions_row:
            new_r, new_c = r + dir_r, c + 1 
            if has_footprints(new_r, new_c):
                r, c = new_r, new_c 
                closest = min(closest, r)
                break 
    return closest 

def valid_rows(board):
    R, C = len(board), len(board[0])
    for r in range(R):
        seen = set()
        for c in range(C):
            if board[r][c] in seen:
                return False 
            if board[r][c] != 0:
                seen.add(board[r][c])
    return True 

def valid_subgrid(board, r, c):
    seen = set()
    for new_r in range(r, r + 3):
        for new_c in range(c, c + 3):
            if board[new_r][new_c] in seen:
                return False 
            if board[new_r][new_c] != 0:
                seen.add(board[new_r][new_c])
    return True 
def valid_subgrids(board):
    for r in range(3):
        for c in range(3):
            if not valid_subgrid(board, r * 3, c * 3):
                return False
    return True 

def subgrid_maximums(grid):
    R, C = len(grid), len(grid[0])
    res = [row.copy() for row in grid]
    for r in range(R - 1, -1, -1):
        for c in range(C - 1, -1, -1):
            if r + 1 < R:
                res[r][c] = max(res[r][c], grid[r + 1][c])
            if c + 1 < C:
                res[r][c] = max(res[r][c], grid[r][c + 1])
    return res 

def backward_sum(grid):
    R, C = len(grid), len(grid[0])
    res = [row.copy() for row in grid]
    for r in range(R - 1, -1, -1):
        for c in range(C - 1, -1, -1):
            if r + 1 < R:
                res[r][c] += res[r + 1][c]
            if c + 1 < C:
                res[r][c] += res[r][c + 1]
            if r + 1 < R and c + 1 < C: # subtract double-counted subgrid
                res[r][c] -= res[r + 1][c + 1]
    return res 

class Matrix:
    def __init__(self, grid):
        self.matrix = [row.copy() for row in grid]
    def transpose(self):
        matrix = self.matrix 
        for r in range(len(matrix)):
            for c in range(r):
                matrix[r][c], matrix[c][r] = matrix[c][r], matrix[r][c]
    def reflect_horizontally(self):
        for row in self.matrix:
            row.reverse()
    def reflect_vertically(self):
        self.matrix.reverse()
    def rotate_clockwise(self):
        self.transpose()
        self.reflect_horizontally()
    def rotate_counterclockwise(self):
        self.transpose()
        self.reflect_vertically()
```

<!-- TOC --><a name="29-binary-search"></a>
## 29. Binary Search

```py
def binary_search(arr, target):
    n = len(arr)
    if n == 0:
        return -1 
    l, r = 0, n - 1 
    if arr[l] >= target or arr[r] < target:
        if arr[l] == target:
            return 0
        return -1 
    while r - l > 1:
        mid = (l + r) // 2 
        if arr[mid] < target:
            l = mid 
        else:
            r = mid 
    if arr[r] == target:
        return r 
    return -1 

def is_before(val):
    return not is_stolen(val)
def find_bike(t1, t2):
    l, r = t1, t2 
    while r - l > 1:
        mid = (l + r) // 2 
        if is_before(mid):
            l = mid 
        else:
            r = mid 
    return r 

def valley_min_index(arr):
    def is_before(i):
        return i == 0 or arr[i] < arr[i - 1]
    l, r = 0, len(arr) - 1 
    if is_before(r):
        return arr[r]
    while r - l > 1:
        mid = (l + r) // 2 
        if is_before(mid):
            l = mid 
        else:
            r = mid 
    return mid(arr[l], arr[r])

def two_array_two_sum(sorted_arr, unsorted_arr):
    for i, val in enumerate(unsorted_arr):
        idx = binary_search(sorted_arr, -val)
        if idx != -1:
            return [idx, i]
    return [-1, -1]

def is_before(grid, i, target):
    num_cols = len(grid[0])
    row, col = i // num_cols, i % num_cols 
    return grid[row][col] < target 
    
def find_through_api(target):
    def is_before(idx):
        return fetch(idx) < target
    l, r = 0, 1
    # get rightmost boundary 
    while fetch(r) != -1:
        r *= 2 
    # binary search
    # ...

# is it impossible to split arr into k subarrays, each with sum <= max_sum?
def is_before(arr, k, max_sum):
    splits_required = get_splits_required(arr, max_sum)
    return splits_required > k 
# return min number of subarrays with a given max sum, assume max_sum >= max(arr)
def get_splits_required(arr, max_sum):
    splits_required = 1 
    current_sum = 0
    for num in arr:
        if current_sum + num > max_sum:
            splits_required += 1 
            current_sum = num # start new subarray with current number 
        else:
            current_sum += num 
    return splits_required
def min_subarray_sum_split(arr, k):
    l, r = max(arr), sum(arr) # range for max subarray sum 
    if not is_before(arr, k, l):
        return l 
    while r - l > 1:
        mid = (l + r) // 2 
        if is_before(arr, k, mid):
            l = mid 
        else:
            r = mid 
    return r 

def num_refills(a, b):
    # can we pour 'num_pours' times?
    def is_before(num_pours):
        return num_pours * b <= a 
    # exponential search (repeatedly doubling until find upper bound)
    k = 1 
    while is_before(k * 2):
        k *= 2 
    # binary search between k and k * 2 
    l, r = k, k * 2 
    while r - l > 1:
        gap = r - l
        half_gap = gap >> 1 # bit shift instead of division
        mid = l + half_gap
        if is_before(mid):
            l = mid 
        else:
            r = mid 
    return l 

def get_ones_in_row(row):
    if row[0] == 0:
        return 0 
    if row[-1] == 1:
        return len(row)
    def is_before_row(idx):
        return row[idx] == 1 
    l, r = 0, len(row)
    while r - l > 1:
        mid = (l + r) // 2 
        if is_before_row(mid):
            l = mid 
        else:
            r = mid 
    return r 
def is_before(picture):
    water = 0
    for row in picture:
        water += get_ones_in_row(row)
    total = len(picture[0]) ** 2 
    return water / total < 0.5 
```

<!-- TOC --><a name="30-set-maps"></a>
## 30. Set & Maps

```py
def account_sharing(connections):
    seen = set()
    for ip, username in connections:
        if username in seen:
            return ip 
        seen.add(username)
    return ""

def most_shared_account(connections):
    user_to_count = dict()
    for _, user in connections:
        if not user in user_to_count:
            user_to_count[user] = 0
        user_to_count[user] += 1 
    most_shared_user = None 
    for user, count in user_to_count.items():
        if not most_shared_account or count > user_to_count[most_shared_user]:
            most_shared_user = user 
    return most_shared_user

def multi_account_cheating(users):
    unique_lists = set()
    for _, ips in users:
        immutable_list = tuple(sorted(ips))
        if immutable_list in unique_lists:
            return True 
        unique_lists.add(immutable_list)
    return False 

class DomainResolver:
    def __init__(self):
        self.ip_to_domains = dict()
        self.domain_to_subdomains = dict()
    def register_domain(self, ip, domain):
        if ip not in self.ip_to_domains:
            self.ip_to_domains[ip] = set()
        self.ip_to_domains[ip].add(domain)
    def register_subdomain(self, domain, subdomain):
        if domain not in self.domain_to_subdomains:
            self.domain_to_subdomains[domain] = set()
        self.domain_to_subdomains[domain].add(subdomain)
    def has_subdomain(self, ip, domain, subdomain):
        if ip not in self.ip_to_domains:
            return False 
        if domain not in self.domain_to_subdomains:
            return False 
        return subdomain in self.domain_to_subdomains[domain]
    
def find_squared(arr):
    # create map from number to index (allow multiple indices per number)
    num_to_indices = {}
    for i, num in enumerate(arr):
        if num not in num_to_indices:
            num_to_indices[num] = []
        num_to_indices[num].append(i)
    res = []
    # iterate through each number and check if its square exists in map 
    for i, num in enumerate(arr):
        square = num ** 2 
        if square in num_to_indices:
            for j in num_to_indices[square]:
                res.append([i, j])
    return res 

def suspect_students(answers, m, students):
    def same_row(desk1, desk2):
        return (desk1 - 1) // m == (desk2 - 1) // m 
    desk_to_index = {}
    for i, [student_id, desk, student_answers] in enumerate(students):
        if student_answers != answers:
            desk_to_index[desk] = i 
    sus_pairs = []
    for student_id, desk, answers in students:
        other_desk = desk + 1 
        if same_row(desk, other_desk) and other_desk in desk_to_index:
            other_student = students[desk_to_index[other_desk]]
            if answers == other_student[2]:
                sus_pairs.append([student_id, other_student[0]])
    return sus_pairs

def alphabetic_sum_product(words, target):
    sums = set()
    for word in words:
        sums.add(alphabetical_sum(word))
    for i in sums:
        if target % i != 0:
            continue 
        for j in sums:
            k = target / (i * j)
            if k in sums:
                return True 
    return False 

def find_anomalies(log):
    opened = {} # ticket -> agent who opened it 
    working_on = {} # agent -> ticket they're working on 
    seen = set() # tickets that were opened or closed 
    anomalies = set()
    for agent, action, ticket in log:
        if ticket in anomalies:
            continue 
        if action == "open":
            if ticket in seen:
                anomalies.add(ticket)
                continue 
            if agent in working_on:
                # if agent is working on another ticket, that ticket is anomalous 
                anomalies.add(working_on[agent])
            opened[ticket] = agent 
            working_on[agent] = ticket 
            seen.add(ticket)
        else:
            if ticket not in opened or opened[ticket] != agent:
                anomalies.add(ticket)
                continue 
            if agent not in working_on or working_on[agent] != ticket:
                anomalies.add(ticket)
                continue 
            del working_on[agent]
            del opened[ticket]
    # any tickets still open are anomalous 
    anomalies.update(opened.keys())
    return list(anomalies)

def set_intersection(sets):
    res = sets[0]
    for i in range(1, len(sets)):
        res = {elem for elem in sets[i] if elem in res}
    return res 
```

<!-- TOC --><a name="31-sorting"></a>
## 31. Sorting

```py
def mergesort(arr):
    n = len(arr)
    if n <= 1:
        return arr 
    left = mergesort(arr[:n // 2])
    right = mergesort(arr[n // 2:])
    return merge(left, right)

def quicksort(arr):
    if len(arr) <= 1:
        return arr 
    pivot = random.choice(arr)
    smaller, equal, larger = [], [], []
    for x in arr:
        if x < pivot: smaller.append(x)
        if x == pivot: equal.append(x)
        if x > pivot: larger.append(x)
    return quicksort(smaller) + equal + quicksort(larger)

def counting_sort(arr):
    if not arr: return []
    R = max(arr)
    counts = [0] * (R + 1)
    for x in arr:
        counts[x] += 1 
    res = []
    for x in range(R + 1):
        while counts[x] > 0:
            res.append(x)
            counts[x] -= 1 
    return res 

def descending_sort(strings):
    return sorted(strings, key=lambda s: s.lower(), reverse=True)

def sort_by_interval_end(intervals):
    return sorted(intervals, key=lambda interval: interval[1])

def sort_value_then_suit(deck):
    suit_map = {'clubs': 0, 'hearts': 1, 'spades': 2, 'diamonds': 3}
    return sorted(deck, key=lambda card: (card.value, suit_map[card.suit]))

def new_deck_order(deck):
    suit_map = {'hearts': 0, 'clubs': 1, 'diamonds': 2, 'spades': 3}
    return sorted(deck, key=lambda card: (suit_map[card.suit], card.value))

def stable_sort_by_value(deck):
    return sorted(deck, key=lambda card: card.value)

def letter_occurrences(word):
    letter_to_count = dict()
    for c in word:
        if c not in letter_to_count:
            letter_to_count[c] = 0
        letter_to_count[c] += 1 
    tuples = []
    for letter, count in letter_to_count.items():
        tuples.append((letter, count))
    tuples.sort(key=lambda x: (-x[1], x[0]))
    res = []
    for letter, _ in tuples:
        res.append(letter)
    return res 

def are_circles_nested(circles):
    circles.sort(key = lambda c: c[1], reverse=True)
    for i in range(len(circles) - 1):
        if not contains(circles[i], circles[i + 1]):
            return False 
    return True 
def contains(c1, c2):
    (x1, y1), r1 = c1 
    (x2, y2), r2 = c2 
    center_distance = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return center_distance + r2 < r1 

def process_operations(nums, operations):
    n = len(nums)
    deleted = set()
    sorted_indices = []
    for i in range(n):
        sorted_indices.append(i)
    sorted_indices.sort(key=lambda i: nums[i])
    smallest_idx = 0
    for op in operations:
        if 0 <= op < n:
            deleted.add(op)
        else:
            # skip until the next non-deleted smallest index 
            while smallest_idx < n and sorted_indices[smallest_idx] in deleted:
                smallest_idx += 1 
            if smallest_idx < n:
                deleted.add(sorted_indices[smallest_idx])
                smallest_idx += 1 
    res = []
    for i in range(n):
        if not i in deleted:
            res.append(nums[i])
    return res 

class Spreadsheet:
    def __init__(self, rows, cols):
        self.rows = rows 
        self.cols = cols 
        self.sheet = []
        for _ in range(rows):
            self.sheet.append([0] * cols)
    def set(self, row, col, value):
        self.sheet[row][col] = value 
    def get(self, row, col):
        return self.sheet[row][col]
    def sort_rows_by_column(self, col):
        self.sheet.sort(key=lambda row: row[col])
    def sort_columns_by_row(self, row):
        columns_with_values = []
        for col in range(self.cols):
            columns_with_values.append((col, self.sheet[row][col]))
        sorted_columns = sorted(columns_with_values, key=lambda x: x[1])
        sorted_sheet = []
        for r in range(self.rows):
            new_row = []
            for col, _ in sorted_columns:
                new_row.append(self.sheet[r][col])
            sorted_sheet.append(new_row)
        self.sheet = sorted_sheet 

def bucket_sort(books):
    if not books: return []
    min_year = min(book.year_published for book in books)
    max_year = max(book.year_published for book in books)
    buckets = [[] for _ in range(max_year - min_year + 1)]
    for book in books:
        buckets[book.year_published - min_year].append(book)
    res = []
    for bucket in buckets:
        for book in bucket:
            res.append(book)
    return res 

def quickselect(arr, k):
    if len(arr) == 1:
        return arr[0]
    pivot_index = random.randint(0, len(arr) - 1)
    pivot = arr[pivot_index]
    smaller, larger = [], []
    for x in arr:
        if x < pivot: smaller.append(x)
        elif x > pivot: larger.append(x)
    if k <= len(smaller):
        return quickselect(smaller, k)
    elif k == len(smaller) + 1:
        return pivot 
    else:
        return quickselect(larger, k - len(smaller) - 1)
def first_k(arr, k):
    if len(arr) == 0: return []
    kth_val = quickselect(arr, k)
    return [x for x in arr if x <= kth_val]
```

<!-- TOC --><a name="32-stacks-queues"></a>
## 32. Stacks & Queues

```py
class Stack:
    def __init__(self):
        self.array = []
    def push(self, value):
        self.array.append(value)
    def pop(self):
        if self.is_empty():
            raise IndexError('stack is empty')
        val = self.array[-1]
        self.array.pop()
        return val 
    def peek(self):
        if self.is_empty():
            raise IndexError('stack is empty')
        return self.array[-1]
    def size(self):
        return len(self.array)

def compress_array(arr):
    stack = []
    for num in arr:
        while stack and stack[-1] == num:
            num += stack.pop()
        stack.append(num)
    return stack 

def compress_array_by_k(arr, k):
    stack = []
    def merge(num):
        if not stack or stack[-1][0] != num:
            stack.append([num, 1])
        elif stack[-1][1] < k - 1:
            stack[-1][1] += 1 
        else:
            stack.pop()
            merge(num * k)
    for num in arr:
        merge(num)
    res = []
    for num, count in stack:
        for _ in range(count):
            res.append(num)
    return res 

class ViewerCounter:
    def __init__(self, window):
        self.queues = {"guest": Queue(), "follower": Queue(), "subscriber": Queue()}
        self.window = window 
    def join(self, t, v):
        self.queues[v].put(t)
    def get_viewers(self, t, v):
        queue = self.queues[v]
        while not queue.empty() and queue.peek() < t - self.window:
            queue.pop()
        return queue.size()

def current_url(actions):
    stack = []
    for action, value in actions:
        if action == "go":
            stack.append(value)
        else:
            while len(stack) > 1 and value > 0:
                stack.pop()
                value -= 1 
    return stack[-1]

def current_url_followup(actions):
    stack = []
    forward_stack = []
    for action, value in actions:
        if action == "go":
            stack.append(value)
            forward_stack = []
        elif action == "back":
            while len(stack) > 1 and value > 0:
                forward_stack.append(stack.pop())
                value -= 1 
        else:
            while forward_stack and value > 0:
                stack.append(forward_stack.pop())
                value -= 1 
    return stack[-1]

def balanced(s):
    height = 0
    for c in s:
        if c == '(':
            height += 1 
        else:
            height -= 1 
            if height < 0:
                return False 
    return height == 0

def max_balanced_partition(s):
    height = 0 
    res = 0 
    for c in s:
        if c == '(':
            height += 1 
        else:
            height -= 1 
            if height == 0:
                res += 1 
    return res 

def balanced_brackets(s, brackets):
    open_to_close = dict()
    close_set = set()
    for pair in brackets:
        open_to_close[pair[0]] = pair[1]
        close_set.add(pair[1])
    stack = []
    for c in s:
        if c in open_to_close:
            stack.append(open_to_close[c])
        elif c in close_set:
            if not stack or stack[-1] != c:
                return False 
            stack.pop()
    return len(stack) == 0
```

<!-- TOC --><a name="33-recursion"></a>
## 33. Recursion

```py
def moves(seq):
    res = []
    def moves_rec(pos):
        if pos == len(seq):
            return 
        if seq[pos] == '2':
            moves_rec(pos+1)
            moves_rec(pos+2)
        else:
            res.append(seq[pos])
            moves_rec(pos+1)
    moves_rec(0)
    return ''.join(res)

def nested_array_sum(arr):
    res = 0
    for elem in arr:
        if isinstance(elem, int):
            res += elem 
        else:
            res += nested_array_sum(elem)
    return res 

def reverse_in_place(arr):
    reverse_rec(arr, 0, len(arr) - 1)
    def reverse_rec(arr, i, j):
        if i >= j:
            return 
        arr[i], arr[j] = arr[j], arr[i]
        reverse_rec(arr, i + 1, j - 1)

def power(a, p, m):
    if p == 0:
        return 1 
    if p % 2 == 0:
        half = power(a, p // 2, m)
        return (half * half) % m 
    return (a * power(a, p - 1, m)) % m 

def fib(n):
    memo = {}
    def fib_rec(i):
        if i <= 1:
            return 1 
        if i in memo:
            return memo[i]
        memo[i] = fib_rec(i - 1) + fib_rec(i - 2)
        return memo[i]
    return fib_rec(n)

def blocks(n):
    memo = dict()
    def roof(n):
        if n == 1:
            return 1 
        if n in memo:
            return memo[n]
        memo[n] = roof(n - 1) * 2 + 1 
        return memo[n]
    def blocks_rec(n):
        if n == 1:
            return 1 
        return blocks_rec(n - 1) * 2 + roof(n)
    return blocks_rec(n)

def max_laminal_sum(arr):
    # return max sum for subliminal array in arr[l:r]
    def max_laminal_sum_rec(l, r):
        if r - l == 1:
            return arr[l]
        mid = (l + r) // 2 
        option1 = max_laminal_sum_rec(l, mid)
        option2 = max_laminal_sum_rec(mid, r)
        option3 = sum(arr)
        return max(option1, option2, option3)
    return max_laminal_sum_rec(0, len(arr))
```

<!-- TOC --><a name="34-linked-lists"></a>
## 34. Linked Lists

```py

```

<!-- TOC --><a name="35-trees"></a>
## 35. Trees 

```py

```

<!-- TOC --><a name="36-graphs"></a>
## 36. Graphs

```py

```

<!-- TOC --><a name="37-heaps"></a>
## 37. Heaps 

```py

```

<!-- TOC --><a name="38-sliding-windows"></a>
## 38. Sliding Windows 

```py

```

<!-- TOC --><a name="39-backtracking"></a>
## 39. Backtracking

```py

```

<!-- TOC --><a name="40-dynamic-programming"></a>
## 40. Dynamic Programming

```py

```

<!-- TOC --><a name="41-greedy-algorithms"></a>
## 41. Greedy Algorithms

```py

```

<!-- TOC --><a name="42-topological-sort"></a>
## 42. Topological Sort 

```py

```

<!-- TOC --><a name="43-prefix-sums"></a>
## 43. Prefix Sums 

```py

```

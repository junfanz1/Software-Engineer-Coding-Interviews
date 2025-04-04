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
class Node:
    def __init__(self, val):
        self.val = val 
        self.prev = None
        self.next = None 

def add_to_end(head, val):
    cur = head 
    while cur.next:
        cur = cur.next 
    cur.next = Node(val)

class SinglyLinkedList:
    def __init__(self):
        self.head = None 
        self._size = 0 
    def size(self):
        return self._size 
    def push_front(self, val):
        new = Node(val)
        new.next = self.head 
        self.head = new 
        self._size += 1 
    def pop_front(self):
        if not self.head:
            return None 
        val = self.head.val 
        self.head = self.head.next 
        self._size -= 1 
        return val 
    def push_back(self, val):
        new = Node(val)
        self._size += 1 
        if not self.head:
            self.head = new 
            return 
        cur = self.head 
        while cur.next:
            cur = cur.next 
        cur.next = new 
    def pop_back(self):
        if not self.head:
            return None 
        self._size -= 1
        if not self.head.next:
            val = self.head.val 
            self.head = None 
            return val 
        cur = self.head 
        while cur.next and cur.next.next:
            cur = cur.next 
        val = cur.next.val 
        cur.next = None 
        return val 
    def contains(self, val):
        cur = self.head 
        while cur:
            if cur.val == val:
                return cur 
            cur = cur.next 
        return None 
    
class Node:
    def __init__(self, val):
        self.val = val 
        self.next = None 
class Queue:
    def __init__(self):
        self.head = None 
        self.tail = None 
        self._size = 0 
    def empty(self):
        return not self.head 
    def size(self):
        return self._size 
    def push(self, val):
        new = Node(val)
        if self.tail:
            self.tail.next = new 
        self.tail = new 
        if not self.head:
            self.head = new 
        self._size += 1 
    def pop(self):
        if self.empty():
            raise IndexError('empty queue')
        val = self.head.val 
        self.head = self.head.next 
        if not self.head:
            self.tail = None 
        self._size -= 1
        return val 
    
def copy_list(head):
    if not head:
        return None 
    new_head = Node(head.val)
    cur_new = new_head 
    cur_old = head.next 
    while cur_old:
        cur_new.next = Node(cur_old.val)
        cur_new = cur_new.next 
        cur_old = cur_old.next 
    return new_head 

def reverse_list(head):
    prev = None 
    cur = head 
    while cur:
        nxt = cur.next 
        cur.next = prev 
        prev = cur 
        cur = nxt 
    return prev 

def reverse_section(head, left, right):
    dummy = Node(0)
    dummy.next = head 
    # find nodes before and after section
    if left == 0:
        prev = dummy 
    else:
        prev = node_at_index(head, left - 1)
    if not prev or not prev.next:
        # nothing to reverse 
        return head 
    nxt = node_at_index(head, right + 1) # may be none 
    # break out section
    section_head = prev.next 
    prev.next = None 
    section_tail = section_head 
    while section_tail.next != nxt:
        section_tail = section_tail.next 
    section_tail.next = None 
    # reverse section, same as reverse linked list solution
    old_section_head = section_head 
    new_section_head = reverse_list(section_head)
    # reattach section
    prev.next = new_section_head
    old_section_head.next = nxt 
    return dummy.next 

def has_cycle(head):
    slow, fast = head, head 
    while fast and fast.next:
        slow = slow.next 
        fast = fast.next.next 
        if slow == fast:
            return True 
    return False 

def convert_to_array(self, node):
    cur = node 
    while cur.prev:
        cur = cur.prev 
    res = []
    while cur:
        res.append(cur.val)
        cur = cur.next 
    return res 

def get_middle(head):
    slow, fast = head, head 
    while fast and fast.next:
        slow = slow.next 
        fast = fast.next.next
    return slow 

def remove_kth_node(head, k):
    if not head:
        return None 
    dummy = Node(0)
    dummy.next = head 
    fast = dummy
    slow = dummy 
    for _ in range(k):
        fast = fast.next 
    while fast and fast.next:
        fast = fast.next 
        slow = slow.next 
    slow.next = slow.next.next 
    return dummy.next 

def merge(head1, head2):
    dummy = Node(0)
    cur = dummy 
    p1, p2 = head1, head2 
    while p1 and p2:
        cur.next = p1 
        cur = cur.next 
        p1 = p1.next 
        cur.next = p2 
        p2 = p2.next 
        cur = cur.next 
    if p1:
        cur.next = p1 
    else:
        cur.next = p2 
    return dummy.next 

def remove_duplicates(head):
    cur = head 
    while cur and cur.next:
        if cur.val == cur.next.val:
            cur.next = cur.next.next 
        else:
            cur = cur.next 
    return head 
```

<!-- TOC --><a name="35-trees"></a>
## 35. Trees 

```py
# DFS
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val 
        self.left = left 
        self.right = right 
    def is_leaf(node):
        if not node:
            return False 
        return not node.left and not node.right 
    def children_values(node):
        if not node:
            return []
        values = []
        if node.left:
            values.append(node.left.val)
        if node.right:
            values.append(node.right.val)
        return values 
    def grandchildren_values(node):
        if not node:
            return []
        values = []
        for child in [node.left, node.right]:
            if child and child.left:
                values.append(child.left.val)
            if child and child.right:
                values.append(child.right.val)
        return values 
    def subtree_size(node):
        if not node:
            return 0 
        left_size = subtree_size(node.left)
        right_size = subtree_size(node.right)
        return left_size + right_size + 1 # 1 for node 
    def subtree_height(node):
        if not node:
            return 0 
        left_height = subtree_height(node.left)
        right_height = subtree_height(node.right)
        return max(left_height, right_height) + 1
    
class Node:
    def __init__(self, id, parent, left, right):
        self.id = id 
        self.parent = parent 
        self.left = left 
        self.right = right 
    def is_root(node):
        return not node.parent 
    def ancestor_ids(node):
        ids = []
        while node.parent:
            node = node.parent 
            ids.append(node.id)
        return ids 
    def depth(node):
        res = 0 
        while node.parent:
            node = node.parent 
            res += 1 
        return res 
    def LCA(node1, node2):
        depth1 = depth(node1)
        depth2 = depth(node2)
        while depth1 > depth2:
            node1 = node1.parent 
            depth1 -= 1
        while depth2 > depth1:
            node2 = node2.parent 
            depth2 -= 1 
        while node1.id != node2.id:
            node1 = node1.parent 
            node2 = node2.parent 
        return node1.id 
    def distance(node1, node2):
        lca_id = LCA(node1, node2)
        dist = 0 
        while node1.id != lca_id:
            dist += 1 
            node1 = node1.parent 
        while node2.id != lca_id:
            dist += 1 
            node2 = node2.parent 
        return dist 
    def size(node):
        if not node:
            return 0 
        return size(node.left) + size(node.right) + 1 
    def preorder(root):
        if not root:
            return 
        print(root.val)
        preorder(root.left)
        preorder(root.right)
    def inorder(root):
        if not root:
            return 
        inorder(root.left)
        print(root.val)
        inorder(root.right)
    def postorder(root):
        if not root:
            return 
        postorder(root.left)
        postorder(root.right)
        print(root.val)
    def visit(node, info_passed_down):
        if base_case:
            return info_to_pass_up
        a = vist(node.left, info_to_pass_down)
        b = visit(node.right, info_to_pass_down)
        global_state = info_stored_globally
        return info_to_pass_up
    
def longest_aligned_chain(root):
    res = 0 
    def visit(node, depth): # inner recursive function
        nonlocal res # to make res visible inside visit()
        if not node:
            return 0 
        left_chain = visit(node.left, depth + 1)
        right_chain = visit(node.right, depth + 1)
        current_chain = 0 
        if node.val == depth:
            current_chain = 1 + max(left_chain, right_chain)
            res = max(res, current_chain)
        return current_chain 
    visit(root, 0) # trigger DFS, which updates global res
    return res 

def hidden_message(root):
    message = []
    def visit(node):
        if not node:
            return 
        if node.text[0] == 'b':
            message.append(node.text[1])
            visit(node.left)
            visit(node.right)
        elif node.text[0] == 'i':
            visit(node.left)
            message.append(node.text[1])
            visit(node.right)
        else:
            visit(node.left)
            visit(node.right)
            message.append(node.text[1])
    visit(root)
    return ''.join(message)

def most_stacked(root):
    pos_to_count = dict()
    def visit(node, r, c):
        if not node:
            return 
        if (r, c) not in pos_to_count:
            pos_to_count[(r, c)] = 0 
        pos_to_count[(r, c)] += 1 
        visit(node.left, r + 1, c)
        visit(node.right, r, c + 1)
    visit(root, 0, 0)
    return max(pos_to_count.values())

def invert(root):
    if not root:
        return None 
    root.left, root.right = invert(root.right), invert(root.left)
    return root 

def evaluate(root):
    if root.kind == "num":
        return root.num 
    children_evals = []
    for child in root.children:
        children_evals.append(evaluate(child))
    if root.kind == "sum":
        return sum(children_evals)
    if root.kind == "product":
        return product(children_evals)
    if root.kind == "max":
        return max(children_evals)
    if root.kind == "min":
        return min(children_evals)
    raise ValueError('invalid node kind')

# BFS
def level_order(root):
    Q = Queue()
    Q.add(root)
    while not Q.empty():
        node = Q.pop()
        if not node:
            continue 
        print(node.eval)
        Q.add(node.left)
        Q.add(node.right)

def node_depth_queue_recipe(root):
    Q = Queue()
    Q.add((root, 0))
    while not Q.empty():
        node, depth = Q.pop()
        if not node:
            continue 
        # do something with node and depth
        Q.add((node.left, depth+1))
        Q.add((node.right, depth+1))

def left_view(root):
    if not root:
        return []
    Q = Queue()
    Q.add((root, 0))
    res = [root.val]
    current_depth = 0 
    while not Q.empty():
        node, depth = Q.pop()
        if not node:
            continue 
        if depth == current_depth + 1:
            res.append(node.val)
            current_depth += 1 
        Q.add((node.left, depth+1))
        Q.add((node.right, depth+1))
    return res 

def level_counts(root):
    Q = Queue()
    Q.add((root, 0))
    level_count = defaultdict(int)
    while not Q.empty():
        node, depth = Q.pop()
        if not node:
            continue 
        level_count[depth] += 1 
        Q.add((node.left, depth + 1))
        Q.add((node.right, depth + 1))
    return level_count
def most_prolific_level(root):
    level_count = level_counts(root)
    res = -1 
    max_prolificness = -1 # less than any valid prolificness
    for level in level_count:
        if level + 1 not in level_count:
            continue 
        prolificness = level_count[level + 1] / level_count[level]
        if prolificness > max_prolificness:
            max_prolificness = prolificness
            res = level 
    return res 

def zig_zag_order(root):
    res = []
    Q = Queue()
    Q.add((root, 0))
    cur_level = []
    cur_depth = 0 
    while not Q.empty():
        node, depth = Q.pop()
        if not node:
            continue 
        if depth > cur_depth:
            if cur_depth % 2 == 0:
                res += cur_level 
            else:
                res += cur_level[::-1] # reverse order
            cur_level = []
            cur_depth = depth 
        cur_level.append(node)
        Q.add((node.left, depth + 1))
        Q.add((node.right, depth + 1))
    if cur_depth % 2 == 0: # add last level 
        res += cur_level
    else:
        res += cur_level[::-1]
    return res 

# Binary Search Tree
def find(root, target):
    cur_node = root
    while cur_node:
        if cur_node.val == target:
            return True 
        elif cur_node.val > target:
            cur_node = cur_node.left 
        else:
            cur_node = cur_node.right 
    return False 

def find_closest(root, target):
    cur_node = root 
    next_above, next_below = math.inf, -math.inf
    while cur_node:
        if cur_node.val == target:
            return cur_node.val 
        elif cur_node.val > target:
            next_above = cur_node.val 
            cur_node = cur_node.left 
        else:
            next_below = cur_node.val 
            cur_node = cur_node.right 
    if next_above - target < target - next_below:
        return next_above 
    return next_below 

def is_bst(root):
    prev_value = -math.inf 
    res = True 
    def visit(node):
        nonlocal prev_value, res 
        if not node or not res:
            return 
        visit(node.left)
        if node.val < prev_value:
            res = False 
        else:
            prev_value = node.val 
        visit(node.right)
    visit(root)
    return res 
```

<!-- TOC --><a name="36-graphs"></a>
## 36. Graphs

```py
def num_nodes(graph):
    return len(graph)
def num_edges(graph):
    count = 0 
    for node in range(len(graph)):
        count += len(graph[node])
    return count // 2 # halved because we counted each edge from both endpoints 
def degree(graph, node):
    return len(graph[node])
def print_neighbors(graph, node):
    for nbr in graph[node]:
        print(nbr)
def build_adjency_list(V, edge_list):
    graph = [[] for _ in range(V)]
    for node1, node2 in edge_list:
        graph[node1].append(node2)
        graph[node2].append(node1)
    return graph
def adjacent(graph, node1, node2):
    for nbr in graph[node1]:
        if nbr == node2: return True 
    return False 

def validate(graph):
    V = len(graph)
    for node in range(V):
        seen = get()
        for nbr in graph[node]:
            if nbr < 0 or nbr >= V: return False # invalid node index 
            if nbr == node: return False # self-loop
            if nbr in seen: return False # parallel edge
            seen.add(nbr)
    edges = set()
    for node1 in range(V):
        for node2 in graph[node1]:
            edge = (min(node1, node2), max(node1, node2))
            if edge in edges:
                edges.remove(edge)
            else:
                edges.add(edge)
    return len(edges) == 0

def graph_dfs(graph, start)：
    visited = {start}
    def visit(node):
        # do something
        for nbr in graph[node]:
            if not nbr in visited:
                visited.add(nbr)
                visit(nbr)
    visit(start)

def tree_dfs(root):
    def visit(node):
        # do something
        if root.left:
            visit(root.left)
        if root.right:
            visit(root.right)
    if root:
        visit(root)

def count_connected_components(graph):
    count = 0 
    visited = set()
    for node in range(len(graph)):
        if node not in visited:
            visited.add(node)
            visit(node)
            count += 1 
    return count 

def path(graph, node1, node2):
    predecessors = {node2: None} # starting node doesn't have predecessor
    def visit(node):
        for nbr in graph[node]:
            if nbr not in predecessors:
                predecessors[nbr] = node 
                visit(nbr)
    visit(node2)
    if node1 not in predecessors:
        return [] # node1 node2 disconnected
    path = [node1]
    while path[len(path) - 1] != node2:
        path.append(predecessors[path[len(path) - 1]])
    return path 

def is_tree(graph):
    predecessors = {0: None} # start from node 0 (doesn't matter)
    found_cycle = False 
    def visit(node):
        nonlocal found_cycle 
        if found_cycle:
            return 
        for nbr in graph[node]:
            if nbr not in predecessors:
                predecessors[nbr] = node 
                visit(nbr)
            elif nbr != predecessors[node]:
                found_cycle = True 
    visit(0)
    connected = len(predecessors) == len(graph)
    return not found_cycle and connected 

def connected_component_queries(graph, queries):
    node_to_cc = {}
    def visit(node, cc_id):
        if node in node_to_cc:
            return 
        node_to_cc[node] = cc_id
        for nbr in graph[node]:
            visit(nbr, cc_id)
    cc_id = 0 
    for node in range(len(graph)):
        if node not in node_to_cc:
            visit(node, cc_id)
            cc_id += 1 
    res = []
    for node1, node2 in queries:
        res.append(node_to_cc[node1] == node_to_cc[node2])
    return res 

def strongly_connected(graph):
    V = len(graph)
    visited = set()
    visit(graph, visited, 0)
    if len(visited) < V:
        return False 
    reverse_graph = [[] for _ in range(V)]
    for node in range(V):
        for nbr in graph[node]:
            reverse_graph[nbr].append(node)
    reverse_visited = set()
    visit(reverse_graph, reverse_visited, 0)
    return len(reverse_visited) == V 

def max_hilliness(graph, heights):
    node_to_cc = label_nodes_with_cc_ids(graph)
    V = len(graph)
    cc_to_elevation_gain_sum = {}
    cc_to_num_edges = {}
    for node in range(V):
        cc = node_to_cc[node]
        if cc not in cc_to_num_edges:
            cc_to_elevation_gain_sum[cc] = 0 
            cc_to_num_edges[cc] = 0 
        for nbr in graph[node]:
            if nbr > node:
                cc_to_num_edges[cc] += 1 
                cc_to_elevation_gain_sum[cc] += abs(heights[node] - heights[nbr])
    res = 0 
    for cc in cc_to_num_edges:
        res = max(res, cc_to_elevation_gain_sum[cc] / cc_to_num_edges[cc])

def first_time_all_connected(V, cables):
    def visit(graph, visited, node):
        for nbr in graph[node]:
            if nbr not in visited:
                visited.add(nbr)
                visit(graph, visited, nbr)

    def is_before(cable_index):
        graph = [[] for _ in range(V)]
        for i in range(cable_index + 1):
            node1, node2 = cables[i]
            graph[node1].append(node2)
            graph[node2].append(node1)
        visited = {0}
        visit(graph, visited, 0)
        return len(visited) < V 
    l, r = 0, len(cables) - 1 
    if is_before(r):
        return -1 
    while r - l > 1:
        mid = l + (r - l) // 2 
        if is_before(mid):
            l = mid 
        else:
            r = mid 
    return r 

# BFS
def graph_bfs(graph, start):
    Q = Queue()
    Q.push(start)
    distances = {start: 0}
    while not Q.empty():
        node = Q.pop()
        for nbr in graph[node]:
            if nbr not in distances:
                distances[nbr] = distances[node] + 1
                Q.push(nbr)
    # do something 

def tree_bfs(root):
    Q = Queue()
    Q.push(root)
    while not Q.empty():
        node = Q.pop()
        if not node:
            continue 
        # do something 
        Q.push(node.left)
        Q.push(node.right)

def shortest_path_queries(graph, start, queries):
    Q = Queue()
    Q.push(start)
    predecessors = {start: None}
    while not Q.empty():
        node = Q.pop()
        for nbr in graph[node]:
            if nbr not in predecessors:
                predecessors[nbr] = node 
                Q.push(nbr)
    res = []
    for node in queries:
        if node not in predecessors:
            res.append([])
        else:
            path = [node]
            while path[len(path) - 1] != start:
                path.append(predecessors[path[len(path) - 1]])
            path.reverse()
            res.append(path)
    return res 

def walking_distance_to_coffee(graph, node1, node2, node3):
    distances1 = bfs(graph, node1) # BFS
    distances2 = bfs(graph, node2)
    distances3 = bfs(graph, node3)
    res = math.inf 
    for i in range(len(graph)):
        res = min(res, distances1[i] + distances2[i] + distances3[i])
    return res 

def multisource_bfs(graph, sources):
    Q = Queue()
    distances = {}
    for start in sources:
        Q.push(start)
        distances[start] = 0
    while not Q.empty(): # BFS
        node = Q.pop()
        for nbr in graph[node]:
            if nbr not in distances:
                distances[nbr] = distances[node] + 1 
                Q.push(nbr)
    # do something 

def grid_dfs(grid, start_r, start_c):
    # returns if (r, c) is in bounds, not visited, and walkable
    def is_valid(r, c):
        # do something 
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    visited = {(start_r, start_c)}
    def visit(r, c):
        # do something with (r, c)
        for dir_r, dir_c in directions:
            nbr_r, nbr_c = r + dir_r, c + dir_c 
            if is_valid(nbr_r, nbr_c):
                visited.add((nbr_r, nbr_c))
                visit(nbr_r, nbr_c)
    visit(start_r, start_c)

def grid_bfs(grid, start_r, start_c):
    # returns if (r, c) is in bounds, not visited, and walkable
    def is_valid(r, c):
        # do something 
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    Q = Queue()
    Q.push((start_r, start_c))
    distances = {(start_r, start_c): 0}
    while not Q.empty():
        r, c = Q.pop()
        for dir_r, dir_c in directions:
            nbr_r, nbr_c = r + dir_r, c + dir_c 
            if is_valid(nbr_r, nbr_c):
                distances[(nbr_r, nbr_c)] = distances[(r, c)] + 1 
                Q.push((nbr_r, nbr_c))
    # do something with distances 

def count_islands(grid):
    R, C = len(grid), len(grid[0])
    count = 0 
    visited = set()
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 1 and (r, c) not in visited:
                visited.add((r, c))
                dfs(grid, visited, r, c) # normal grid DFS
                count += 1 
    return count 

def exit_distances(maze):
    R, C = len(maze), len(maze[0])
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    distances = [[-1] * C for _ in range(R)]
    Q = Queue()
    for r in range(R):
        for c in range(C):
            if maze[r][c] == 'o':
                distances[r][c] = 0
                Q.push((r, c))
    while not Q.empty():
        r, c = Q.pop()
        for dir_r, dir_c in directions:
            nbr_r, nbr_c = r + dir_r, c + dir_c
            if (0 <= nbr_r < R and 0 <= nbr_c < C and 
                maze[nbr_r][nbr_c] != 'x' and distances[nbr_r][nbr_c] == -1):
                distances[nbr_r][nbr_c] = distances[r][c] + 1 
                Q.push((nbr_r, nbr_c))
    return distances 

def segment_distance(min1, max1, min2, max2):
    return max(0, max(min1, min2) - min(max1, max2))
def distance(furniture1, furniture2):
    x_min1, y_min1, x_max1, y_max1 = furniture1 
    x_min2, y_min2, x_max2, y_max2 = furniture2 
    x_gap = segment_distance(x_min1, x_max1, x_min2, x_max2)
    y_gap = segment_distance(y_min1, y_max1, y_min2, y_max2)
    if x_gap == 0:
        return y_gap 
    elif y_gap == 0:
        return x_gap 
    else:
        return math.sqrt(x_gap ** 2 + y_gap ** 2)
def can_reach(furniture, d):
    V = len(furniture)
    graph = [[] for _ in range(V)]
    for i in range(V):
        for j in range(i + 1, V):
            if distance(furniture[i], furniture[j]) <= d:
                graph[i].append(j)
                graph[j].append(i)
    visited = {0}
    def visit(node): # DFS
        # ... 
    visit(0)
    return V-1 in visited 
```

<!-- TOC --><a name="37-heaps"></a>
## 37. Heaps 

```py
def first_k(arr, k):
    arr.sort()
    return arr[:k]
def first_k_min_heap(arr, k):
    min_heap = Heap(priority_comparator=lambda x, y: x < y, heap=arr)
    res = []
    for i in range(k):
        res.append(min_heap.pop())
    return res 

def parent(idx):
    if idx == 0:
        return -1 # root has no parent 
    return (idx - 1) // 2 
def left_child(idx):
    return 2 * idx + 1 
def right_child(idx):
    return 2 * idx + 2 

class Heap:
    # if higher_priority(x, y) is True, x has higher priority than y
    def __init__(self, higher_priority=lambda x, y: x < y, heap=None):
        self.heap = []
        if heap is not None:
            self.heap = heap
        self.heap = heap if heap is not None else []
        self.higher_priority = higher_priority
        if heap:
            self.heapify()
    def size(self):
        return len(self.heap)
    def top(self):
        if not self.heap:
            return None 
        return self.heap[0]
    def push(self, elem):
        self.heap.append(elem)
        self.bubble_up(len(self.heap)-1)
    def bubble_up(self, idx):
        if idx == 0:
            return # root can't be bubbled up 
        parent_idx = parent(idx)
        if self.higher_priority(self.heap[idx], self.heap[parent_idx]):
            self.heap[idx], self.heap[parent_idx] = self.heap[parent_idx], self.heap[idx]
            self.bubble_up[parent_idx]
    def pop(self):
        if not self.heap: return None 
        top = self.heap[0]
        if len(self.heap) == 1:
            self.heap = []
            return top 
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        self.bubble_down(0)
        return top 
    def bubble_down(self, idx): 
        l_i, r_i = left_child(idx), right_child(idx)
        is_leaf = l_i >= len(self.heap)
        if is_leaf: return # leaves can't be bubbled down
        child_i = l_i # index for highest priority child 
        if r_i < len(self.heap) and self.higher_priority(self.heap[r_i], self.heap[l_i]):
            child_i = r_i 
        if self.higher_priority(self.heap[child_i], self.heap[idx]):
            self.heap[idx], self.heap[child_i] = self.heap[child_i], self.heap[idx]
            self.bubble_down(child_i)
    def heapify(self):
        for idx in range(len(self.heap) // 2, -1, -1):
            self.bubble_down(idx)
    
def heapsort(arr):
    min_heap = Heap(priority_comparator=lambda x, y: x < y, heap=arr)
    res = []
    for _ in range(len(arr)):
        res.append(min_heap.pop())
    return res 

class TopSongs:
    def __init__(self, k):
        self.k = k 
        self.min_heap = Heap(higher_priority=lambda x, y: x[1] < y[1])
    def register_plays(self, title, plays):
        if self.min_heap.size() < self.k:
            self.min_heap.push((title, plays))
        elif plays > self.min_heap.top()[1]:
            self.min_heap.pop()
            self.min_heap.push((title, plays))
    def top_k(self):
        top_songs = []
        for title, _ in self.min_heap.heap:
            top_songs.append(title)
        return top_songs 
    
class TopSongsWithUpdates:
    def __init__(self, k):
        self.k = k 
        self.max_heap = Heap(higher_priority=lambda x, y: x[1] > y[1])
        self.total_plays = {}
    def register_plays(self, title, plays):
        new_total_plays = plays 
        if title in self.total_plays:
            new_total_plays += self.total_plays[title]
        sekf.total_plays[title] = new_total_plays
        self.max_heap.push((title, new_total_plays))
    def top_k(self):
        top_songs = []
        while len(top_songs) < self.k and self.max_heap.size() > 0:
            title, plays = self.max_heap.pop()
            if self.total_plays[title] == plays: # not stale 
                top_songs.append(title)
        # restore max-heap 
        for title in top_songs:
            self.max_heap.push((title, self.total_plays[title]))
        return top_songs
    
class PopularSongs:
    def __init__(self):
        # max-heap for lower half 
        self.lower_max_heap = Heap(higher_priority=lambda x, y: x > y)
        # min-heap for upper half 
        self.upper_min_heap = Heap()
        self.play_counts = {}
    def register_plays(self, title, plays):
        self.play_counts[title] = plays 
        if self.upper_min_heap.size() == 0 or plays >= self.upper_min_heap.top():
            self.upper_min_heap.push(plays)
        else:
            self.lower_max_heap.push(plays)
        # distribute elements if they're off by more than one 
        if self.lower_max_heap.size() > self.upper_min_heap.size():
            self.upper_min_heap.push(self.lower_max_heap.pop())
        elif self.upper_min_heap.size() > self.lower_max_heap.size() + 1:
            self.lower_max_heap.push(self.upper_min_heap.pop())
    def is_popular(self, title):
        if title not in self.play_counts:
            return False 
        if self.lower_max_heap.size() == self.upper_min_heap.size():
            median = (self.upper_min_heap.top() + self.lower_max_heap.top()) / 2 
        else:
            median = self.upper_min_heap.top()
        return self.play_counts[title] > median 
    
def top_k_across_genres(genres, k):
    initial_elems = [] # (plays, genre_index, song_index) tuples.
    for genre_index, song_list in enumerate(genres):
        plays = song_list[0][1]
        initial_elems.append((plays, genre_index, 0))
    max_heap = Heap(higher_priority=lambda x, y: x[0] > y[0], heap=initial_elems)
    top_k = []
    while len(top_k) < k and max_heap.size() > 0:
        plays, genre_index, song_index = max_heap.pop()
        song_name = genres[genre_index][song_index][0]
        top_k.append(song_name)
        song_index += 1 
        if song_index < len(genres[genre_index]):
            plays = genres[genre_index][song_index][1]
            max_heap.push((plays, genre_index, song_index))
    return top_k 

def make_playlist(songs):
    # group songs by artist
    artist_to_songs = {}
    for song, artist in songs:
        if artist not in artist_to_songs:
            artist_to_songs[artist] = []
        artist_to_songs[artist].append(song)
    heap = Heap(higher_priority=lambda a, b: len(a[1]) > len(b[1]))
    for artist, song_list in artist_to_songs.items():
        heap.push((artist, song_list))
    res = []
    last_artist = None 
    while heap.size() > 0:
        artist, song_list = heap.pop()
        if artist != last_artist:
            res.append(song_list.pop())
            last_artist = artist
            if song_list: # if artist has more songs, readd it
                heap.push((artist, song_list))
        else:
            # find different artist 
            if heap.size() == 0:
                return [] # no valid solution 
            artist2, song_list2 = heap.pop()
            res.append(song_list2.pop())
            last_artist = artist2 
            # readd artists we popped 
            if song_list2:
                heap.push((artist2, song_list2))
            heap.push((artist, song_list))
    return res 

def sum_of_powers(primes, n):
    m = 10**9 + 7 
    # initialize heap with first power of each prime 
    # each element is tuple (power, base)
    elems = [(p, p) for p in primes]
    min_heap = Heap(higher_priority=lambda x, y: x[0] < y[0], heap=elems)
    res = 0 
    for _ in range(n):
        power, base = min_heap.pop()
        res = (res + power) % m 
        min_heap.push(((power * base) % m, base))
    return res 
```

<!-- TOC --><a name="38-sliding-windows"></a>
## 38. Sliding Windows 

```py
def most_weekly_sales(sales):
    l, r = 0, 0 
    window_sum = 0 
    cur_max = 0
    while r < len(sales):
        window_sum += sales[r]
        r += 1 
        if r - l == 7:
            cur_max = max(cur_max, window_sum)
            window_sum -= sales[l]
            l += 1 
    return cur_max 

def has_unique_k_days(best_seller, k):
    l, r = 0, 0 
    window_counts = {}
    while r < len(best_seller):
        if not best_seller[r] in window_counts:
            window_counts[best_seller[r]] = 0 
        window_counts[best_seller[r]] += 1 
        r += 1 
        if r - l == k:
            if len(window_counts) == k:
                return True 
            window_counts[best_seller[l]] -= 1 
            if window_counts[best_seller[l]] == 0:
                del window_counts[best_seller[l]]
            l += 1 
    return False 

def max_no_bad_days(sales):
    l, r = 0, 0 
    cur_max = 0
    while r < len(sales):
        can_grow = sales[r] >= 10 
        if can_grow:
            r += 1 
            cur_max = max(cur_max, r - l)
        else:
            l = r + 1 
            r = r + 1 
    return cur_max 

def has_enduring_best_seller_streak(best_seller, k):
    l, r = 0, 0 
    cur_max = 0 
    while r < len(best_seller):
        can_grow = l == r or best_seller[l] == best_seller[r]
        if can_grow:
            r += 1 
            if r - l == k:
                return True 
        else:
            l = r 
    return False 

def max_subarray_sum(arr):
    max_val = max(arr)
    if max_val <= 0: # edge case without positive values 
        return max_val 
    l, r = 0, 0
    window_sum = 0
    cur_max = 0 
    while r < len(arr):
        can_grow = window_sum + arr[r] >= 0
        if can_grow:
            window_sum += arr[r]
            r += 1 
            cur_max = max(cur_max, window_sum)
        else:
            window_sum = 0 
            l = r + 1 
            r = r + 1 
    return cur_max 

def max_at_most_3_bad_days(sales):
    l, r = 0, 0 
    window_bad_days = 0 
    cur_max = 0
    while r < len(sales):
        can_grow = sales[r] >= 10 or window_bad_days < 3 
        if can_grow:
            if sales[r] < 10:
                window_bad_days += 1 
            r += 1 
            cur_max = max(cur_max, r - l)
        else:
            if sales[l] < 10:
                window_bad_days -= 1 
            l += 1 
    return cur_max 

def max_consecutive_with_k_boosts(projected_sales, k):
    l, r = 0, 0 
    used_boosts = 0 
    cur_max = 0
    while r < len(projected_sales):
        can_grow = used_boosts + max(10 - projected_sales[r], 0) <= k
        if can_grow:
            used_boosts += max(10 - projected_sales[r], 0)
            r += 1 
            cur_max = max(cur_max, r - l)
        elif l == r:
            r += 1 
            l += 1 
        else:
            used_boosts -= max(10 - projected_sales[l], 0)
            l += 1
    return cur_max 

def max_at_most_k_distinct(best_seller, k):
    l, r = 0, 0 
    window_counts = {}
    cur_max = 0
    while r < len(best_seller):
        can_grow = best_seller[r] in window_counts or len(window_counts) + 1 <= k
        if can_grow:
            if not best_seller[r] in window_counts:
                window_counts[best_seller[r]] = 0 
            window_counts[best_seller[r]] += 1 
            r += 1 
            cur_max = max(cur_max, r - l)
        else:
            window_counts[best_seller[l]] -= 1 
            if window_counts[best_seller[l]] == 0:
                del window_counts[best_seller[l]]
            l += 1 
    return cur_max 

def shortest_over_20_sales(sales):
    l, r = 0, 0
    window_sum = 0
    cur_min = math.inf 
    while True:
        must_grow = window_sum <= 20 
        if must_grow:
            if r == len(sales):
                break 
            window_sum += sales[r]
            r += 1 
        else:
            cur_min = min(cur_min, r - l)
            window_sum -= sales[l]
            l += 1 
    if cur_min == math.inf:
        return -1 
    return cur_min 

def shortest_with_all_letters(s1, s2):
    l, r = 0, 0 
    missing = {}
    for c in s2:
        if not c in missing:
            missing[c] = 0
        missing[c] += 1 
    distinct_missing = len(missing)
    cur_min = math.inf 
    while True:
        must_grow = distinct_missing > 0 
        if must_grow:
            if r == len(s1):
                break 
            if s1[r] in missing:
                missing[s1[r]] -= 1
                if missing[s1[r]] == 0:
                    distinct_missing -= 1
            r += 1
        else:
            cur_min = min(cur_min, r - l)
            if s1[l] in missing:
                missing[s1[l]] += 1 
                if missing[s1[l]] == 1:
                    distinct_missing += 1 
            l += 1 
    return cur_min if cur_min != math.inf else -1 

def smallest_range_with_k_elements(arr, k):
    arr.sort()
    l, r = 0, 0 
    best_low, best_high = 0, math.inf 
    while True:
        must_grow = (r - l) < k 
        if must_grow:
            if r == len(arr):
                break 
            r += 1 
        else:
            if arr[r - 1] - arr[l] < best_high - best_low:
                best_low, best_high = arr[l], arr[r - 1]
            l += 1 
    return [best_low, best_high]

def count_at_most_k_bad_days(sales, k):
    l, r = 0, 0 
    window_bad_days = 0
    count = 0 
    while r < len(sales):
        can_grow = sales[r] >= 10 or window_bad_days < k
        if can_grow:
            if sales[r] < 10:
                window_bad_days += 1 
            r += 1 
            count += r - l 
        else:
            if sales[l] < 10:
                window_bad_days -= 1 
            l += 1 
    return count 

def count_exactly_k_bad_days(sales, k):
    if k == 0:
        return count_at_most_k_bad_days(sales, 0)
    return count_at_most_k_bad_days(sales, k) - count_at_most_k_bad_days(sales, k - 1)

def count_at_least_k_bad_days(sales, k):
    n = len(sales)
    total_subarrays = n * (n + 1) // 2 
    if k == 0:
        return total_subarrays
    return total_subarrays - count_at_most_k_bad_days(sales, k - 1)

def count_at_most_k_drops(arr, k):
    l, r = 0, 0
    window_drops = 0 
    count = 0 
    while r < len(arr):
        can_grow = r == 0 or arr[r] >= arr[r - 1] or window_drops < k 
        if can_grow:
            if r > 0 and arr[r] < arr[r - 1]:
                window_drops += 1 
            r += 1 
            count += r - l 
        else:
            if arr[l] > arr[l + 1]:
                window_drops -= 1
            l += 1 
    return count 
def count_exactly_k_drops(arr, k):
    if k == 0:
        return count_at_least_k_drops(arr, 0)
    return count_at_most_k_drops(arr, k) - count_at_most_k_drops(arr, k - 1)
def count_at_least_k_drops(arr, k):
    n = len(arr)
    total_count = n * (n + 1) // 2 
    if k == 0:
        return total_count
    return total_count - count_at_most_k_drops(arr, k - 1)

def count_bad_days_range(sales, k1, k2):
    if k1 == 0:
        return count_at_least_k_bad_days(sales, k2)
    return count_at_least_k_bad_days(sales, k2) - count_at_least_k_bad_days(sales, k1 - 1)

def count_all_3_groups(arr):
    n = len(arr)
    total_count = n * (n + 1) // 2 
    return total_count - count_at_most_2_groups(arr)
def count_at_most_2_groups(arr):
    l, r = 0, 0 
    window_counts = {}
    count = 0 
    while r < len(arr):
        can_grow = arr[r] % 3 in window_counts or len(window_counts) < 2 
        if can_grow:
            if not arr[r] % 3 in window_counts:
                window_counts[arr[r] % 3] = 0 
            window_counts[arr[r] % 3] += 1 
            r += 1 
            count += r - l 
        else:
            window_counts[arr[l] % 3] -= 1 
            if window_counts[arr[l] % 3] == 0:
                del window_counts[arr[l] % 3]
            l += 1 
    return count 
```

<!-- TOC --><a name="39-backtracking"></a>
## 39. Backtracking

```py
def max_sum_path(grid):
    # inefficient backtracking solution, DP is better
    max_sum = -math.inf 
    R, C = len(grid), len(grid[0])
    def visit(r, c, cur_sum):
        nonlocal max_sum
        if r == R - 1 and c == C - 1:
            max_sum = max(max_sum, cur_sum)
            return 
        if r + 1 < R:
            visit(r + 1, c, cur_sum + grid[r + 1][c]) # go down 
        if c + 1 < C:
            visit(r, c + 1, cur_sum + grid[r][c + 1]) # go right 
    visit(0, 0, grid[0][0])
    return max_sum 

# backtracking
def visit(partial_solution):
    if full_solution(partial_solution):
        # process leaf/full solution 
    else:
        for choice in choices(partial_solution):
            # prune children where possible 
            child = apply_choice(partial_solution)
            visit(child)
visit(empty_solution)

def all_subsets(S):
    res = [] # gloabl list of subsets 
    subset = [] # state of current partial solution 
    def visit(i):
        if i == len(S):
            res.append(subset.copy())
            return 
        # choice 1: pick S[i]
        subset.append(S[i])
        visit(i + 1)
        subset.pop() # cleanup work, undo choice 1 
        # choice 2: skip S[i]
        visit(i + 1)
    visit(0)
    return res 

def generate_permutation(arr):
    res = []
    perm = arr.copy()
    def visit(i):
        if i == len(perm) - 1:
            res.append(perm.copy())
            return 
        for j in range(i, len(perm)):
            perm[i], perm[j] = perm[j], perm[i] # pick perm[j]
            visit(i + 1)
            perm[i], perm[j] = perm[j], perm[i] # cleanup work, undo change 
    visit(0)
    return res 

def generate_sentences(sentence, synonyms):
    words = sentence.split()
    res = []
    cur_sentence = []
    def visit(i):
        if i == len(words):
            res.append(" ".join(cur_sentence))
            return 
        if words[i] not in synonyms:
            choices = [words[i]]
        else:
            choices = synonyms.get(words[i])
        for choice in choices:
            cur_sentence.append(choice)
            visit(i + 1)
            cur_sentence.pop() # undo change 
    visit(0)
    return res 

def jumping_numbers(n):
    res = []
    def visit(num):
        if num >= n:
            return 
        res.append(num)
        last_digit = num % 10 
        if last_digit > 0:
            visit(num * 10 + (last_digit - 1))
        if last_digit < 9:
            visit(num * 10 + (last_digit + 1))
    for num in range(1, 10):
        visit(num)
    return sorted(res)

def maximize_style(budget, prices, ratings):
    best_rating_sum = 0
    best_items = []
    n = len(prices)
    items = []
    def visit(i, cur_cost, cur_rating_sum):
        nonlocal best_items, best_rating_sum
        if i == n:
            if cur_rating_sum > best_rating_sum:
                best_rating_sum = cur_rating_sum
                best_items = items.copy()
                return 
            # choice 1: skip item i 
            visit(i + 1, cur_cost, cur_rating_sum)
            # choice 2: pick item i (if within budget)
            if cur_cost + prices[i] <= budget:
                items.append(i)
                visit(i + 1, cur_cost + prices[i], cur_rating_sum + ratings[i])
                items.pop()
    visit(0, 0, 0)
    return best_items
```

<!-- TOC --><a name="40-dynamic-programming"></a>
## 40. Dynamic Programming

```py
def delay(times): 
    n = len(times)
    if n < 3:
        return 0
    memo = {}
    def delay_rec(i):
        if i >= n - 3:
            return times[i]
        if i in memo:
            return memo[i]
        memo[i] = times[i] + min(delay_rec(i + 1), delay_rec(i + 2), delay_rec(i + 3))
        return memo[i]
    return min(delay_rec[0], delay_rec(1), delay_rec(2))

# memoization
# memo = empty map 
# f(subproblem_id):
#     if subproblem is base case:
#         return result direcly 
#     if subproblem in memo map:
#         return cached result 
#     memo[subproblem_id] = recurrence relation formula 
#     return memo[subproblem_id]
# return f(initial subproblem)

def max_path(grid):
    R, C = len(grid), len(grid[0])
    memo = {}
    def max_path_rec(r, c):
        if r == R - 1 and c == C - 1:
            return grid[r][c]
        if (r, c) in memo:
            return memo[(r, c)]
        elif r == R - 1:
            memo[(r, c)] = grid[r][c] + max_path_rec(r, c + 1)
        elif c == C - 1:
            memo[(r, c)] = grid[r][c] + max_path_recI(r + 1, c)
        else:
            memo[(r, c)] = grid[r][c] + max(max_path_rec(r + 1, c), max_path_rec(r, c + 1))
        return memo[(r, c)]
    return max_path_rec(0, 0)

def min_split(arr, k):
    n = len(arr)
    memo = {}
    def min_split_rec(i, x):
        if (i, x) in memo:
            return memo[(i, x)]
        # base case 
        if n - i == x: # put each element in its own subarray 
            memo[(i, x)] = max(arr[i:])
        elif x == 1: # put all elements in one subarray 
            memo[(i, x)] = sum(arr[i:])
        else: # general case 
            current_sum = 0 
            res = math.inf 
            for p in range(i, n - x + 1):
                current_sum += arr[p]
                res = min(res, max(current_sum, min_split_rec(p + 1, x - 1)))
            memo[(i, x)] = res 
        return memo[(i, x)]
    return min_split_rec(0, k)

def num_ways():
    memo = {}
    def num_ways_rec(i):
        if i > 21:
            return 1 
        if 16 <= i <= 21:
            return 0 
        if i in memo:
            return memo[i]
        memo[i] = 0 
        for card in range(1, 11):
            memo[i] += num_ways_rec(i + card)
        return memo[i]
    return num_ways_rec(0)

def lcs(s1, s2):
    memo = {}
    def lcs_rec(i1, i2):
        if i1 == len(s1) or i2 == len(s2):
            return 0 
        if (i1, i2) in memo:
            return memo[(i1, i2)]
        if s1[i1] == s2[i2]:
            memo[(i1, i2)] = 1 + lcs_rec(i1 + 1, i2 + 1)
        else:
            memo[(i1, i2)] = max(lcs_rec(i1 + 1, i2), lcs_rec(i1, i2 + 1))
        return memo[(i1, i2)]
    return lcs_rec(0, 0)

def lcs_reconstruction(s1, s2):
    memo = {}
    def lcs_res(i1, i2):
        if i1 == len(s1) or i2 == len(s2):
            return ""
        if (i1, i2) in memo:
            return memo[(i1, i2)]
        if s1[i1] == s2[i2]:
            memo[(i1, i2)] = s1[i1] + lcs_res(i1 + 1, i2 + 1)
        else:
            opt1, opt2 = lcs_rec(i1 + 1, i2), lcs_res(i1, i2 + 1)
            if len(opt1) >= len(opt2):
                memo[(i1, i2)] = opt1
            else:
                memo[(i1, i2)] = opt2 
        return memo[(i1, i2)]
    return lcs_res(0, 0)

def lcs_reconstruction_optimal(s1, s2):
    memo = {}
    def lcs_rec(s1, s2):
        # same as before
    i1, i2 = 0, 0 
    res = []
    while i1 < len(s1) and i2 < len(s2):
        if s1[i1] == s2[i2]:
            res.append(s1[i1])
            i1 += 1 
            i2 += 1 
        elif lcs_rec(i1 + 1, i2) > lcs_rec(i1, i2 + 1):
            i1 += 1 
        else:
            i2 += 1 
    return ''.join(res)

def delay(times):
    n = len(times)
    if n < 3:
        return 0 
    dp = [0] * n 
    dp[n - 1], dp[n - 2], dp[n - 3] = times[n - 1], times[n - 2], times[n - 3]
    for i in range(n - 4, -1, -1):
        dp[i] = times[i] + min(dp[i + 1], dp[i + 2], dp[i + 3])
    return min(dp[0], dp[1], dp[2])

def delay_optimized(times):
    n = len(times)
    if n < 3: 
        return 0 
    dp1, dp2, dp3 = times[n - 3], times[n - 2], times[n - 1]
    for i in range(n - 4, -1, -1):
        cur = times[i] + min(dp1, dp2, dp3)
        dp1, dp2, dp3 = cur, dp1, dp2
    return min(dp1, dp2, dp3)
```

<!-- TOC --><a name="41-greedy-algorithms"></a>
## 41. Greedy Algorithms

```py
def most_non_overlapping_intervals(intervals):
    intervals.sort(key=lambda x: x[1])
    count = 0 
    prev_end = -math.inf
    for l, r in intervals:
        if l > prev_end:
            count += 1 
            prev_end = r 
    return count 

def can_reach_goal(jumping_points, k, max_aging):
    n = len(jumping_points)
    gaps = []
    for i in range(1, n):
        gaps.append(jumping_points[i] - jumping_points[i - 1])
    gaps.sort()
    total_aging = sum(gaps[:n - 1 - k])
    return total_aging <= max_aging

def minimize_distance(points, center1, center2):
    n = len(points)
    assignment = [0] * n 
    baseline = 0 
    c1_count = 0
    for i, p in enumerate(points):
        if dist(p, center1) <= dist(p, center2):
            assignment[i] = 1 
            baseline += dist(p, center1)
            c1_count += 1 
        else:
            assignment[i] = 2 
            baseline += dist(p, center2)
    if c1_count == n // 2:
        return baseline
    switch_costs = []
    for i, p in enumerate(points):
        if assignment[i] == 1 and c1_count > n // 2:
            switch_costs.append(dist(p, center2) - dist(p, center1))
        if assignment[i] == 2 and c1_count < n // 2:
            switch_costs.append(dist(p, center1) - dist(p, center2))
    res = baseline 
    switch_costs.sort()
    for cost in switch_costs[:abs(c1_count - n // 2)]:
        res += cost 
    return res 

def min_middle_sum(arr):
    arr.sort()
    middle_sum = 0 
    for i in range(len(arr) // 3):
        middle_sum += arr[i * 2 + 1]
    return middle_sum

def min_script_runs(meetings):
    meetings.sort(key=lambda x: x[1])
    count = 0 
    prev_end = -math.inf
    for l, r in meetings:
        if l > prev_end:
            count += 1 
            prev_end = r 
    return count 

def latest_reachable_year(jumping_points, k, max_aging):
    gaps = []
    for i in range(1, len(jumping_points)):
        gaps.append(jumping_points[i] - jumping_points[i - 1])
    min_heap = Heap()
    total_gap_sum = 0 
    sum_heap = 0 
    for i, gap in enumerate(gaps):
        aged = total_gap_sum - sum_heap 
        min_heap.push(gap)
        sum_heap += gap 
        total_gap_sum += gap 
        if min_heap.size() > k:
            smallest_jump = min_heap.pop()
            sum_heap -= smallest_jump 
        new_aged = total_gap_sum - sum_heap
        if new_aged > max_aging:
            # we can't reach the end of gap i
            # we get to jumping_points[i] and age naturally from there 
            remaining_aging = max_aging - aged 
            return jumping_points[i] + remaining_aging
    # reached last jumping point 
    aged = total_gap_sum - sum_heap 
    remaining_aging = max_aging - aged 
    return jumping_points[len(jumping_points) - 1] + remaining_aging
```

<!-- TOC --><a name="42-topological-sort"></a>
## 42. Topological Sort 

```py
def topological_sort(graph):
    # initialization
    V = len(graph)
    in_degrees = [0 for _ in range(V)]
    for node in range(V):
        for nbr in graph[node]: # for weighted graphs, unpack edges: nbr, _ 
            in_degrees[nbr] += 1 
    degree_zero = []
    for node in range(V):
        if in_degrees[node] == 0:
            degree_zero.append(node)
    # main peel-off loop 
    topo_order = []
    while degree_zero:
        node = degree_zero.pop()
        topo_order.append(node)
        for nbr in graph[node]: # for weighted graphs, unpack edges: nbr, _ 
            in_degrees[nbr] -= 1 
            if in_degrees[nbr] == 0:
                degree_zero.append(nbr)
    if len(topo_order) < V:
        return [] # there'a a cycle, some nodes couldn't be peeled off 
    return topo_order 

def distance(graph, start):
    topo_order = topological_sort(graph) 
    distances = {start: 0}
    for node in topo_order:
        if node not in distance: continue 
        for nbr, weight in graph[node]:
            if nbr not in distances or distances[node] + weight < distances[nbr]:
                distances[nbr] = distances[node] + weight 
    res = []
    for i in range(len(graph)):
        if i in distances:
            res.append(distances[i])
        else:
            res.append(math.inf)
    return res 

def shortest_path(graph, start, goal):
    topo_order = topological_sort(graph)
    distances = {start: 0}
    predecessors = {}
    for node in topo_order:
        if node not in distances: continue 
        for nbr, weight in graph[node]:
            if nbr not in distances or distances[node] + weight < distance[nbr]:
                distances[nbr] = distances[node] + weight 
                predecessors[nbr] = node 
    if goal not in distances:
        return []
    path = [goal]
    while path[-1] != start:
        path.append(predecessors[path[-1]])
    path.reverse()
    return path 

def path_count(graph, start):
    topo_order = topological_sort(graph)
    counts = [0] * len(graph)
    counts[start] = 1 
    for node in topo_order:
        for nbr in graph[node]:
            counts[nbr] += counts[node]
    return counts 

def compile_time(seconds, imports):
    V = len(seconds)
    graph = [[] for _ in range(V)]
    for package in range(V):
        for imported_package in imports[package]:
            graph[imported_package].append(package)
    topo_order = topological_sort(graph)
    durations = {}
    for node in topo_order:
        if node not in durations:
            durations[node] = seconds[node]
        for nbr in graph[node]:
            if nbr not in durations:
                durations[nbr] = 0 
            durations[nbr] = max(durations[nbr], seconds[nbr] + durations[node])
    return max(durations.values())
```

<!-- TOC --><a name="43-prefix-sums"></a>
## 43. Prefix Sums 

```py
def channel_views(views, periods):
    prefix_sum = [0] * len(views)
    prefix_sum[0] = views[0]
    for i in range(1, len(views)):
        prefix_sum[i] = prefix_sum[i - 1] + views[i]
    res = []
    for l, r in periods:
        if l == 0:
            res.append(prefix_sum[r])
        else:
            res.append(prefix_sum[r] - prefix_sum[l - 1])
    return res 

# # initialization
# # initialize prefix_sum with the same length as input array 
# prefix_sum[0] = arr[0] # at least one element 
# for i from 1 to len(arr) - 1:
#     prefix_sum[i] = prefix_sum[i - 1] + arr[i]
# # query: sum of subarray [l, r]
# if l == 0:
#     return prefix_sum[r]
# return prefix_sum[r] - prefix_sum[l - 1]

def good_reception_scores(likes, dislikes, periods):
    positive_days = [0] * len(likes)
    for i in range(likes):
        if likes[i] > dislikes[i]:
            positive_days[i] = 1 
    # build prefix sum for positive_days array and query it with each period 

def exclusive_product_array(arr):
    m = 10 ** 9 + 7 
    n = len(arr)
    prefix_product = [1] * n 
    prefix_product[0] = arr[0]
    for i in range(1, n):
        prefix_product[i] = (prefix_product[i - 1] * arr[i]) % m 
    postfix_product = [1] * n 
    postfix_product[n - 1] = arr[n - 1]
    for i in range(n - 2, -1, -1):
        postfix_product[i] = (postfix_product[i + 1] * arr[i]) % m 
    res = [1] * n 
    res[0] = postfix_product[1]
    res[n - 1] = prefix_product[n - 2]
    for i in range(1, n - 1):
        res[i] = (prefix_product[i - 1] * postfix_product[i + 1]) % m 
    return res 

def balanced_index(arr):
    prefix_sum = 0 
    postfix_sum = sum(arr) - arr[0]
    for i in range(len(arr)):
        if prefix_sum == postfix_sum:
            return i 
        prefix_sum += arr[i]
        if i + 1 < len(arr):
            postfix_sum -= arr[i + 1]
    return -1 

def max_total_deviation(likes, dislikes):
    scores = [likes[i] - dislikes[i] for i in range(len(likes))]
    scores.sort()
    n = len(scores)
    prefix_sum = [0] * n 
    prefix_sum[0] = scores[0]
    for i in range(1, n):
        prefix_sum[i] = prefix_sum[i - 1] + scores[i]
    max_deviation = 0
    for i in range(n):
        left, right = 0, 0 
        if i > 0:
            left = i * scores[i] - prefix_sum[i - 1]
        if i < n - 1:
            right = prefix_sum[n - 1] - prefix_sum[i] - (n - i - 1) * scores[i]
        max_deviation = max(max_deviation, left + right)
    return max_deviation

def count_subarrays(arr, k):
    prefix_sum = # ...
    prefix_sum_to_count = {0: 1} # for empty prefix 
    count = 0 
    for val in prefix_sum:
        if val - k in prefix_sum_to_count:
            count += prefix_sum_to_count[val - k]
        if val not in prefix_sum_to_count:
            prefix_sum_to_count[val] = 0 
        prefix_sum_to_count[val] += 1 
    return count 

def longest_subarray_with_sum_k(arr, k):
    prefix_sum = # ...
    prefix_sum_to_index = {0: -1} # for empty prefix 
    res = -1 
    for r, val in enumerate(prefix_sum):
        if val - k in prefix_sum_to_index:
            l = prefix_sum_to_index[val - k]
            res = max(res, r - l)
        if val not in prefix_sum_to_index:
            prefix_sum_to_index[val] = r 
    return res 

def range_updates(n, votes):
    diff = [0] * n 
    for l, r, v in votes:
        diff[l] += v 
        if r + 1 < n:
            diff[r + 1] -= v 
    prefix_sum = [0] * n 
    prefix_sum[0] = diff[0]
    for i in range(1, n):
        prefix_sum[i] = prefix_sum[i - 1] + diff[i]
    return prefix_sum

def most_booked_slot(slots, bookings):
    n = len(slots)
    diff = [0] * n 
    for l, r, c in bookings:
        diff[l] += c 
        if r + 1 < n:
            diff[r + 1] -= c 
    prefix_sum = [0] * n 
    prefix_sum[0] = diff[0]
    for i in range(1, n):
        prefix_sum[i] = prefix_sum[i - 1] + diff[i]
    max_bookings, max_index = 0, -1 
    for i in range(n):
        total_bookings = prefix_sum[i] + slots[i]
        if total_bookings > max_bookings:
            max_bookings, max_index = total_bookings, i 
    return max_index
```

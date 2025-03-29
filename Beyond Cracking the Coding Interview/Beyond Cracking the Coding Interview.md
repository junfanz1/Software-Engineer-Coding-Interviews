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

```

<!-- TOC --><a name="31-sorting"></a>
## 31. Sorting

```py

```

<!-- TOC --><a name="32-stacks-queues"></a>
## 32. Stacks & Queues

```py

```

<!-- TOC --><a name="33-recursion"></a>
## 33. Recursion

```py

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

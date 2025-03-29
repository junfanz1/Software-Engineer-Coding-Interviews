Beyond Cracking the Coding Interview - by Gayle Laakmann McDowell, Mike Mroczka, Aline Lerner, Nil Mamano, 2025

> [Book Link](https://www.amazon.com/Beyond-Cracking-Coding-Interview-Successfully/dp/195570600X)

 <img src="https://github.com/user-attachments/assets/62a0e128-7ca0-49b4-a53f-fed51456f06b" width="35%" height="35%">

# Contents

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [20. Anatomy of Coding Interview](#20-anatomy-of-coding-interview)
- [25. Dynamic Arrays](#25-dynamic-arrays)

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

# 28. Grids & Matrices

```py

```

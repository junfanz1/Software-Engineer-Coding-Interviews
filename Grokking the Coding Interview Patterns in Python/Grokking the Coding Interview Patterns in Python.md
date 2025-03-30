> [Educative - Grokking the Coding Interview Patterns in Python](https://www.educative.io/courses/grokking-coding-interview-in-python)

# Contents

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

   * [1. Two Pointers](#1-two-pointers)
   * [2. Fast Slow Pointers](#2-fast-slow-pointers)
   * [3. Sliding Window ](#3-sliding-window)
   * [4. Merge Intervals ](#4-merge-intervals)
   * [5. Manipulation of Linked List ](#5-manipulation-of-linked-list)
   * [6. Heaps](#6-heaps)
   * [7. K-way merge ](#7-k-way-merge)
   * [8. Top K Elements](#8-top-k-elements)
   * [9. Modified Binary Search](#9-modified-binary-search)
   * [10. Subsets](#10-subsets)
   * [11. Greedy Algorithm](#11-greedy-algorithm)
   * [12. Backtracking](#12-backtracking)
   * [13. Dynamic Programming](#13-dynamic-programming)
   * [14. Cyclic Sort ](#14-cyclic-sort)
   * [15. Topological Sort ](#15-topological-sort)
   * [16. Sort and Search](#16-sort-and-search)
   * [17. Matrices](#17-matrices)
   * [18. Stacks](#18-stacks)
   * [19. Graphs](#19-graphs)
   * [20. Tree DFS](#20-tree-dfs)
   * [21. Tree BFS ](#21-tree-bfs)
   * [22. Trie](#22-trie)
   * [23. Hash Maps ](#23-hash-maps)
   * [24. Knowing what to Track](#24-knowing-what-to-track)
   * [25. Union Find ](#25-union-find)
   * [26. Custom Data Structures ](#26-custom-data-structures)
   * [27. Bitwise Manipulation](#27-bitwise-manipulation)
   * [28. Math and Geometry](#28-math-and-geometry)
   * [29. Challenges](#29-challenges)

<!-- TOC end -->

<!-- TOC --><a name="1-two-pointers"></a>
## 1. Two Pointers

```py
def is_palindrome(s):
    left = 0 
    right = len(s) - 1 
    while left < right: 
        if s[left] != s[right]:
            return False 
        left += 1 
        right -= 1 
    return True 

def three_sum(nums):
    nums.sort()
    result = []
    n = len(nums)
    for pivot in range(n - 2):
        # if current number > 0, break the loop as no valid triplets possible
        if nums[pivot] > 0:
            break 
        # skip duplciate values for pivot 
        if pivot > 0 and nums[pivot] == nums[pivot - 1]:
            continue 
        # two pointer 
        low, high = pivot + 1, n - 1 
        while low < high:
            total = nums[pivot] + nums[low] + nums[high]
            if total < 0:
                low += 1 
            elif total > 0:
                high -= 1
            else:
                # find triplet 
                result.append([nums[pivot], nums[low], nums[high]])
                low += 1 
                high -= 1 
                # skip duplicates for low and high pointers 
                while low < high and nums[low] == nums[low - 1]:
                    low += 1
                while low < high and nums[high] == nums[high + 1]:
                    high -= 1 
    return result

def remove_nth_last_node(head, n):
    right = head 
    left = head 
    # move right pointer n elements away from left pointer
    for i in range(n):
        right = right.next 
    # remove head node
    if not right:
        return head.next 
    # move both pointers until right pointer reaches last node
    while right.next:
        right = right.next 
        left = left.next 
    # left pointer now at n-1 th element, link it to next next element
    left.next = left.next.next
    return head 

def sort_colors(colors):
    start, current, end = 0, 0, len(colors) - 1 
    # iterate through list until current pointer exceeds end pointer 
    while current <= end:
        if colors[current] == 0:
            colors[start], colors[current] = colors[current], colors[start]
            # move both start and current pointers forward
            current += 1 
            start += 1 
        elif colors[current] == 1:
            current += 1 
        else:
            colors[current], colors[end] = colors[end], colors[current]
            end -= 1 
    return colors 

def reverse_words(sentence):
    sentence = sentence.strip()
    result = sentence.split()
    left, right = 0, len(result) - 1 
    while left <= right:
        result[left], result[right] = result[right], result[left]
        left += 1 
        right -= 1 
    return " ".join(result)

def valid_word_abbreviation(word, abbr):
    word_index, abbr_index = 0, 0
    while abbr_index < len(abbr):
        if abbr[abbr_index].isdigit():
            # check if there's leading zero 
            if abbr[abbr_index] == '0':
                return False 
            num = 0 
            while abbr_index < len(abbr) and abbr[abbr_index].isdigit():
                num = num * 10 + int(abbr[abbr_index])
                abbr_index += 1 
            # skip the number of characters in word as found in abbreviation
            word_index += num
        else:
            # check if characters the match, then increment pointers, otherwise return False 
            if word_index >= len(word) or word[word_index] != abbr[abbr_index]:
                return False 
            word_index += 1 
            abbr_index += 1 
    # both indices have reached the end of their strings
    return word_index == len(word) and abbr_index == len(abbr)

def is_strobogrammatic(num):
    dict = {'0':'0', '1':'1', '8':'8', '6':'9', '9':'6'}
    left = 0 
    right = len(num) - 1 
    while left <= right:
        # if current digit is valid and matches corresponding rotated value 
        if num[left] not in dict or dict[num[left]] != num[right]:
            return False 
        left += 1 
        right -= 1 
    return True # if all digit pairs are valid 

def min_moves_to_make_palindrome(s):
    s = list(s) # string to list 
    moves = 0 
    i, j = 0, len(s) - 1 
    while i < j:
        k = j 
        while k > i:
            # if matching found
            if s[i] == s[k]:
                # move matching character to correct position on the right
                for m in range(k, j):
                    s[m], s[m + 1] = s[m + 1], s[m]
                    moves += 1 
                j -= 1 
                break
            k -= 1 
        # if no matching character found, move to center of palindrome
        if k == i:
            moves += len(s) // 2 - i 
        i += 1 
    return moves 

 def find_next_permutation(digits):
    # find first digit smaller than digit after it 
    i = len(digits) - 2 
    while i >= 0 and digits[i] >= digits[i + 1]:
        i -= 1 
    if i == -1:
        return False 
    # find next largest digit to swap with digits[i]
    j = len(digits) - 1 
    while digits[j] <= digits[i]:
        j -= 1 
    # swap and reverse rest to get smallest next permutation 
    digits[i], digits[j] = digits[j], digits[i]
    digits[i + 1:] = reversed(digits[i + 1:])
    return True 
def find_next_palindrome(num_str):
    n = len(num_str)
    if n == 1:
        return ""
    half_length = n // 2 
    left_half = list(num_str[:half_length])
    if not find_next_permutation(left_half):
        return ""
    if n % 2 == 0:
        next_palindrome = ''.join(left_half + left_half[::-1])
    else:
        middle_digit = num_str[half_length]
        next_palindrome = ''.join(left_half + [middle_digit] + left_half[::-1])
    if next_palindrome > num_str:
        return next_palindrome
    return ""

def lowest_common_ancestor(p, q):
    ptr1, ptr2 = p, q 
    while ptr1 != ptr2:
        # Move ptr1 to parent node or switch to the other node if reached the root
        if ptr1.parent:
            ptr1 = ptr1.parent 
        else:
            ptr1 = q 
        if ptr2.parent:
            ptr2 = ptr2.parent 
        else:
            ptr2 = p 
    # ptr1 ptr2 are the same at this point
    return ptr1 

def count_pairs(nums, target):
    nums.sort()
    count = 0 
    low, high = 0, len(nums) - 1
    while low < high:
        if nums[low] + nums[high] < target:
            count += high - low 
            low += 1 
        else:
            high -= 1 
    return count 
```

<!-- TOC --><a name="2-fast-slow-pointers"></a>
## 2. Fast Slow Pointers

```py

```

<!-- TOC --><a name="3-sliding-window"></a>
## 3. Sliding Window 

```py

```

<!-- TOC --><a name="4-merge-intervals"></a>
## 4. Merge Intervals 

```py

```

<!-- TOC --><a name="5-manipulation-of-linked-list"></a>
## 5. Manipulation of Linked List 

```py

```

<!-- TOC --><a name="6-heaps"></a>
## 6. Heaps

```py

```

<!-- TOC --><a name="7-k-way-merge"></a>
## 7. K-way merge 

```py

```

<!-- TOC --><a name="8-top-k-elements"></a>
## 8. Top K Elements

```py

```

<!-- TOC --><a name="9-modified-binary-search"></a>
## 9. Modified Binary Search

```py

```

<!-- TOC --><a name="10-subsets"></a>
## 10. Subsets

```py

```

<!-- TOC --><a name="11-greedy-algorithm"></a>
## 11. Greedy Algorithm

```py

```

<!-- TOC --><a name="12-backtracking"></a>
## 12. Backtracking

```py

```

<!-- TOC --><a name="13-dynamic-programming"></a>
## 13. Dynamic Programming

```py

```

<!-- TOC --><a name="14-cyclic-sort"></a>
## 14. Cyclic Sort 

```py

```

<!-- TOC --><a name="15-topological-sort"></a>
## 15. Topological Sort 

```py

```

<!-- TOC --><a name="16-sort-and-search"></a>
## 16. Sort and Search

```py

```

<!-- TOC --><a name="17-matrices"></a>
## 17. Matrices

```py

```

<!-- TOC --><a name="18-stacks"></a>
## 18. Stacks

```py

```

<!-- TOC --><a name="19-graphs"></a>
## 19. Graphs

```py

```

<!-- TOC --><a name="20-tree-dfs"></a>
## 20. Tree DFS

```py

```

<!-- TOC --><a name="21-tree-bfs"></a>
## 21. Tree BFS 

```py

```

<!-- TOC --><a name="22-trie"></a>
## 22. Trie

```py

```

<!-- TOC --><a name="23-hash-maps"></a>
## 23. Hash Maps 

```py

```

<!-- TOC --><a name="24-knowing-what-to-track"></a>
## 24. Knowing what to Track

```py

```

<!-- TOC --><a name="25-union-find"></a>
## 25. Union Find 

```py

```

<!-- TOC --><a name="26-custom-data-structures"></a>
## 26. Custom Data Structures 

```py

```

<!-- TOC --><a name="27-bitwise-manipulation"></a>
## 27. Bitwise Manipulation

```py

```

<!-- TOC --><a name="28-math-and-geometry"></a>
## 28. Math and Geometry

```py

```

<!-- TOC --><a name="29-challenges"></a>
## 29. Challenges


```py

```


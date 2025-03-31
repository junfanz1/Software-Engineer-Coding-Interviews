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
def is_happy_number(n):
    def sum_of_squared_digits(number):
        total_sum = 0
        while number > 0:
            number, digit = divmod(number, 10)
            total_sum += digit ** 2 
        return total_sum
    slow_pointer = n
    fast_pointer = sum_of_squared_digits(n)
    while fast_pointer != 1 and slow_pointer != fast_pointer:
        slow_pointer = sum_of_squared_digits(slow_pointer)
        fast_pointer = sum_of_squared_digits(sum_of_squared_digits(fast_pointer))
    if (fast_pointer == 1):
        return True 
    return False 

def detect_cycle(head):
    if head is None:
        return False 
    slow, fast = head, head 
    # run the loop until we reach the end of linked list or find cycle 
    while fast and fast.next:
        slow = slow.next 
        fast = fast.next.next 
        if slow == fast:
            return True 
    # if reach the end not found cycle
    return False 

def get_middle_node(head):
    slow = head 
    fast = head 
    while fast and fast.next:
        slow = slow.next 
        fast = fast.next.next 
    return slow 

def circular_array_loop(nums):
    size = len(nums)
    for i in range(size):
        slow = fast = i 
        # set true in forward if element > 0, false otherwise 
        forward = nums[i] > 0 
        while True:
            # move slow pointer to one step 
            slow = next_step(slow, nums[slow], size)
            # if cycle not possible, break loop and start from next element 
            if is_not_cycle(nums, forward, slow):
                break 
            # first move of fast pointer 
            fast = next_step(fast, nums[fast], size)
            if is_not_cycle(nums, forward, fast):
                break 
            # second move of fast pointer
            fast = next_step(fast, nums[fast], size)
            if is_not_cycle(nums, forward, fast):
                break 
            if slow == fast:
                return True 
    return False 
def next_step(pointer, value, size):
    return (pointer + value) % size 
def is_not_cycle(nums, prev_direction, pointer):
    curr_direction = nums[pointer] >= 0 
    if (prev_direction != curr_direction) or (nums[pointer] % len(nums) == 0):
        return True 
    else:
        return False 
    
def find_duplicate(nums):
    fast = slow = nums[0]
    # traverse until intersection point is found 
    while True:
        slow = nums[slow]
        # move fast pointer two times fast 
        fast = nums[nums[fast]]
        if slow == fast:
            break # intersection found as two pointers meet 
    # slow pointer start at starting point 
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    return fast 

def palindrome(head):
    slow = head 
    fast = head 
    while fast and fast.next:
        slow = slow.next 
        fast = fast.next.next 
    revert_data = reverse_linked_list(slow)
    check = compare_two_halves(head, revert_data)
    reverse_linked_list(revert_data)
    if check:
        return True 
    return False 
def compare_two_halves(first_half, second_half):
    while first_half and second_half:
        if first_half.val != second_half.val:
            return False 
        else:
            first_half = first_half.next 
            second_half = second_half.next 
    return True 
def reverse_linked_list(slow_ptr):
    prev = None 
    next = None 
    curr = slow_ptr
    while curr is not None:
        next = curr.next 
        curr.next = prev 
        prev = curr 
        curr = next 
    return prev 

def twin_sum(head):
    slow, fast = head, head 
    while fast and fast.next:
        slow = slow.next 
        fast = fast.next.next 
    curr, prev = slow, None 
    while curr:
        temp = curr.next 
        curr.next = prev 
        prev = curr 
        curr = temp 
    max_sum = 0
    curr = head 
    while prev:
        max_sum = max(max_sum, curr.val + prev.val)
        prev = prev.next 
        curr = curr.next 
    return max_sum 

def split_circular_linked_list(head):
    slow = fast = head 
    while fast.next != head and fast.next.next != head:
        slow = slow.next 
        fast = fast.next.next 
    head1 = head 
    head2 = slow.next 
    slow.next = head1 
    fast = head2 
    while fast.next != head:
        fast = fast.next 
    fast.next = head2 
    return [head1, head2]

def count_cycle_length(head):
    slow = head 
    fast = head 
    while fast and fast.next:
        slow = slow.next 
        fast = fast.next.next 
        if slow == fast:
            length = 1 
            slow = slow.next 
            while slow != fast:
                length += 1 
                slow = slow.next
            return length 
    return 0 # now cycle found
```

<!-- TOC --><a name="3-sliding-window"></a>
## 3. Sliding Window 

```py
def findRepeatedDnaSequences(s):
    to_int = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded_sequence = [to_int[c] for c in s]
    k = 10 
    n = len(s)
    if n <= k:
        return []
    a = 4 # base-4 encoding
    h = 0 # hash
    seen_hashes, output = set(), set() # to track hashes and repeated sequences 
    a_k = 1 # stores a^L for efficient rolling hash updates 
    # initial hash computation for first 10-letter substring 
    for i in range(k):
        h = h * a + encoded_sequence[i]
        a_k *= a 
    seen_hashes.add(h) # store initial hash 
    # sliding window to update hash 
    for start in range(1, n - k + 1):
        # remove leftmost character and add new rightmost character
        h = h * a - encoded_sequence[start - 1] * a_k + encoded_sequence[start + k - 1]
        # if hash has been seen_hashes before, add substring to output
        if h in seen_hashes:
            output.add(s[start: start + k])
        else:
            seen_hashes.add(h)
    return list(output) # convert set to list 

from collections import deque
# clean up deque
def clean_up(i, current_window, nums):
    # remove all indexes from current_window whose corresponding values <= current element
    while current_window and nums[i] >= nums[current_window[-1]]:
        current_window.pop()
# find max in all possible windows
def find_max_sliding_window(nums, w):
    if len(nums) == 1:
        return nums 
    output = []
    current_window = deque()
    for i in range(w):
        clean_up(i, current_window, nums)
        current_window.append(i)
    output.append(nums[current_window[0]])
    for i in range(w, len(nums)):
        clean_up(i, current_window, nums)
        if current_window and current_window[0] <= (i - w):
            current_window.popleft()
        current_window.append(i)
        output.append(nums[current_window[0]])
    return output 

def min_window(str1, str2):
    size_str1, size_str2 = len(str1), len(str2)
    min_sub_len = float('inf')
    index_s1, index_s2 = 0, 0 
    min_subsequence = ""
    while index_s1 < size_str1:
        if str1[index_s1] == str2[index_s2]:
            index_s2 += 1 
            # if index_s2 has reached the end of str2
            if index_s2 == size_str2:
                start, end = index_s1, index_s1 
                index_s2 -= 1 
                # decrement pointer index_s2 and start reverse loop 
                while index_s2 >= 0:
                    if str1[start] == str2[index_s2]:
                        index_s2 -= 1
                    # decrement start pointer to find the start point of required subsequence 
                    start -= 1 
                start += 1 
                # check if min_sub_len of sub sequence pointed by start and end pointers < current min min_sub_len 
                if end - start < min_sub_len:
                    min_sub_len = end - start 
                    min_subsequence = str1[start:end+1] # update to new shorter string 
                index_s1 = start 
                index_s2 = 0 
        # increment pointer to check next character in str1
        index_s1 += 1 
    return min_subsequence

def longest_repeating_character_replacement(s, k):
    string_length = len(s)
    length_of_max_substring = 0
    start = 0 
    char_freq = {}
    most_freq_char = 0 
    for end in range(string_length):
        if s[end] not in char_freq:
            char_freq[s[end]] = 1 
        else:
            char_freq[s[end]] += 1 
        most_freq_char = max(most_freq_char, char_freq[s[end]])
        # if number of replacements in current window exceed limit, slide the window
        if end - start + 1 - most_freq_char > k:
            char_freq[s[start]] -= 1
            start += 1 
        # if window is the longest, update length of max substring 
        length_of_max_substring = max(end - start + 1, length_of_max_substring)
    return length_of_max_substring

def min_window(s: str, t: str) -> str:
    if not t:
        return ""
    req_count = {}
    window = {}
    # populate req_count with character frequencies of t
    for char in t:
        req_count[char] = req_count.get(char, 0) + 1 
    current = 0
    required = len(req_count) # total number of unique characters in t 
    # result variables to track best window
    res = [-1, -1] # start, end indices of min window 
    res_len = float("inf") # length of min window 
    left = 0 
    for right in range(len(s)):
        char = s[right]
        # if char in t, update window count 
        if char in req_count:
            window[char] = window.get(char, 0) + 1 
            # if freq of char in window match required frequency, update current
            if window[char] == req_count[char]:
                current += 1 
        # contract window while all required chars are present 
        while current == required:
            # update result if current window < previous best
            if (right - left + 1) < res_len:
                res = [left, right]
                res_len = (right - left + 1)
            left_char = s[left]
            if left_char in req_count:
                window[left_char] -= 1 
                # if frequency of left_char in window < required, update frequency 
                if window[left_char] < req_count[left_char]:
                    current -= 1 
            left += 1 # move left pointer to shrink window 
    # return min window if found, otherwise empty string
    return s[res[0]:res[1] + 1] if res_len != float("inf") else ""

def find_longest_substring(input_str):
    if len(input_str) == 0:
        return 0 
    window_start, longest, window_length = 0, 0, 0
    last_seen_at = {}
    for index, val in enumerate(input_str):
        # if current element not in hash map, store it 
        if val not in last_seen_at:
            last_seen_at[val] = index 
        else:
            # element have appeared before, check if it occurs before or after window_start
            if last_seen_at[val] >= window_start:
                window_length = index - window_start
                if longest < window_length:
                    longest = window_length 
                window_start = last_seen_at[val] + 1 
            # update last occurrence of element in hash map 
            last_seen_at[val] = index 
    index += 1
    # uopdate longest substring's length and start
    if longest < index - window_start:
        longest = index - window_start
    return longest 

def min_sub_array_len(target, nums):
    window_size = float('inf')
    start = 0
    sum = 0
    for end in range(len(nums)):
        sum += nums[end]
        # remove elements from window start while sum > target 
        while sum >= target:
            curr_subarr_size = (end + 1) - start 
            window_size = min(window_size, curr_subarr_size)
            # remove element from window start 
            sum -= nums[start]
            start += 1 
    if window_size != float('inf'):
        return window_size
    return 0 

def find_max_average(nums, k):
    current_sum = sum(nums[:k])
    max_sum = current_sum 
    for i in range(k, len(nums)):
        current_sum += nums[i] - nums[i - k]
        max_sum = max(max_sum, current_sum)
    return max_sum / k

def diet_plan_performance(calories, k, lower, upper):
    points = 0 
    # initial window of first k days
    current_sum = sum(calories[:k])
    if current_sum < lower:
        points -= 1 
    elif current_sum > upper:
        points += 1 
    # slide window across rest of days 
    for i in range(k, len(calories)):
        current_sum = current_sum - calories[i - k] + calories[i]
        if current_sum < lower:
            points -= 1 
        elif current_sum > upper:
            points += 1 
    return points 

def total_fruit(fruits):
    baskets = {}
    collected = 0
    left = 0 
    for right in range(len(fruits)):
        baskets[fruits[right]] = baskets.get(fruits[right], 0) + 1 
        while len(baskets) > 2:
            baskets[fruits[left]] -=1 
            # remove fruit type from basket if count = 0
            if baskets[fruits[left]] == 0:
                del baskets[fruits[left]]
            left += 1 
        collected = max(collected, right - left + 1)
    return collected
            
def contains_nearby_duplicate(nums, k):
    seen = set()
    for i in range(len(nums)):
        if nums[i] in seen:
            return True # duplicate found within range
        seen.add(nums[i])
        # maintain sliding window size 
        if len(seen) > k:
            seen.remove(nums[i - k]) # remove oldest element outside range 
    return False # no duplicate found 

def max_frequency(nums, k):
    nums.sort()
    left = 0 
    max_freq = 0 
    window_sum = 0
    for right in range(len(nums)):
        target = nums[right] 
        window_sum += target 
        # check if total required increments > k
        while (right - left + 1) * target > window_sum + k:
            window_sum -= nums[left] # remove leftmost element 
            left += 1 # shrink window
        max_freq = max(max_freq, right - left + 1)
    return max_freq
```

<!-- TOC --><a name="4-merge-intervals"></a>
## 4. Merge Intervals 

```py
def merge_intervals(intervals):
    if not intervals:
        return None 
    result = []
    result.append([intervals[0][0], intervals[0][1]])
    for i in range(1, len(intervals)):
        last_added_interval = result[len(result) - 1]
        cur_start = intervals[i][0]
        cur_end = intervals[i][1]
        prev_end = last_added_interval[1]
        if cur_start <= prev_end:
            result[-1][-1] = max(cur_end, prev_end)
        else:
            result.append([cur_start, cur_end])
    return result 

def insert_interval(existing_intervals, new_interval):
    new_start, new_end = new_interval[0], new_interval[1]
    i = 0 
    n = len(existing_intervals)
    output = []
    while i < n and existing_intervals[i][0] < new_start:
        output.append(existing_intervals[i])
        i = i + 1 
    if not output or output[-1][1] < new_start:
        output.append(new_interval)
    else:
        output[-1][-1] = max(output[-1][1], new_end)
    while i < n:
        ei = existing_intervals[i]
        start, end = ei[0], ei[1]
        if output[-1][1] < start:
            output.append(ei)
        else:
            output[-1][1] = max(output[-1][1], end)
        i += 1 
    return output 

def intervals_intersection(interval_list_a, interval_list_b):
    intersections = []
    i = j = 0
    while i < len(interval_list_a) and j < len(interval_list_b):
        start = max(interval_list_a[i][0], interval_list_b[j][0])
        end = min(interval_list_a[i][1], interval_list_b[j][1])
        if start <= end: # if intersection
            intersections.append([start, end])
        if interval_list_a[i][1] < interval_list_b[j][1]:
            i += 1 
        else:
            j += 1 
    return intersections

class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.closed = True  # by default, the interval is closed
    # set the flag for closed/open
    def set_closed(self, closed):
        self.closed = closed
    def __str__(self):
        return "[" + str(self.start) + ", " + str(self.end) + "]" \
            if self.closed else \
                "(" + str(self.start) + ", " + str(self.end) + ")"
import heapq 
def employee_free_time(schedule):
    heap = []
    # iterate for all employees' schedules, add start of each schedule's first interval
    for i in range(len(schedule)):
        heap.append((schedule[i][0].start, i, 0))
    # heap from array elements 
    heapq.heapify(heap)
    result = []
    previous = schedule[heap[0][1]][heap[0][2]].start 
    # iterate till heap is empty 
    while heap:
        # pop element from heap and set value of i, j
        _, i, j = heapq.heappop(heap)
        interval = schedule[i][j]
        if interval.start > previous:
            # means interval is free, so add it 
            result.append(Interval(previous, interval.start))
        previous = max(previous, interval.end)
        # if another interval in current employee's schedule, push into heap
        if j + 1 < len(schedule[i]):
            heapq.heappush(heap, (schedule[i][j + 1].start, i, j + 1))
    # when heap empty, return result 
    return result 

def count_days(days, meetings):
    meetings.sort()
    occupied = 0 
    start, end = meetings[0]
    for i in range(1, len(meetings)):
        # if meeting overlaps with current merged meeting
        if meetings[i][0] <= end:
            end = max(end, meeting[i][1])
        else:
            occupied += (end - start + 1)
            start, end = meetings[i]
    occupied += (end - start + 1)
    return days - occupied
```

<!-- TOC --><a name="5-manipulation-of-linked-list"></a>
## 5. Manipulation of Linked List 

```py
def reverse(head):
    prev, next = None, None 
    curr = head 
    while curr is not None:
        next = curr.next 
        curr.next = prev 
        prev = curr 
        curr = next 
    head = prev 
    return head 

def reverse_linked_list(head, k):
    previous, current, next = None, head, None 
    for _ in range(k):
        next = current.next 
        current.next = previous 
        previous = current 
        current = next 
    return previous, current 
def reverse_k_groups(head, k):
    dummy = ListNode(0)
    dummy.next = head 
    ptr = dummy 
    while (ptr != None):
        tracker = ptr 
        for i in range(k):
            if tracker == None:
                break 
            tracker = tracker.next 
        if tracker == None:
            break 
        previous, current = reverse_linked_list(ptr.next, k)
        last_node_of_reversed_group = ptr.next 
        last_node_of_reversed_group.next = current 
        ptr.next = previous 
        ptr = last_node_of_reversed_group
    return dummy.next 

def reverse_between(head, left, right):
    if not head or left == right:
        return head 
    dummy = ListNode(0)
    dummy.next = head 
    prev = dummy 
    # move prev to the node before left position
    for _ in range(left - 1):
        prev = prev.next
    curr = prev.next 
    for _ in range(right - left):
        next_node = curr.next 
        curr.next = next_node.next 
        next_node.next = prev.next 
        prev.next = next_node 
    # updated head of linked list
    return dummy.next 

def reorder_list(head):
    if not head:
        return head 
    # fine middle of linked list 
    slow = fast = head 
    while fast and fast.next:
        slow = slow.next 
        fast = fast.next.next 
    # reverse second part of list, 123456 -> 123 654
    prev, curr = None, slow
    while curr:
        curr.next, prev, curr = prev, curr, curr.next 
    # merge 123 654 to 162534 
    first, second = head, prev 
    while second.next:
        first.next, first = second, first.next 
        second.next, second = first, second.next 
    return head 

def swap(node1, node2):
    temp = node1.val 
    node1.val = node2.val 
    node2.val = temp 
def swap_nodes(head, k):
    count = 0 
    front, end = None, None 
    curr = head 
    while curr:
        count += 1 
        if end is not None: # kth node has been found
            end = end.next # move end pointer to find kth node from end of linked list 
        if count == k: # curr is at kth node from start
            front = curr 
            end = head 
        curr = curr.next 
    swap(front, end)
    return head

def reverse_even_length_groups(head):
    prev = head 
    group_len = 2 
    while prev.next:
        node = prev 
        num_nodes = 0 
        for i in range(group_len):
            if not node.next:
                break 
            num_nodes += 1 
            node = node.next 
        # odd length 
        if num_nodes % 2:
            prev = node 
        # even length 
        else:
            reverse = node.next 
            curr = prev.next 
            for j in range(num_nodes):
                curr_next = curr.next 
                curr.next = reverse 
                reverse = curr 
                curr = curr_next 
            prev_next = prev.next 
            prev.next = node 
            prev = prev_next 
        group_len += 1 
    return head 

def remove_duplicates(head):
    current = head 
    while current is not None and current.next is not None:
        if current.next.val == current.val:
            # if duplicate, skip it
            current.next = current.next.next 
        else:
            # if not duplicate, just move to next
            current = current.next 
    return head 

def remove_elements(head, k):
    dummy = ListNode(0)
    dummy.next = head 
    prev = dummy 
    curr = head 
    while curr is not None:
        if curr.val == k:
            # update next pointer of previous node to skip current node
            prev.next = curr.next 
            curr = curr.next 
        else:
            # current node's data doesn't match k, move both pointers forward
            prev = curr 
            curr = curr.next 
    return dummy.next

def split_list_to_parts(head, k):
    ans = [None] * k 
    size = 0 
    current = head 
    while current is not None:
        size += 1 
        current = current.next 
    # base size of each part 
    split = size // k 
    remaining = size % k 
    current = head 
    prev = current 
    for i in range(k):
        new = current 
        current_size = split 
        if remaining > 0:
            remaining -= 1
            current_size += 1 
        # traverse current part to its end 
        j = 0 
        while j < current_size:
            prev = current 
            if current is not None:
                current = current.next 
            j += 1 
        # disconnect current part from rest of list 
        if prev is not None:
            prev.next = None 
        ans[i] = new 
    return ans 

def delete_nodes(head, m, n):
    current = head 
    last_m_node = head 
    while current:
        m_count = m 
        while current and m_count > 0:
            last_m_node = current 
            current = current.next 
            m_count -= 1 
        n_count = n 
        while current and n_count > 0:
            current = current.next 
            n_count -= 1 
        last_m_node.next = current 
    return head 
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


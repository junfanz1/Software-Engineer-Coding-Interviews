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
   * [9. Binary Search](#9-binary-search)
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
from heapq import heappush, heappop 
import heapq

def maximum_capital(c, k, capitals, profits):
    current_capital = c 
    capitals_min_heap = []
    profits_max_heap = []
    # insert all capitals values to minheap
    for x in range(0, len(capitals)):
        heappush(capitals_min_heap, (capitals[x], x))
    for _ in range(k):
        # store negative as we need max heap
        while capitals_min_heap and capitals_min_heap[0][0] <= current_capital:
            c, i = heappop(capitals_min_heap)
            heappush(profits_max_heap, (-profits[i]))
        # if max heap empty
        if not profits_max_heap:
            break 
        # select from maxheap with max profit, pop a negated element
        j = -heappop(profits_max_heap)
        current_capital = current_capital + j
    return current_capital

class MedianOfStream:
    def __init__(self):
        self.max_heap_for_smallnum = []
        self.min_heap_for_largenum = []
    def insert_num(self, num):
        if not self.max_heap_for_smallnum or -self.max_heap_for_smallnum[0] >= num:
            heappush(self.max_heap_for_smallnum, -num)
        else:
            heappush(self.min_heap_for_largenum, num)
        if len(self.max_heap_for_smallnum) > len(self.min_heap_for_largenum) + 1:
            heappush(self.min_heap_for_largenum, -heappop(self.max_heap_for_smallnum))
        elif len(self.max_heap_for_smallnum) < len(self.min_heap_for_largenum):
            heappush(self.max_heap_for_smallnum, -heappop(self.min_heap_for_largenum))
    def find_median(self):
        if len(self.max_heap_for_smallnum) == len(self.min_heap_for_largenum):
            return -self.max_heap_for_smallnum[0] / 2.0 + self.min_heap_for_largenum[0] / 2.0
        return -self.max_heap_for_smallnum[0] / 1.0
    
def median_sliding_window(nums, k):
    medians = []
    outgoing_num = {}
    small_list = [] # max heap
    large_list = [] # min heap
    # * -1 for max heap
    for i in range(0, k):
        heappush(small_list, -1 * nums[i])
    # transfer top 50% of numbers from max heap to min heap while restoring sign of each number
    for i in range(0, k // 2):
        element = heappop(small_list)
        heappush(large_list, -1 * element)
    # keep heaps balanced
    balance = 0
    i = k 
    while True:
        # window size odd
        if (k & 1) == 1:
            medians.append(float(small_list[0] * -1))
        else:
            medians.append((float(small_list[0] * -1) + float(large_list[0])) * 0.5)
        # all element processed, break loop
        if i >= len(nums):
            break 
        # outgoing number
        out_num = nums[i - k]
        # incoming number
        in_num = nums[i]
        i += 1
        # if outgoing number from max heap
        if out_num <= (small_list[0] * -1):
            balance -= 1
        else:
            balance += 1 
        # add/update outgoing number in hash map
        if out_num in outgoing_num:
            outgoing_num[out_num] = outgoing_num[out_num] + 1
        else:
            outgoing_num[out_num] = 1 
        # if incoming number < top of max heap, add in heap, otherwise add in min heap 
        if small_list and in_num <= (small_list[0] * -1):
            balance += 1 
            heappush(small_list, in_num * -1)
        else:
            balance -= 1 
            heappush(large_list, in_num)
        if balance < 0:
            heappush(small_list, (-1 * large_list[0]))
            heappop(large_list)
        elif balance > 0:
            heappush(large_list, (-1 * small_list[0]))
            heappop(small_list)
        # heaps balanced, reset balance to 0
        balance = 0 
        # remove invalid numbers in hash map from top of max heap
        while (small_list[0] * -1) in outgoing_num and (outgoing_num[(small_list[0] * -1)] > 0):
            outgoing_num[small_list[0] * -1] = outgoing_num[small_list[0] * -1] - 1
            heappop(small_list)
        # remove invalid numbers in hash map from top of min heap
        while large_list and large_list[0] in outgoing_num and (outgoing_num[large_list[0]] > 0):
            outgoing_num[large_list[0]] = outgoing_num[large_list[0]] - 1
            heappop(large_list)
    return medians 

def min_machines(tasks):
    tasks.sort()
    machines = [] # min heap
    for task in tasks:
        start, end = task 
        if machines and machines[0] <= start:
            # if earliest machine is free, reuse machine 
            heapq.heappop(machines)
        heapq.heappush(machines, end) # assign a machine 
    # return size of heap as min number of machine 
    return len(machines)


def most_booked(meetings, rooms):
    count = [0] * rooms 
    available = [i for i in range(rooms)]
    used_rooms = []
    meetings.sort()
    for start_time, end_time in meetings:
        # free up rooms that have finished 
        while used_rooms and used_rooms[0][0] <= start_time:
            ending, room = heapq.heappop(used_rooms)
            heapq.heappush(available, room)
        # if no rooms available, delay meeting 
        if not available:
            end, room = heapq.heappop(used_rooms)
            end_time = end + (end_time - start_time)
            heapq.heappush(available, room)
        # allocate meeting to available room with lowest number 
        room = heapq.heappop(available)
        heapq.heappush(used_rooms, (end_time, room))
        count[room] += 1
    # room held most meetings 
    return count.index(max(count))

def largest_integer(num):
    digits = [int(d) for d in str(num)]
    odd_heap = []
    even_heap = []
    for d in digits:
        if d % 2 == 0:
            heapq.heappush(even_heap, -d) # negative for max heap
        else:
            heapq.heappush(odd_heap, -d)
    result = []
    for d in digits:
        if d % 2 == 0:
            largest_even = -heapq.heappop(even_heap)
            result.append(largest_even)
        else:
            largest_odd = -heapq.heappop(odd_heap)
            result.append(largest_odd)
    return int(''.join(map(str, result)))

def find_right_interval(intervals):
    result = [-1] * len(intervals)
    start_heap = []
    end_heap = []
    for i, interval in enumerate(intervals):
        heapq.heappush(start_heap, (interval[0], i))
        heapq.heappush(end_heap, (interval[1], i))
    # process each interval based on end points 
    while end_heap:
        value, index = heapq.heappop(end_heap)
        # remove all start points from start_heap < current end point 
        while start_heap and start_heap[0][0] < value:
            heapq.heappop(start_heap)
        # if start heap not empty, top element is smallest valid right interval 
        if start_heap:
            result[index] = start_heap[0][1]
    return result 

def connect_sticks(sticks):
    heapq.heapify(sticks) # convert list to minheap
    total_cost = 0 
    # continue until only one stick in the heap
    while len(sticks) > 1:
        first = heapq.heappop(sticks)
        second = heapq.heappop(sticks)
        cost = first + second 
        total_cost += cost 
        # push combined stick back into heap
        heapq.heappush(sticks, cost)
    return total_cost

def longest_diverse_string(a, b, c):
    pq = []
    if a > 0:
        heapq.heappush(pq, (-a, "a")) # push a with its count 
    if b > 0:
        heapq.heappush(pq, (-b, "b"))
    if c > 0:
        heapq.heappush(pq, (-c, "c"))
    result = []
    while pq:
        # pop character with highest remaining freq
        count, character = heapq.heappop(pq)
        count = -count # convert back to positive 
        if (
            len(result) >= 2 
            and result[-1] == character 
            and result[-2] == character
        ):
            # rule violated, no alternative character exists
            if not pq:
                break 
            # use next most frequent character temporarily
            tempCnt, tempChar = heapq.heappop(pq)
            result.append(tempChar) # add alternative character
            # push alternative character back with its updated count
            if (tempCnt + 1) < 0:
                heapq.heappush(pq, (tempCnt + 1, tempChar))
            # push original character back to heap to try adding it later 
            heapq.heappush(pq, (-count, character))
        else:
            # if no violation, add current character to result 
            count -= 1 
            result.append(character)
            # push character back to heap if still has remaining
            if count > 0:
                heapq.heappush(pq, (-count, character))
    return "".join(result)

def gain(passes, total):
    return (float(passes + 1) / (total + 1)) - (float(passes) / total)
def max_average_ratio(classes, extraStudents):
    max_heap = []
    for passes, total in classes:
        heapq.heappush(max_heap, (-gain(passes, total), passes, total))
    # distributed extra students
    for _ in range(extraStudents):
        current_gain, passes, total = heapq.heappop(max_heap)
        passes += 1
        total += 1
        heapq.heappush(max_heap, (-gain(passes, total), passes, total))
    total_ratio = sum(float(passes) / total for _, passes, total in max_heap)
    return total_ratio / len(classes)

def smallest_chair(times, target_friend):
    sorted_friends = sorted(enumerate(times), key=lambda x: x[1][0])
    available_chairs = []
    occupied_chairs = []
    chair_index = 0
    # process each friend in order of arrival time
    for friend_id, (arrival, leaving) in sorted_friends:
        while occupied_chairs and occupied_chairs[0][0] <= arrival:
            _, freed_chair = heapq.heappop(occupied_chairs) # remove release chair
            heapq.heappush(available_chairs, freed_chair)
        # assign smallest available chair
        if available_chairs:
            assigned_chair = heapq.heappop(available_chairs)
        else:
            assigned_chair = chair_index # use new chair if none are available
            chair_index += 1 # move to next available chair number
        heapq.heappush(occupied_chairs, (leaving, assigned_chair)) # store chair assignment with leaving time 
        if friend_id == target_friend:
            return assigned_chair
```

<!-- TOC --><a name="7-k-way-merge"></a>
## 7. K-way merge 

```py
import heapq
def merge_sorted(nums1, m, nums2, n):
    p1 = m - 1 
    p2 = n - 1
    for p in range(n + m - 1, -1, -1):
        if p2 < 0:
            break 
        if p1 >= 0 and nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1 
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
    return nums1 

def k_smallest_number(lists, k):
    list_length = len(lists)
    kth_smallest = []
    for index in range(list_length):
        if len(list[index]) == 0: # if no elements in input list, continue to next iteration
            continue 
        else:
            # place first element of each list in minheap
            heappush(kth_smallest, (lists[index][0], index, 0))
    # set a counter to match if kth element = that counter, return number 
    numbers_checked, smallest_number = 0, 0
    while kth_smallest:
        smallest_number, list_index, num_index = heappop(kth_smallest)
        numbers_checked += 1 
        if numbers_checked == k:
            break 
        # if more elements in list of top element, add next element of that list to minheap 
        if num_index + 1 < len(list[list_index]):
            heappush(
                kth_smallest, (lists[list_index][num_index + 1], list_index, num_index + 1)
            )
    return smallest_number

def k_smallest_pairs(list1, list2, k):
    list_length = len(list1)
    min_heap = []
    pairs = []
    for i in range(min(k, list_length)):
        heappush(min_heap, (list1[i] + list2[0], i, 0))
    counter = 1 
    while min_heap and counter <= k:
        sum_of_pairs, i, j = heappop(min_heap) 
        pairs.append([list1[i], list2[j]])
        next_element = j + 1
        # if next element available for list2 then add to heap
        if len(list2) > next_element:
            heappush(
                min_heap,
                (list1[i] + list2[next_element], i, next_element)
            )
        counter += 1 
    return pairs 

def merge_2_lists(head1, head2):
    dummy = ListNode(-1)
    prev = dummy 
    while head1 and head2:
        if head1.val <= head2.val:
            prev.next = head1 
            head1 = head1.next 
        else:
            prev.next = head2 
            head2 = head2.next 
        prev = prev.next 
    if head1 is not None:
        prev.next = head1 
    else:
        prev.next = head2 
    return dummy.next 
def merge_k_lists(lists):
    if len(lists) > 0:
        step = 1 
        while step < len(lists):
            # merge lists that're step apart
            for i in range(0, len(lists) - step, step * 2):
                lists[i].head = merge_2_lists(lists[i].head, lists[i + step].head)
            step *= 2 
        return lists[0].head 
    return 

def kth_smallest_element(matrix, k):
    row_count = len(matrix)
    min_numbers = []
    for index in range(min(row_count, k)):
        heappush(min_numbers, (matrix[index][0], index, 0))
    numbers_checked, smallest_element = 0, 0
    while min_numbers:
        smallest_element, row_index, col_index = heappop(min_numbers)
        numbers_checked += 1 
        if numbers_checked == k:
            break # return smallest element
        # if current popped element has next element in its row, add next element to minheap
        if col_index + 1 < len(matrix[row_index]):
            heappush(min_numbers, (matrix[row_index][col_index + 1], row_index, col_index + 1))
    return smallest_element

def kth_smallest_prime_fraction(arr, k):
    n = len(arr)
    min_heap = []
    for j in range(1, n):
        min_heap.append((arr[0] / arr[j], 0, j))
    heapq.heapify(min_heap)
    # remove smallest fraction k - 1 times 
    for _ in range(k - 1):
        value, i, j = heapq.heappop(min_heap)
        if i + 1 < j:
            heapq.heappush(min_heap, (arr[i + 1] / arr[j], i + 1, j))
    # kth smallest fraction now at top of min heap
    _, i, j = heapq.heappop(min_heap)
    return [arr[i], arr[j]]

def nth_super_ugly_number(n, primes):
    ugly = [1]
    # each prime starts with value as first multiple, and index 0 (pointing to 1 in ugly list)
    min_heap = [(prime, prime, 0) for prime in primes]
    heapq.heapify(min_heap) # convert list to heap to maintain minheap
    # until find n super ugly numbers 
    while len(ugly) < n:
        next_ugly, prime, index = heapq.heappop(min_heap)
        # avoid duplicates by only appending if different from last added 
        if next_ugly != ugly[-1]:
            ugly.append(next_ugly)
        heapq.heappush(min_heap, (prime * ugly[index + 1], prime, index + 1))
    return ugly[-1]
```

<!-- TOC --><a name="8-top-k-elements"></a>
## 8. Top K Elements

```py
import heapq 
class KthLargest:
    def __init__(self, k, nums):
        self.top_k_heap = []
        self.k = k 
        for element in nums:
            self.add(element)
    def add(self, val):
        if len(self.top_k_heap) < self.k:
            heapq.heappush(self.top_k_heap, val)
        elif val > self.top_k_heap[0]:
            heapq.heappop(self.top_k_heap)
            heapq.heappush(self.top_k_heap, val)
        return self.top_k_heap[0]
    
from collections import Counter 
def reorganize_string(str):
    char_counter = Counter(str)
    most_freq_chars = []
    for char, count in char_counter.items():
        most_freq_chars.append([-count, char])
    heapq.heapify(most_freq_chars)
    previous = None 
    result = ""
    while len(most_freq_chars) > 0 or previous:
        if previous and len(most_freq_chars) == 0:
            return ""
        count, char = heapq.heappop(most_freq_chars)
        result = result + char 
        count = count + 1 
        if previous:
            heapq.heappush(most_freq_chars, previous)
            previous = None 
        if count != 0:
            previous = [count, char]
    return result 

from point import Point
def k_closest(points, k):
    points_max_heap = []
    for i in range(k):
        heapq.heappush(points_max_heap, points[i])
    for i in range(k, len(points)):
        if points[i].distance_from_origin() < points_max_heap[0].distance_from_origin():
            heapq.heappop(points_max_heap)
            heapq.heappush(points_max_heap, points[i])
    return list(points_max_heap)

from heapq import heappush, heappop
def top_k_frequent(arr, k):
    num_frequency_map = {}
    for num in arr:
        num_frequency_map[num] = num_frequency_map.get(num, 0) + 1 
    top_k_elements = []
    for num, frequency in num_frequency_map.items():
        heappush(top_k_elements, (frequency, num))
        if len(top_k_elements) > k:
            heappop(top_k_elements)
    top_numbers = []
    while top_k_elements:
        top_numbers.append(heappop(top_k_elements)[1])
    return top_numbers

def find_kth_largest(nums, k):
    k_numbers_min_heap = []
    # add first k elements to the list 
    for i in range(k):
        heapq.heappush(k_numbers_min_heap, nums[i])
    # remaining elements in nums array
    for i in range(k, len(nums)):
        if nums[i] > k_numbers_min_heap[0]:
            heapq.heappop(k_numbers_min_heap) # remove smallest
            heapq.heappush(k_numbers_min_heap, nums[i]) # add current
    return k_numbers_min_heap # root of heap as kth largest element

from collections import defaultdict
def third_max(nums):
    heap = []
    taken = set()
    for index in range(len(nums)):
        # skip number if already in set duplicate
        if nums[index] in taken:
            continue 
        if len(heap) == 3:
            if heap[0] < nums[index]:
                # remove smallest from both heap and set
                taken.remove(heap[0])
                heapq.heappop(heap)
                heapq.heappush(heap, nums[index])
                taken.add(nums[index])
        else:
            heapq.heappush(heap, nums[index])
            taken.add(nums[index])
    if len(heap) == 1:
        return heap[0]
    elif len(heap) == 2:
        first_num = heap[0]
        heapq.heappop(heap)
        return max(first_num, heap[0])
    return heap[0]

def max_subsequence(nums, k):
    min_heap = []
    for i, num in enumerate(nums):
        heapq.heappush(min_heap, (num, i))
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    result = sorted(min_heap, key=lambda x: x[1]) # sort by original index 
    return [x[0] for x in result] # extract values 

def min_cost_to_hire_workers(quality, wage, k):
    workers = sorted([(w / q, q) for w, q in zip(wage, quality)])
    heap = []
    total_quality = 0 
    min_cost = float('inf')
    for ratio, q in workers:
        heapq.heappush(heap, -q)
        total_quality += q 
        if len(heap) > k: # more than k workers, remove the largest quality
            total_quality += heapq.heappop(heap)
        if len(heap) == k:
            min_cost = min(min_cost, ratio * total_quality)
    return min_cost

def max_score(nums, k):
    max_heap = [-num for num in nums]
    heapq.heapify(max_heap)
    score = 0
    for _ in range(k):
        largest = -heapq.heappop(max_heap)
        score += largest 
        reduced = (largest + 2) // 3 
        heapq.heappush(max_heap, -reduced) # negate for max heap
    return score

def kth_largest_integer(nums, k):
    heap = []
    for num in nums:
        heapq.heappush(heap, int(num))
        if len(heap) > k:
            heapq.heappop(heap)
    return str(heap[0])
```

<!-- TOC --><a name="9-binary-search"></a>
## 9. Binary Search

```py
def binary_search(nums, target):
    low = 0 
    high = len(nums) - 1 
    while low <= high:
        mid = low + ((high - low) // 2)
        if nums[mid] == target:
            return mid 
        elif target < nums[mid]:
            high = mid - 1 
        elif target > nums[mid]:
            low = mid + 1 
    return -1 

def binary_search_rotated(nums, target):
    low = 0
    high = len(nums) - 1 
    while low <= high:
        mid = low + (high - low) // 2 
        if nums[mid] == target:
            return mid 
        if nums[low] <= nums[mid]:
            if nums[low] <= target and target < nums[mid]:
                high = mid - 1 
            else:
                low = mid + 1 
        else:
            if nums[mid] < target and target <= nums[high]:
                low = mid + 1 
            else:
                high = mid - 1 
    return -1 

from bad_version import BadVersion 
class Solution(BadVersion):
    def first_bad_version(self, n):
        first = 1 
        last = n 
        while first <= last:
            mid = first + (last - first) // 2 
            if self.is_bad_version(mid):
                last = mid - 1 
            else:
                first = mid + 1 
        return first 
    
import random 
class RandomPickWithWeight:
    def __init__(self, weights):
        self.running_sums = []
        running_sum = 0
        for w in weights:  
            running_sum += w 
            self.running_sums.append(running_sum)
        self.total_sum = running_sum
    def pick_index(self):
        target = random.randint(1, self.total_sum)
        low = 0 
        high = len(self.running_sums)
        while low < high:
            mid = low + (high - low) // 2 
            if target > self.running_sums[mid]:
                low = mid + 1 
            else:
                high = mid 
        return low 

def binary_search(array, target):
    left = 0 
    right = len(array) - 1 
    while left <= right:
        mid = (left + right) // 2 
        if array[mid] == target:
            return mid 
        if array[mid] < target:
            left = mid + 1 
        else:
            right = mid - 1 
    return left 
def find_closest_elements(nums, k, target):
    if len(nums) == k:
        return nums 
    if target <= nums[0]:
        return nums[0:k]
    if target >= nums[-1]:
        return nums[len(nums)-k : len(nums)]
    first_closest = binary_search(nums, target)
    window_left = first_closest - 1 
    window_right = window_left + 1 
    while (window_right - window_left - 1) < k:
        if window_left == -1:
            window_right += 1 
            continue 
        # if right pointer out of bound, or element pointed to by left pointer is closer to target
        if window_right == len(nums) or abs(nums[window_left] - target) <= abs(nums[window_right] - target):
            window_left -= 1 
        else:
            window_right += 1 
    return nums[window_left + 1: window_right]

def single_non_duplicate(nums):
    l = 0 
    r = len(nums) - 1 
    while l != r:
        mid = l + (r - l) // 2 
        if mid % 2 == 1:
            mid -= 1
        if nums[mid] == nums[mid + 1]:
            l = mid + 2 
        else:
            r = mid 
    return nums[l]

def calculate_sum(index, mid, n):
    count = 0
    if mid > index:
        count += (mid + mid - index) * (index + 1) // 2 
    else:
        count += (mid + 1) * mid // 2 + index - mid + 1 
    if mid >= n - index:
        count += (mid + mid - n + 1 + index) * (n - index) // 2 
    else:
        count += (mid + 1) * mid // 2 + n - index - mid 
    return count - mid 
def max_value(n, index, maxSum):
    left, right = 1, maxSum 
    while left < right:
        mid = (left + right + 1) // 2 
        if calculate_sum(index, mid, n) <= maxSum:
            left = mid 
        else:
            right = mid - 1 
    return left 

def find_k_weakest_rows(matrix, k):
    m = len(matrix)
    n = len(matrix[0])
    def binary_search(row):
        low = 0 
        high = n 
        while low < high:
            mid = low + (high - low) // 2 
            if row[mid] == 1:
                low = mid + 1 
            else:
                high = mid 
        return low 
    # priority queue / minheap to store k weakest rows
    pq = []
    for i, row in enumerate(matrix):
        strength = binary_search(row) # find strength of row
        entry = (-strength, -i) # negative to prioritize weak rows and small index
        if len(pq) < k or entry > pq[0]: # add row to heap if haven't found k rows
            heapq.heappush(pq, entry)
        if len(pq) > k: # remove strongest row if heap has >k rows
            heapq.heappop(pq)
    indexes = [] # k weakest rows from heap
    while pq:
        strength, i = heapq.heappop(pq)
        indexes.append(-i) # append index of weakest row 
    indexes = indexes[::-1] # reverse to get order from weakest to strongest
    return indexes 

def can_split(nums, k, mid):
    subarrays = 1 
    current_sum = 0
    for num in nums:
        if current_sum + num > mid:
            subarrays += 1 
            current_sum = num 
            if subarrays > k:
                return False 
        else:
            current_sum += num 
    return True 
def split_array(nums, k):
    left, right = max(nums), sum(nums)
    while left < right:
        mid = (left + right) // 2 
        if can_split(nums, k, mid):
            right = mid 
        else:
            left = mid + 1 
    return left 
```

<!-- TOC --><a name="10-subsets"></a>
## 10. Subsets

```py
def get_bit(num, bit):
    temp = (1 << bit)
    temp = temp & num 
    if temp == 0:
        return 0 
    return 1 
def find_all_subsets(nums):
    subsets = []
    if not nums:
        return [[]]
    else:
        subsets_count = 2 ** len(nums)
        for i in range(0, subsets_count):
            subset = set()
            for j in range(0, len(nums)):
                if get_bit(i, j) == 1 and nums[j] not in subset:
                    subset.add(nums[j])
            if i == 0:
                subsets.append([])
            else:
                subsets.append(list(subset))
    return subsets 

def swap_char(word, i, j):
    swap_index = list(word)
    swap_index[i], swap_index[j] = swap_index[j], swap_index[i]
    return ''.join(swap_index)
def permute_string_rec(word, current_index, result):
    if current_index == len(word) - 1:
        result.append(word)
        return 
    for i in range(current_index, len(word)):
        swapped_str = swap_char(word, current_index, i)
        permute_string_rec(swapped_str, current_index + 1, result)
def permute_word(word):
    result = []
    permute_string_rec(word, 0, result)
    return result

def backtrack(index, path, digits, letters, combinations):
    if len(path) == len(digits):
        combinations.append(''.join(path))
        return 
    possible_letters = letters[digits[index]]
    if possible_letters:
        for letter in possible_letters:
            path.append(letter)
            backtrack(index + 1, path, digits, letters, combinations)
            path.pop()
def letter_combinations(digits):
    combinations = []
    if len(digits) == 0:
        return []
    digits_mapping = {
        "1": [""],
        "2": ["a", "b", "c"],
        "3": ["d", "e", "f"],
        "4": ["g", "h", "i"],
        "5": ["j", "k", "l"],
        "6": ["m", "n", "o"],
        "7": ["p", "q", "r", "s"],
        "8": ["t", "u", "v"],
        "9": ["w", "x", "y", "z"]}
    backtrack(0, [], digits, digits_mapping, combinations)
    return combinations

def back_track(n, left_count, right_count, output, result):
    if left_count >= n and right_count >= n:
        result.append("".join(output))
    if left_count < n: 
        output.append('(')
        back_track(n, left_count + 1, right_count, output, result)
        output.pop()
    if right_count < left_count:
        output.append(')')
        back_track(n, left_count, right_count + 1, output, result)
        output.pop()

def letter_case_permutation(s):
    result = [""]
    for ch in s:
        if ch.isalpha():
            result = [str + ch.lower() for str in result] + [str + ch.upper() for str in result]
        else:
            result = [str + ch for str in result]
    return result 
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


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
def jump_game(nums):
    target_num_index = len(nums) - 1 
    for i in range(len(nums) - 2, -1, -1):
        if target_num_index <= i + nums[i]:
            target_num_index = i 
    if target_num_index == 0:
        return True
    return False 

def rescue_boats(people, limit):
    people.sort()
    left = 0 
    right = len(people) - 1
    boats = 0
    while left <= right:
        if people[left] + people[right] <= limit:
            left += 1 
        right -= 1
        boats += 1 
    return boats 

def gas_station_journey(gas, cost):
    if sum(cost) > sum(gas):
        return -1 
    current_gas, starting_index = 0, 0
    for i in range(len(gas)):
        current_gas += (gas[i] - cost[i])
        if current_gas < 0:
            current_gas = 0
            starting_index = i + 1 
    return starting_index

def two_city_scheduling(costs):
    total_cost = 0 
    costs.sort(key=lambda x: x[0] - x[1])
    cost_length = len(costs)
    for i in range(cost_length // 2):
        total_cost = total_cost + costs[i][0] + costs[cost_length-i-1][1]
    return total_cost

import heapq
def min_refuel_stops(target, start_fuel, stations):
    if start_fuel >= target:
        return 0
    max_heap = []
    i, n = 0, len(stations)
    stops = 0 
    max_distance = start_fuel
    while max_distance < target:
        if i < n and stations[i][0] <= max_distance:
            heapq.heappush(max_heap, -stations[i][1])
            i += 1 
        elif not max_heap:
            return -1 
        else:
            max_distance += -heapq.heappop(max_heap)
            stops += 1 
    return stops 

from collections import Counter 
def largest_palindrome(num):
    # count freq of each digit in input string
    occurences = Counter(num)
    first_half = []
    middle = ""
    for digit in range(9, -1, -1):
        digit_char = str(digit)
        if digit_char in occurences:
            digit_count = occurences[digit_char]
            num_pairs = digit_count // 2 
            if num_pairs:
                if not first_half and not digit:
                    occurences["0"] = 1 
                else:
                    first_half.append(digit_char * num_pairs)
        if digit_count % 2 and not middle:
            middle = digit_char
    if not middle and not first_half:
        return "0"
    return "".join(first_half + [middle] + first_half[::-1])

def find_content_children(greed_factors, cookie_sizes):
    greed_factors.sort()
    cookie_sizes.sort()
    current_child, current_cookie = 0, 0 
    content_children = 0 
    while current_child < len(greed_factors) and current_cookie < len(cookie_sizes):
        if cookie_sizes[current_cookie] >= greed_factors[current_child]:
            content_children += 1 
            current_child += 1 
        current_cookie += 1 
    return content_children

from collections import Counter 
import heapq 
def min_cost_to_rearrange_fruits(basket1, basket2):
    combined = basket1 + basket2 
    combined_counter = Counter(combined)
    for count in combined_counter.values():
        if count % 2 != 0:
            return -1 
    counter1 = Counter(basket1)
    counter2 = Counter(basket2)
    excess1 = []
    excess2 = []
    for fruit in combined_counter:
        diff = counter1[fruit] - counter2[fruit]
        if diff > 0:
            excess1.extend([fruit] * (diff // 2))
        elif diff < 0:
            excess2.extend([fruit] * (-diff // 2))
    excess1.sort()
    excess2.sort(reverse=True)
    min_fruit_cost = min(combined_counter.keys())
    total_cost = 0 
    for i in range(len(excess1)):
        total_cost += min(2 * min_fruit_cost, excess1[i], excess2[i])
    return total_cost

def num_steps(str):
    length = len(str)
    steps = 0
    c = 0
    for i in range(length - 1, 0, -1):
        digit = int(str[i]) + c 
        if digit % 2 == 1:
            steps += 2 
            c = 1 
        else:
            steps += 1 
    return steps + c 

def max_swap(num):
    num_string = list(str(num))
    n = len(num_string)
    max_digit_index = index_1 = index_2 = -1
    for i in range(n - 1, -1, -1):
        if max_digit_index == -1 or num_string[i] > num_string[max_digit_index]:
            max_digit_index = i 
        elif num_string[i] < num_string[max_digit_index]:
            index_1 = i 
            index_2 = max_digit_index
    if index_1 != -1 and index_2 != -1:
        num_string[index_1], num_string[index_2] = num_string[index_2], num_string[index_1]
    return int("".join(num_string))

def can_place_flowers(flowerbed, n):
    count = 0 
    for i in range(len(flowerbed)):
        if flowerbed[i] == 0:
            left = i == 0 or flowerbed[i - 1] == 0 
            right = i == len(flowerbed) - 1 or flowerbed[i + 1] == 0 
            # if both left and right are empty, plant a flower
            if left and right:
                flowerbed[i] = 1 
                count += 1 
                # if n flowers are planted, true
                if count == n:
                    return True 
    return count >= n 

def largest_odd_number(num):
    for i in range(len(num) - 1, -1, -1):
        if int(num[i]) % 2 == 1:
            return num[:i + 1]
    return "" # if no odd digit found, return empty string
```

<!-- TOC --><a name="12-backtracking"></a>
## 12. Backtracking

```py
def is_valid_move(proposed_row, proposed_col, solution):
    for i in range(0, proposed_row):
        old_row = i 
        old_col = solution[i]
        diagonal_offset = proposed_row - old_row 
        if (old_col == proposed_col or old_col == proposed_col - diagonal_offset 
            or old_col == proposed_col + diagonal_offset):
            return False 
    return True 
# recursive worker function
def solve_n_queens_rec(n, solution, row, results):
    if row == n:
        results.append(solution[:])
        return 
    for i in range(0, n):
        valid = is_valid_move(row, i, solution)
        if valid:
            solution[row] = i 
            solve_n_queens_rec(n, solution, row + 1, results)
def solve_n_queens(n):
    results = []
    solution = [-1] * n 
    solve_n_queens_rec(n, solution, 0, results)
    return len(results)

# stack version queen
def solve_n_queens(n):
    results = []
    solution = [-1] * n 
    sol_stack = []
    row = 0 
    col = 0
    while row < n:
        # check if queen can be placed in any column of this row 
        while col < n:
            if is_valid_move(row, col, solution):
                # save to current solution on stack
                sol_stack.append(col)
                solution[row] = col 
                row += 1 
                col = 0 
                # move on to check next row and break out inner loop
                break 
            col += 1 

        # if checked all columns
        if col == n:
            # if working on solution 
            if sol_stack:
                # backtracking, as current row doesn't offer safe spot given previous move
                # get setup to check previous row with next column
                col = sol_stack[-1] + 1 
                sol_stack.pop()
                row -= 1 
            else:
                # backtracked all the way and found dead-end, break out inner loop, no more solutions exist 
                break

        if row == n:
            # add solution to results 
            results.append(solution[:])
            row -= 1 # backtrack for next solution 
            col = sol_stack[-1] + 1 
            sol_stack.pop()
    return len(results)

def word_search(grid, word):
    n = len(grid)
    m = len(grid[0])
    for row in range(n):
        for col in range(m):
            if dfs(row, col, word, 0, grid):
                return True 
    return False 
def dfs(row, col, word, index, grid):
    if len(word) == index:
        return True 
    if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) or grid[row][col] != word[index]:
        return False 
    temp = grid[row][col]
    grid[row][col] = '*'
    for rowOffset, colOffset in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        if dfs(row + rowOffset, col + colOffset, word, index + 1, grid):
            return True 
    grid[row][col] = temp 
    return False 

from TreeNode import * 
from BinaryTree import * 
def rob(root):
    return max(heist(root))
def heist(root):
    if root == None:
        return [0, 0]
    left_subtree = heist(root.left)
    right_subtree = heist(root.right)
    include_root = root.data + left_subtree[1] + right_subtree[1]
    exclude_root = max(left_subtree) + max(right_subtree)
    return [include_root, exclude_root]

def valid(segment):
    segment_length = len(segment)
    if segment_length > 3: 
        return False 
    return int(segment) <= 255 if segment[0] != '0' else len(segment) == 1 
def update_segment(s, curr_dot, segments, result):
    segment = s[curr_dot + 1: len(s)]
    if valid(segment):
        segments.append(segment)
        result.append('.'.join(segments))
        segments.pop()
def backtrack(s, prev_dot, dots, segments, result):
    size = len(s)
    for curr_dot in range(prev_dot + 1, min(size - 1, prev_dot + 4)):
        segment = s[prev_dot + 1: curr_dot + 1]
        if valid(segment):
            segments.append(segment)
            if dots - 1 == 0:
                update_segment(s, curr_dot, segments, result)
            else:
                backtrack(s, curr_dot, dots - 1, segments, result)
            segments.pop()
def restore_ip_addresses(s):
    result, segments = [], []
    backtrack(s, -1, 3, segments, result)
    return result 

def flood_fill(grid, sr, sc, target):
    if grid[sr][sc] == target:
        return grid 
    else:
        old_target = grid[sr][sc]
        grid[sr][sc] = target 
        dfs(grid, sr, sc, old_target, target)
        return grid 
def dfs(grid, row, col, old_target, new_target):
    adjacent_cells = [[0, 1], [1, 0], [-1, 0], [0, -1]]
    grid_length = len(grid)
    total_cells = len(grid[0])
    for cell_value in adjacent_cells:
        i = row + cell_value[0]
        j = col + cell_value[1]
        if i < grid_length and i >= 0 and j < total_cells and j >= 0 and grid[i][j] == old_target:
            grid[i][j] = new_target
            dfs(grid, i, j, old_target, new_target)

def min_moves(grid):
    zeros = []
    extras = []
    min_moves = float('inf')
    total_stones = sum(sum(row) for row in grid)
    if total_stones != 9:
        return -1 
    def solve(i, count):
        if i >= len(zeros):
            nonlocal min_moves 
            min_moves = min(min_moves, count)
            return 
        for k in range(len(extras)):
            if extras[k][2] != 0:
                extras[k][2] -= 1 
                solve(i + 1, abs(extras[k][0] - zeros[i][0]) + abs(extras[k][1] - zeros[i][1]) + count)
                extras[k][2] += 1 
    for x in range(3):
        for y in range(3):
            if grid[x][y] == 0:
                zeros.append([x, y])
            elif grid[x][y] > 1:
                extras.append([x, y, grid[x][y] - 1])
    if len(zeros) == 0:
        return 0 
    solve(0, 0)
    return min_moves 

from binarytree import BinaryTree
from treenode import TreeNode 
def binary_tree_path(root):
    def backtrack(root, path):
        if root:
            path += str(root.data)
            if not root.left and not root.right:
                paths.append(path)
            else:
                path += '->'
                backtrack(root.left, path)
                backtrack(root.right, path)
    paths = []
    backtrack(root, '')
    return paths 

from math import pow 
def binary_watch_rec(position, hours, minutes, enabled, result):
    if enabled == 0:
        if hours <= 11 and minutes <= 59:
            time = f"{hours}:{'0' if minutes < 10 else ''}{minutes}"
            result.append(time)
        return # end recursion
    for i in range(position, 10):
        h, m = hours, minutes 
        if i <= 3:
            hours += int(pow(2, i))
        else:
            minutes += int(pow(2, i - 4))
        binary_watch_rec(i + 1, hours, minutes, enabled - 1, result)
        hours, minutes = h, m 
def read_binary_watch(enabled):
    result = []
    binary_watch_rec(0, 0, 0, enabled, result)
    return result 

import collections 
def dfs(current, n, balance):
    while current < n and not balance[current]:
        current += 1 
    if current == n:
        return 0 
    cost = float('inf')
    for next in range(current + 1, n):
        if balance[next] * balance[current] < 0:
            balance[next] += balance[current]
            cost = min(cost, 1 + dfs(current + 1, n, balance))
            balance[next] -= balance[current]
    return cost 
def min_transfers(transactions):
    balance_map = collections.defaultdict(int)
    for a, b, amount in transactions:
        balance_map[a] += amount 
        balance_map[b] -= amount 
    balance = [amount for amount in balance_map.values() if amount]
    n = len(balance)
    return dfs(0, n, balance)

def max_unique_split(s):
    seen = set()
    return backtrack(s, 0, seen)
def backtrack(s, start, seen):
    if start == len(s):
        return 0
    max_count = 0
    for end in range(start + 1, len(s) + 1):
        sub_string = s[start: end]
        if sub_string not in seen:
            seen.add(sub_string)
            max_count = max(max_count, 1 + backtrack(s, end, seen))
            seen.remove(sub_string)
    return max_count 
```

<!-- TOC --><a name="13-dynamic-programming"></a>
## 13. Dynamic Programming

```py
def find_max_knapsack_profit(capacity, weights, values):
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        for j in range(capacity, weights[i] - 1, -1):
            dp[j] = max(values[i] + dp[j - weights[i]], dp[j])
    return dp[capacity]

def calculate_min_coins(coins, remaining_amount, counter):
    if remaining_amount < 0:
        return -1 
    if remaining_amount == 0:
        return 0 
    if counter[remaining_amount - 1] != float('inf'):
        return counter[remaining_amount - 1]
    minimum = float('inf')
    for s in coins:
        result = calculate_min_coins(coins, remaining_amount - s, counter)
        if result >= 0 and result < minimum:
            minimum = 1 + result 
    counter[remaining_amount - 1] = minimum if minimum != float('inf') else -1 
    return counter[remaining_amount - 1]
def coin_change(coins, total):
    if total < 1:
        return 0 
    return calculate_min_coins(coins, total, [float('inf')] * total)

def find_tribonacci(n):
    if n < 3:
        return 1 if n else 0 
    first_num, second_num, third_num = 0, 1, 1 
    for _ in range(n - 2):
        first_num, second_num, third_num = second_num, third_num, first_num + second_num + third_num 
    return third_num

def can_partition_array(nums):
    array_sum = sum(nums)
    if array_sum % 2 != 0:
        return False 
    subset_sum = array_sum // 2 
    dp = [[False for i in range(len(nums) + 1)] for j in range(subset_sum + 1)]
    for i in range(0, len(nums) + 1):
        dp[0][i] = True 
    for i in range(1, subset_sum + 1):
        for j in range(1, len(nums) + 1):
            if nums[j - 1] > i:
                dp[i][j] = dp[i][j - 1]
            else:
                dp[i][j] = dp[i - nums[j - 1]][j - 1] or dp[i][j - 1]
    return dp[subset_sum][len(nums)]

def counting_bits(n):
    result = [0] * (n + 1)
    if n == 0:
        return result 
    result[0] = 0 
    result[1] = 1 
    for x in range(2, n + 1):
        if x % 2 == 0:
            result[x] = result[x // 2]
        else:
            result[x] = result[x // 2] + 1 
    return result 

def update_matrix(mat):
    m, n = len(mat), len(mat[0])
    for r in range(m):
        for c in range(n):
            if mat[r][c] > 0:
                above = mat[r - 1][c] if r > 0 else math.inf 
                left = mat[r][c - 1] if c > 0 else math.inf 
                mat[r][c] = min(above, left) + 1 
    for r in range(m - 1, -1, -1):
        for c in range(n - 1, -1, -1):
            if mat[r][c] > 0:
                below = mat[r + 1][c] if r < m - 1 else math.inf 
                right = mat[r][c + 1] if c < n - 1 else math.inf
                min_distance = min(below, right) + 1 
                mat[r][c] = min(mat[r][c], min_distance)
    return mat 

def house_robber(money):
    if len(money) == 0 or money is None:
        return 0 
    if len(money) == 1:
        return money[0]
    return max(house_robber_helper(money[:-1]), house_robber_helper(money[1:]))
def house_robber_helper(money):
    lookup_array = [0 for x in range(len(money) + 1)]
    lookup_array[0] = 0
    lookup_array[1] = money[0]
    for i in range(2, len(money)+1):
        lookup_array[i] = max(money[i-1] + lookup_array[i-2], lookup_array[i-1])
    return lookup_array[len(money)]

def max_product(nums):
    if len(nums) == 0:
        return 0 
    max_so_far = nums[0]
    min_so_far = nums[0]
    result = max_so_far 
    for i in range(1, len(nums)):
        curr = nums[i]
        prev_max_so_far = max_so_far 
        max_so_far = max(curr, max_so_far * curr, min_so_far * curr)
        min_so_far = min(curr, prev_max_so_far * curr, min_so_far * curr)
        result = max(max_so_far, result)
    return result 

def combination_sum(nums, target):
    dp = [[] for _ in range(target + 1)]
    dp[0].append([])
    for i in range(1, target + 1):
        for j in range(len(nums)):
            if nums[j] <= i:
                for prev in dp[i - nums[j]]:
                    temp = prev + [nums[j]]
                    temp.sort()
                    if temp not in dp[i]:
                        dp[i].append(temp)
    return dp[target]

def word_break(s, word_dict):
    n = len(s)
    word_set = set(word_dict)
    dp = [False] * (n + 1)
    dp[0] = True 
    dp[0] = True 
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True 
                break # if substring is found, no need to check further smaller substrings
    return dp[n]

def count_palindromic_substrings(s):
    count = 0 
    dp = [[False for i in range(len(s))] for i in range(len(s))]
    for i in range(len(s)):
        dp[i][i] = True 
        count += 1 
    for i in range(len(s) - 1):
        dp[i][i + 1] = (s[i] == s[i + 1])
        count += dp[i][i + 1]
    for length in range(3, len(s) + 1):
        i = 0 
        for j in range(length - 1, len(s)):
            dp[i][j] = dp[i + 1][j - 1] and (s[i] == s[j])
            count += dp[i][j]
            i += 1
    return count 

def longest_common_subsequence(str1, str2):
    n = len(str1)
    m = len(str2)
    dp = [[-1 for x in range(m)] for y in range(n)]
    return longest_common_subsequence_helper(str1, str2, 0, 0, dp)
def longest_common_subsequence_helper(str1, str2, i, j, dp):
    if i == len(str1) or j == len(str2):
        return 0 
    elif dp[i][j] == -1:
        if str1[i] == str2[j]:
            dp[i][j] = 1 + longest_common_subsequence_helper(str1, str2, i+1, j+1, dp)
        else:
            dp[i][j] = max(
                longest_common_subsequence_helper(str1, str2, i+1, j, dp),
                longest_common_subsequence_helper(str1, str2, i, j+1, dp)
            )
    return dp[i][j]

def word_break(s, word_dict):
    dp = [[]] * (len(s) + 1)
    dp[0] = [""]
    for i in range(1, len(s) + 1):
        prefix = s[:i]
        temp = []
        for j in range(0, i):
            suffix = prefix[j:]
            if suffix in word_dict:
                for substring in dp[j]:
                    # merge suffix with already calculated results
                    temp.append((substring + " " + suffix).strip())
        dp[i] = temp 
    return dp[len(s)] # all sentences formed from complete string s 

def num_of_decodings(decode_str):
    str_len = len(decode_str)
    dp = [0] * (str_len + 1)
    dp[0] = 1 
    if decode_str[0] != '0':
        dp[1] = 1 
    else:
        return 0 
    for i in range(2, str_len + 1):
        if decode_str[i - 1] != '0':
            dp[i] += dp[i - 1]
        if decode_str[i - 2] == '1' or (decode_str[i - 2] == '2' and decode_str[i - 1] <= '6'):
            dp[i] += dp[i - 2]
    return dp[str_len]

MOD = 10 ** 9 + 7 
def count_good_subsequences(s):
    N = len(s) + 1 
    factorials = [1] * N 
    inverses = [1] * N 
    for i in range(1, N):
        factorials[i] = factorials[i - 1] * i % MOD 
        inverses[i] = quick_modular_inverse(factorials[i], MOD - 2, MOD)
    frequency_count = [0] * 26 
    max_count = 1 
    for char in s:
        max_count = max(max_count, frequency_count[ord(char) - ord('a')] + 1)
        frequency_count[ord(char) - ord('a')] += 1 
    final_count = 0 
    for i in range(1, max_count + 1):
        count = 1 
        for j in range(26):
            if frequency_count[j] >= i:
                count = count * (combination(frequency_count[j], i, factorials, inverses))
        final_count = (final_count + count - 1) % MOD 
    return int(final_count)
def quick_modular_inverse(base, exponent, modulus):
    result = 1 
    while exponent != 0:
        if (exponent & 1) == 1:
            result = result * base % modulus 
        exponent >>= 1 
        base = base * base % modulus 
    return result 
def combination(n, k, factorials, inverses):
    return (factorials[n] * inverses[k] % MOD) * inverses[n - k] % MOD 
```

<!-- TOC --><a name="14-cyclic-sort"></a>
## 14. Cyclic Sort 

```py
def find_missing_number(nums):
    len_nums = len(nums)
    index = 0 
    while index < len_nums:
        value = nums[index]
        if value < len_nums and value != nums[value]:
            nums[index], nums[value] = nums[value], nums[index]
        else:
            index += 1 
    for x in range(len_nums):
        if x != nums[x]:
            return x 
    return len_nums 

def smallest_missing_positive_integer(nums):
    i = 0
    while i < len(nums):
        correct_spot = nums[i] - 1 
        if 0 <= correct_spot < len(nums) and nums[i] != nums[correct_spot]:
            nums[i], nums[correct_spot] = nums[correct_spot], nums[i]
        else:
            i += 1 
    for i in range(len(nums)):
        if i + 1 != nums[i]:
            return i + 1 
    return len(nums) + 1 

def find_corrupt_pair(nums):
    missing = None 
    duplicated = None 
    def swap(arr, first, second):
        arr[first], arr[second] = arr[second], arr[first]
    i = 0 
    while i < len(nums):
        correct = nums[i] - 1 
        if nums[i] != nums[correct]:
            swap(nums, i, correct)
        else:
            i += 1 
    for j in range(len(nums)):
        if nums[j] != j + 1:
            duplicated = nums[j]
            missing = j + 1 
    return [missing, duplicated]

def sort_array_by_parityII(nums):
    i, j = 0, 1 
    while i < len(nums) and j < len(nums):
        if nums[i] % 2 == 0:
            i += 2 
        elif nums[j] % 2 == 1:
            j += 2 
        else:
            nums[i], nums[j] = nums[j], nums[i]
            i += 2 
            j += 2
    return nums 
```

<!-- TOC --><a name="15-topological-sort"></a>
## 15. Topological Sort 

```py
from collections import deque 
def find_compilation_order(dependencies):
    sorted_order = []
    graph = {}
    inDegree = {}
    for x in dependencies:
        parent, child = x[1], x[0]
        graph[parent], graph[child] = [], []
        inDegree[parent], inDegree[child] = 0, 0
    if len(graph) <= 0:
        return sorted_order 
    for dependency in dependencies:
        parent, child = dependency[1], dependency[0]
        graph[parent].append(child)
        inDegree[child] += 1 
    sources = deque()
    for key in inDegree:
        if inDegree[key] == 0:
            sources.append(key)
    while sources:
        vertex = sources.popleft()
        sorted_order.append(vertex)
        for child in graph[vertex]:
            inDegree[child] -= 1 
            if inDegree[child] == 0:
                sources.append(child)
    if len(sorted_order) != len(graph):
        return []
    return sorted_order

from collections import defaultdict, Counter, deque 
def alien_order(words):
    adj_list = defaultdict(set)
    counts = Counter({c: 0 for word in words for c in word})
    for word1, word2 in zip(words, words[1:]):
        for c, d in zip(word1, word2):
            if c != d:
                if d not in adj_list[c]:
                    adj_list[c].add(d)
                    counts[d] += 1 
                break 
        else:
            if len(word2) < len(word1):
                return ""
    result = []
    sources_queue = deque([c for c in counts if counts[c] == 0])
    while sources_queue:
        c = sources_queue.popleft()
        result.append(c)
        for d in adj_list[c]:
            counts[d] -= 1 
            if counts[d] == 0:
                sources_queue.append(d)
    if len(result) < len(counts):
        return ""
    return "".join(result)

def verify_alien_dictionary(words, order):
    if len(words) == 1:
        return True 
    order_map = {}
    for index, val in enumerate(order):
        order_map[val] = index 
        for i in range(len(words) - 1):
            for j in range(len(words[i])):
                if j >= len(words[i + 1]):
                    return False 
                # if letters in same position in two words are different
                if words[i][j] != words[i + 1][j]:
                    # if rank of letter in the current word > rank in the same position in next word
                    if order_map[words[i][j]] > order_map[words[i + 1][j]]:
                        return False 
                    # if find first different character and they're sorted, no need to check remaining letters 
                    break
    return True  

def find_order(n, prerequisites):
    sorted_order = []
    if n <= 0:
        return sorted_order
    in_degree = {i: 0 for i in range(n)}
    graph = {i: [] for i in range(n)} # adjacency list graph 
    for prerequisite in prerequisites:
        parent, child = prerequisite[1], prerequisite[0]
        graph[parent].append(child)
        in_degree[child] += 1 
    sources = deque()
    for key in in_degree:
        if in_degree[key] == 0:
            sources.append(key)
    while sources:
        vertex = sources.popleft()
        sorted_order.append(vertex)
        for child in graph[vertex]:
            in_degree[child] -= 1 
            if in_degree[child] == 0:
                sources.append(child)
    # topological sort is not possible as graph has cycle
    if len(sorted_order) != n:
        return []
    return sorted_order

def can_finish(num_courses, prerequisites):
    counter = 0 
    if num_courses <= 0:
        return True 
    inDegree = {i: 0 for i in range(num_courses)}
    graph = {i: [] for i in range(num_courses)}
    for edge in prerequisites:
        parent, child = edge[1], edge[0]
        graph[parent].append(child)
        inDegree[child] += 1 
    sources = deque()
    for key in inDegree:
        if inDegree[key] == 0:
            sources.append(key)
    while sources:
        course = sources.popleft()
        counter += 1 
        for child in graph[course]:
            inDegree[child] -= 1 
            if inDegree[child] == 0:
                sources.append(child)
    return counter == num_courses

def build_matrix(k, row_conditions, col_conditions):
    order_rows = topological_sort(row_conditions, k)
    order_columns = topological_sort(col_conditions, k)
    if not order_rows or not order_columns:
        return []
    matrix = [[0] * k for _ in range(k)]
    pos_row = {num: i for i, num in enumerate(order_rows)}
    pos_col = {num: i for i, num in enumerate(order_columns)}
    for num in range(1, k + 1):
        if num in pos_row and num in pos_col:
            matrix[pos_row[num]][pos_col[num]] = num 
    return matrix 
def topological_sort(edges, n):
    adj = defaultdict(list)
    order = []
    # 0 = unvisited, 1 = visiting, 2 = processed
    visited = [0] * (n + 1)
    for x, y in edges:
        adj[x].append(y)
    for i in range(1, n + 1):
        if visited[i] == 0:
            if dfs(i, adj, visited, order): # cycle detected
                return []
    order.reverse()
    return order
def dfs(node, adj, visited, order):
    visited[node] = 1 
    for neighbor in adj[node]:
        if visited[neighbor] == 0:
            # of neighbor unvisited, dfs 
            if dfs(neighbor, adj, visited, order):
                return True # if cycle detected
        elif visited[neighbor] == 1: # if neighbor is being visited, cycle detected
            return True 
    visited[node] = 2 
    order.append(node)
    return False 

def longest_path(parent, s):
    n = len(parent)
    in_degree = [0] * n 
    for node in range(1, n):
        in_degree[parent[node]] += 1 
    queue = deque()
    longest_chains = [[0, 0] for _ in range(n)]
    longest_path = 1 
    for node in range(n):
        if in_degree[node] == 0:
            longest_chains[node][0] = 1 
            queue.append(node)
    while queue:
        current_node = queue.popleft()
        par = parent[current_node]
        # if node is not root, has a parent 
        if par != -1:
            longest_chain_from_current = longest_chains[current_node][0]
            if s[current_node] != s[par]:
                if longest_chain_from_current > longest_chains[par][0]:
                    longest_chains[par][1] = longest_chains[par][0]
                    longest_chains[par][0] = longest_chain_from_current
                elif longest_chain_from_current > longest_chains[par][1]:
                    longest_chains[par][1] = longest_chain_from_current
            longest_path = max(longest_path, longest_chains[par][0] + longest_chains[par][1] + 1)
            in_degree[par] -= 1 
            if in_degree[par] == 0:
                longest_chains[par][0] += 1 
                queue.append(par)
    return longest_path
```

<!-- TOC --><a name="16-sort-and-search"></a>
## 16. Sort and Search

```py
def find_distance_value(arr1, arr2, d):
    arr2.sort()
    distance = 0 
    for i in range(len(arr1)):
        left, right = 0, len(arr2) - 1 
        valid = True 
        while left <= right:
            mid = (left + right) // 2 
            if arr2[mid] == arr1[i]:
                valid = False 
                break 
            elif arr2[mid] < arr1[i]:
                left = mid + 1 
            else:
                right = mid - 1 
        if left < len(arr2) and abs(arr2[left] - arr1[i]) <= d:
            valid = False 
        if right >= 0 and abs(arr2[right] - arr1[i]) <= d:
            valid = False 
        if valid:
            distance += 1 
    return distance 

def answer_queries(nums, queries):
    nums.sort()
    prefix_sum = [0] * len(nums) 
    prefix_sum[0] = nums[0]
    for i in range(1, len(nums)):
        prefix_sum[i] = prefix_sum[i - 1] + nums[i]
    def binary_search(prefix_sum, target):
        low, high = 0, len(prefix_sum) - 1 
        while low <= high:
            mid = (low + high) // 2 
            if prefix_sum[mid] <= target:
                low = mid + 1 
            else:
                high = mid - 1 
        return low 
    
    answer = []
    for q in queries:
        index = binary_search(prefix_sum, q)
        answer.append(index)
    return answer 

def target_indices(nums, target):
    nums.sort()
    result = []
    for i in range(len(nums)):
        if nums[i] == target:
            result.append(i)
    return result 

def count_pairs(nums1, nums2):
    n = len(nums1)
    difference = [nums1[i] - nums2[i] for i in range(n)]
    difference.sort()
    count_of_valid_pairs = 0 
    for i in range(0, n):
        if difference[i] > 0:
            count_of_valid_pairs += n - i - 1 
        else:
            left = i + 1 
            right = n - 1 
            while left <= right:
                mid = (left + right) // 2 
                if difference[i] + difference[mid] > 0:
                    right = mid - 1 
                else:
                    left = mid + 1 
            count_of_valid_pairs += n - left 
    return count_of_valid_pairs

def triangle_number(nums):
    nums.sort()
    count = 0
    for i in range(len(nums) - 1, 1, -1):
        left, right = 0, i - 1 
        while left < right:
            if nums[left] + nums[right] > nums[i]:
                count += right - left 
                right -= 1 
            else:
                left += 1 
    return count 

def min_operations(nums, queries):
    nums.sort()
    n = len(nums)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]
    def binary_search(arr, target):
        low, high = 0, len(arr) - 1 
        while low <= high:
            mid = (low + high) // 2
            if arr[mid] < target: 
                low = mid + 1 
            else:
                high = mid -1 
        return low 
    answer = []
    for query in queries:
        idx = binary_search(nums, query)
        left_operations = query * idx - prefix[idx]
        right_operations = (prefix[n] - prefix[idx]) - query * (n - idx)
        answer.append(left_operations + right_operations)
    return answer 

def find_best_value(arr, target):
    arr.sort()
    n = len(arr)
    remaining_target = target 
    for i, num in enumerate(arr):
        if remaining_target <= num * (n - i):
            replacement_value = remaining_target / (n - i)
            if replacement_value - int(replacement_value) == 0.5:
                return int(replacement_value)
            return round(replacement_value)
        remaining_target -= num 
    return arr[-1]

def range_sum(nums, n, left, right):
    mod = 10 ** 9 + 7 
    result = (
        sum_of_first_k(nums, n, right) - sum_of_first_k(nums, n, left- 1)
    ) % mod 
    return (result + mod) % mod 
def sum_of_first_k(nums, n, k):
    min_sum = min(nums)
    max_sum = sum(nums)
    T_left = min_sum 
    T_right = max_sum 
    while T_left <= T_right:
        mid = T_left + (T_right - T_left) // 2 
        if count_and_sum(nums, n, mid)[0] >= k:
            T_right = mid - 1 
        else:
            T_left = mid + 1 
    count, total_sum = count_and_sum(nums, n, T_left)
    return total_sum - T_left * (count - k)
def count_and_sum(nums, n, target):
    count = 0
    current_sum = 0 
    total_sum = 0 
    window_sum = 0 
    i = 0 
    for j in range(n):
        current_sum += nums[j]
        window_sum += nums[j] * (j - i + 1)
        while current_sum > target:
            window_sum -= current_sum 
            current_sum -= nums[i]
            i += 1 
        count += j - i + 1 
        total_sum += window_sum
    return count, total_sum

def can_place_balls(x, position, m):
    prev = position[0]
    balls = 1 
    for i in range(1, len(position)):
        curr = position[i]
        if curr - prev >= x:
            balls += 1 
            prev = curr 
            if balls == m:
                return True 
    return False
def max_distance(position, m):
    position.sort()
    force = 0
    low = 1 
    high = (position[-1] - position[0]) // (m - 1)
    while low <= high:
        mid = (low + high) // 2 
        if can_place_balls(mid, position, m):
            force = mid 
            low = mid + 1 
        else:
            high = mid - 1 
    return force 

def count_pairs_with_distance(nums, d):
    count = 0 
    left = 0
    for right in range(len(nums)):
        while nums[right] - nums[left] > d:
            left += 1 
        count += right - left 
    return count 
def smallest_distance_pair(nums, k):
    nums.sort()
    low = 0 
    high = nums[-1] - nums[0]
    while low < high:
        mid = (low + high) // 2 
        count = count_pairs_with_distance(nums, mid)
        if count < k: 
            low = mid + 1 
        else:
            high = mid 
    return low 

def binary_search(array, target, start):
    low, high = start, len(array) - 1 
    while low <= high:
        mid = (low + high) // 2 
        if array[mid] <= target:
            low = mid + 1 
        else:
            high = mid - 1
    return low 
def min_wasted_space(packages, boxes):
    MOD = 10 ** 9 + 7 
    packages.sort()
    total_package_size = sum(packages)
    min_waste = float('inf')
    for box_sizes in boxes:
        box_sizes.sort()
        if box_sizes[-1] < packages[-1]:
            continue 
        total_space_used = 0 
        start_index = 0
        for box_size in box_sizes:
            end_index = binary_search(packages, box_size, start_index)
            num_packages = end_index - start_index 
            total_space_used += box_size * num_packages 
            start_index = end_index
        min_waste = min(min_waste, total_space_used - total_package_size)
    return (min_waste) % MOD if min_waste != float('inf') else -1 

def find_position(lis, height):
    left, right = 0, len(lis) - 1 
    while left <= right:
        mid = (left + right) // 2 
        if lis[mid] < height:
            left = mid + 1 
        else:
            right = mid - 1 
    return left 
def max_envelopes(envelopes):
    # sort by width in ascending order, if widths are same, sort by height in descending order
    envelopes.sort(key=lambda x: (x[0], -x[1]))
    lis = []
    for width, height in envelopes:
        position = find_position(lis, height)
        if position == len(lis):
            lis.append(height)
        else:
            lis[position] = height 
    return len(lis)
```

<!-- TOC --><a name="17-matrices"></a>
## 17. Matrices

```py
def find_distance_value(arr1, arr2, d):
    arr2.sort()
    distance = 0 
    for i in range(len(arr1)):
        left, right = 0, len(arr2) - 1 
        valid = True 
        while left <= right:
            mid = (left + right) // 2 
            if arr2[mid] == arr1[i]:
                valid = False 
                break 
            elif arr2[mid] < arr1[i]:
                left = mid + 1 
            else:
                right = mid - 1 
        if left < len(arr2) and abs(arr2[left] - arr1[i]) <= d:
            valid = False 
        if right >= 0 and abs(arr2[right] - arr1[i]) <= d:
            valid = False 
        if valid:
            distance += 1 
    return distance 

def answer_queries(nums, queries):
    nums.sort()
    prefix_sum = [0] * len(nums) 
    prefix_sum[0] = nums[0]
    for i in range(1, len(nums)):
        prefix_sum[i] = prefix_sum[i - 1] + nums[i]
    def binary_search(prefix_sum, target):
        low, high = 0, len(prefix_sum) - 1 
        while low <= high:
            mid = (low + high) // 2 
            if prefix_sum[mid] <= target:
                low = mid + 1 
            else:
                high = mid - 1 
        return low 
    
    answer = []
    for q in queries:
        index = binary_search(prefix_sum, q)
        answer.append(index)
    return answer 

def target_indices(nums, target):
    nums.sort()
    result = []
    for i in range(len(nums)):
        if nums[i] == target:
            result.append(i)
    return result 

def count_pairs(nums1, nums2):
    n = len(nums1)
    difference = [nums1[i] - nums2[i] for i in range(n)]
    difference.sort()
    count_of_valid_pairs = 0 
    for i in range(0, n):
        if difference[i] > 0:
            count_of_valid_pairs += n - i - 1 
        else:
            left = i + 1 
            right = n - 1 
            while left <= right:
                mid = (left + right) // 2 
                if difference[i] + difference[mid] > 0:
                    right = mid - 1 
                else:
                    left = mid + 1 
            count_of_valid_pairs += n - left 
    return count_of_valid_pairs

def triangle_number(nums):
    nums.sort()
    count = 0
    for i in range(len(nums) - 1, 1, -1):
        left, right = 0, i - 1 
        while left < right:
            if nums[left] + nums[right] > nums[i]:
                count += right - left 
                right -= 1 
            else:
                left += 1 
    return count 

def min_operations(nums, queries):
    nums.sort()
    n = len(nums)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]
    def binary_search(arr, target):
        low, high = 0, len(arr) - 1 
        while low <= high:
            mid = (low + high) // 2
            if arr[mid] < target: 
                low = mid + 1 
            else:
                high = mid -1 
        return low 
    answer = []
    for query in queries:
        idx = binary_search(nums, query)
        left_operations = query * idx - prefix[idx]
        right_operations = (prefix[n] - prefix[idx]) - query * (n - idx)
        answer.append(left_operations + right_operations)
    return answer 

def find_best_value(arr, target):
    arr.sort()
    n = len(arr)
    remaining_target = target 
    for i, num in enumerate(arr):
        if remaining_target <= num * (n - i):
            replacement_value = remaining_target / (n - i)
            if replacement_value - int(replacement_value) == 0.5:
                return int(replacement_value)
            return round(replacement_value)
        remaining_target -= num 
    return arr[-1]

def range_sum(nums, n, left, right):
    mod = 10 ** 9 + 7 
    result = (
        sum_of_first_k(nums, n, right) - sum_of_first_k(nums, n, left- 1)
    ) % mod 
    return (result + mod) % mod 
def sum_of_first_k(nums, n, k):
    min_sum = min(nums)
    max_sum = sum(nums)
    T_left = min_sum 
    T_right = max_sum 
    while T_left <= T_right:
        mid = T_left + (T_right - T_left) // 2 
        if count_and_sum(nums, n, mid)[0] >= k:
            T_right = mid - 1 
        else:
            T_left = mid + 1 
    count, total_sum = count_and_sum(nums, n, T_left)
    return total_sum - T_left * (count - k)
def count_and_sum(nums, n, target):
    count = 0
    current_sum = 0 
    total_sum = 0 
    window_sum = 0 
    i = 0 
    for j in range(n):
        current_sum += nums[j]
        window_sum += nums[j] * (j - i + 1)
        while current_sum > target:
            window_sum -= current_sum 
            current_sum -= nums[i]
            i += 1 
        count += j - i + 1 
        total_sum += window_sum
    return count, total_sum

def can_place_balls(x, position, m):
    prev = position[0]
    balls = 1 
    for i in range(1, len(position)):
        curr = position[i]
        if curr - prev >= x:
            balls += 1 
            prev = curr 
            if balls == m:
                return True 
    return False
def max_distance(position, m):
    position.sort()
    force = 0
    low = 1 
    high = (position[-1] - position[0]) // (m - 1)
    while low <= high:
        mid = (low + high) // 2 
        if can_place_balls(mid, position, m):
            force = mid 
            low = mid + 1 
        else:
            high = mid - 1 
    return force 

def count_pairs_with_distance(nums, d):
    count = 0 
    left = 0
    for right in range(len(nums)):
        while nums[right] - nums[left] > d:
            left += 1 
        count += right - left 
    return count 
def smallest_distance_pair(nums, k):
    nums.sort()
    low = 0 
    high = nums[-1] - nums[0]
    while low < high:
        mid = (low + high) // 2 
        count = count_pairs_with_distance(nums, mid)
        if count < k: 
            low = mid + 1 
        else:
            high = mid 
    return low 

def binary_search(array, target, start):
    low, high = start, len(array) - 1 
    while low <= high:
        mid = (low + high) // 2 
        if array[mid] <= target:
            low = mid + 1 
        else:
            high = mid - 1
    return low 
def min_wasted_space(packages, boxes):
    MOD = 10 ** 9 + 7 
    packages.sort()
    total_package_size = sum(packages)
    min_waste = float('inf')
    for box_sizes in boxes:
        box_sizes.sort()
        if box_sizes[-1] < packages[-1]:
            continue 
        total_space_used = 0 
        start_index = 0
        for box_size in box_sizes:
            end_index = binary_search(packages, box_size, start_index)
            num_packages = end_index - start_index 
            total_space_used += box_size * num_packages 
            start_index = end_index
        min_waste = min(min_waste, total_space_used - total_package_size)
    return (min_waste) % MOD if min_waste != float('inf') else -1 

def find_position(lis, height):
    left, right = 0, len(lis) - 1 
    while left <= right:
        mid = (left + right) // 2 
        if lis[mid] < height:
            left = mid + 1 
        else:
            right = mid - 1 
    return left 
def max_envelopes(envelopes):
    # sort by width in ascending order, if widths are same, sort by height in descending order
    envelopes.sort(key=lambda x: (x[0], -x[1]))
    lis = []
    for width, height in envelopes:
        position = find_position(lis, height)
        if position == len(lis):
            lis.append(height)
        else:
            lis[position] = height 
    return len(lis)

def set_matrix_zeros(mat):
    rows = len(mat)
    cols = len(mat[0])
    fcol = False 
    frow = False 
    for i in range(rows):
        if mat[i][0] == 0:
            fcol = True 
    for i in range(cols):
        if mat[0][i] == 0:
            frow = True 
    for i in range(1, rows):
        for j in range(1, cols):
            if mat[i][j] == 0:
                mat[0][j] = mat[i][0] = 0
    for i in range(1, cols):
        if mat[0][j] == 0:
            for i in range(1, rows):
                mat[i][j] = 0 
    if fcol:
        for i in range(rows):
            mat[i][0] = 0
    if frow:
        for j in range(cols):
            mat[0][j] = 0 
    return mat 

def rotate_image(matrix):
    n = len(matrix)
    for row in range(n // 2):
        for col in range(row, n - row - 1):
            matrix[row][col], matrix[col][n - 1 - row] = matrix[col][n - 1 - row], matrix[row][col]
            matrix[row][col], matrix[n - 1 - row][n - 1 - col] = matrix[n - 1 - row][n - 1 - col], matrix[row][col]
            matrix[row][col], matrix[n - 1- col][row] = matrix[n - 1- col][row], matrix[row][col]
    return matrix 

def spiral_order(matrix):
    rows, cols = len(matrix), len(matrix[0])
    row, col = 0, -1 
    direction = 1 
    result = []
    while rows > 0 and cols > 0:
        for _ in range(cols):
            col += direction 
            result.append(matrix[row][col])
        row -= 1 
        for _ in range(rows):
            row += direction 
            result.append(matrix[row][col])
        cols -= 1 
        direction *= -1 
    return result 

def find_exit_column(grid):
    result = [-1] * len(grid[0])
    for col in range(len(grid[0])):
        current_col = col 
        for row in range(len(grid)):
            next_col = current_col + grid[row][current_col]
            if next_col < 0 or next_col > len(grid[0]) - 1 or grid[row][current_col] != grid[row][next_col]:
                break 
            if row == len(grid) - 1:
                result[col] = next_col
            current_col = next_col 
    return result 

def transpose_matrix(matrix):
    rows = len(matrix)
    columns = len(matrix[0])
    result = [[0] * rows for _ in range(columns)]
    for i in range(rows):
        for j in range(columns):
            result[j][i] = matrix[i][j]
    return result 

def count_negatives(grid):
    count = 0 
    n = len(grid[0])
    current_index = n - 1 
    for row in grid:
        while current_index >= 0 and row[current_index] < 0:
            current_index -= 1 
        count += (n - (current_index + 1))
    return count 

import collections
def minimum_seconds(land):
    m, n = len(land), len(land[0])
    flood = collections.deque()
    moves = collections.deque()
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    seconds = 0 
    for i in range(m):
        for j in range(n):
            if land[i][j] == "S":
                moves.append((i, j))
            if land[i][j] == "*":
                flood.append((i, j))
    while moves:
        spread, move = len(flood), len(moves)
        for _ in range(spread):
            flood_x, flood_y = flood.popleft()
            for x, y in directions:
                new_x, new_y = flood_x + x, flood_y + y 
                if 0 <= new_x < m and 0 <= new_y < n and land[new_x][new_y] == ".":
                    land[new_x][new_y] = "*"
                    flood.append((new_x, new_y))
        for _ in range(move):
            move_x, move_y = moves.popleft()
            if land[move_x][move_y] == "D":
                return seconds 
            for x, y in directions:
                new_x, new_y = move_x + x, move_y + y 
                if 0 <= new_x < m and 0 <= new_y < n and (land[new_x][new_y] == "." or land[new_x][new_y] == "D"):
                    if land[new_x][new_y] != "D":
                        land[new_x][new_y] = "*"
                    moves.append((new_x, new_y))
        seconds += 1 
    return -1 

def min_area(image, x, y):
    def binary_search(image, low, high, check_func):
        while low < high:
            mid = (low + high) // 2 
            if check_func(mid):
                high = mid 
            else:
                low = mid + 1 
        return low 
    def contains_black_pixel_in_column(mid):
        return any(image[i][mid] == '1' for i in range(len(image)))
    def contains_black_pixel_in_row(mid):
        return '1' in image[mid]
    left = binary_search(image, 0, y, contains_black_pixel_in_column)
    right = binary_search(image, y + 1, len(image[0]), lambda mid: not contains_black_pixel_in_column(mid)) - 1 
    top = binary_search(image, 0, x, contains_black_pixel_in_row)
    bottom = binary_search(image, x + 1, len(image), lambda mid: not contains_black_pixel_in_row(mid)) -1 
    return (right - left + 1) * (bottom - top + 1)

def island_perimeter(grid):
    rows = len(grid)
    cols = len(grid[0])
    perimeter = 0 
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                perimeter += 4
                if r > 0 and grid[r - 1][c] == 1:
                    perimeter -=2 
                if c > 0 and grid[r][c - 1] == 1:
                    perimeter -= 2 
    return perimeter

def construct_2D_array(original, m, n):
    if m * n != len(original):
        return []
    result = [[0] * n for _ in range(m)]
    index = 0 
    for i in range(m):
        for j in range(n):
            result[i][j] = original[index]
            index += 1 
    return result 

def generate_matrix(n):
    matrix = [[0] * n for _ in range(n)]
    count = 1 
    for layer in range((n + 1) // 2):
        for i in range(layer, n - layer):
            matrix[layer][i] = count 
            count += 1
        for i in range(layer + 1, n - layer):
            matrix[i][n - layer - 1] = count 
            count += 1 
        for i in range(n - layer - 2, layer - 1, -1):
            matrix[n - layer - 1][i] = count 
            count += 1 
        for i in range(n - layer - 2, layer, -1):
            matrix[i][layer] = count 
            count += 1
    return matrix

from collections import defaultdict
def max_equal_rows_after_flips(matrix):
    frequencies = defaultdict(int)
    for row in matrix:
        pattern = ""
        for col in range(len(row)):
            if row[0] == row[col]:
                pattern += "T"
            else:
                pattern += "F"
        frequencies[pattern] += 1 
    res = 0
    for count in frequencies.values():
        res = max(count, res)
    return res 

def number_of_clean_rooms(room):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    rows, cols = len(room), len(room[0])
    cleaned = 0 
    curr_row, curr_col, direction = 0, 0, 0
    while not room[curr_row][curr_col] >> (direction + 1) & 1:
        if room[curr_row][curr_col] == 0:
            cleaned += 1 
        room[curr_row][curr_col] |= 1 << (direction + 1)
        next_row = curr_row + directions[direction][0]
        next_col = curr_col + directions[direction][1]
        if 0 <= next_row < rows and 0 <= next_col < cols and room[next_row][next_col] != 1:
            curr_row, curr_col = next_row, next_col 
        else:
            direction = (direction + 1) % 4 
    return cleaned 
```

<!-- TOC --><a name="18-stacks"></a>
## 18. Stacks

```py
def calculator(expression):
    number = 0 
    sign_value = 1 
    result = 0 
    operations_stack = []
    for c in expression:
        if c.isdigit():
            number = number * 10 + int(c)
        if c in "+-":
            result += number * sign_value 
            sign_value = -1 if c == '-' else 1 
            number = 0
        elif c == '(':
            operations_stack.append(result)
            operations_stack.append(sign_value)
            result = 0 
            sign_value = 1 
        elif c == ')':
            result += sign_value * number 
            pop_sign_value = operations_stack.pop()
            result *= pop_sign_value
            second_value = operations_stack.pop()
            result += second_value 
            number = 0
    return result + number * sign_value

def remove_duplicates(s):
    stack = []
    for char in s:
        if stack and stack[-1] == char:
            stack.pop()
        else:
            stack.append(char)
    return "".join(stack)

def min_remove_parentheses(s):
    stack = []
    s_list = list(s)
    for i, val in enumerate(s):
        if len(stack) > 0 and stack[-1][0] == '(' and val == ')':
            stack.pop()
        elif val == '(' or val == ')':
            stack.append([val, i])
    for p in stack:
        s_list[p[1]] = ""
    result = ''.join(s_list)
    return result 

from logs import * 
def exclusive_time(n, logs):
    logs_stack = []
    result = [0] * n 
    for content in logs:
        logs = Log(content)
        if logs.is_start:
            logs_stack.append(logs)
        else:
            top = logs_stack.pop()
            result[top.id] += (logs.time - top.time + 1)
            if logs_stack:
                result[logs_stack[-1].id] -= (logs.time - top.time + 1)
    return result 

from nested_integers import NestedIntegers 
class NestedIterator:
    def __init__(self, nested_list):
        self.nested_list_stack = list(reversed([NestedIntegers(val) for val in nested_list]))
    def has_next(self):
        while len(self.nested_list_stack) > 0:
            top = self.nested_list_stack[-1]
            if top.is_integer():
                return True 
            top_list = self.nested_list_stack.pop().get_list()
            i = len(top_list) - 1 
            while i >= 0:
                self.nested_list_stack.append(top_list[i])
                i -= 1 
        return False 
    def next(self):
        if self.has_next():
            return self.nested_list_stack.pop().get_integer()
        return None 
    
from stack import Stack
class MyQueue(object):
    def __init__(self):
        self.stack1 = Stack()
        self.stack2 = Stack()
    def push(self, x):
        while not self.stack1.is_empty():
            self.stack2.push(self.stack1.pop())
        self.stack1.push(x)
        while not self.stack2.is_empty():
            self.stack1.push(self.stack2.pop())
    def pop(self):
        return self.stack1.pop()
    def peek(self):
        return self.stack1.top()
    def empty(self):
        return self.stack1.is_empty()
        
def daily_temperatures(temperatures):
    n = len(temperatures)
    output = [0] * n 
    stack = []
    for i in range(n):
        while stack and temperatures[i] > temperatures[stack[-1]]:
            j = stack.pop()
            output[j] = i - j 
        stack.append(i)
    return output 

def decode_string(s):
    count_stack = []
    string_stack = []
    current = ""
    k = 0 
    for char in s:
        if char.isdigit():
            k = 10 * k + int(char)
        elif char == "[":
            count_stack.append(k)
            string_stack.append(current)
            k = 0 
            current = ""
        elif char == "]":
            popped_string = string_stack.pop()
            num = count_stack.pop()
            current = popped_string + current * num 
        else:
            current += char 
    return current 

def min_length(s):
    stack = []
    for current_char in s:
        if not stack:
            stack.append(current_char)
            continue 
        if current_char == "B" and stack[-1] == "A":
            stack.pop()
        elif current_char == "D" and stack[-1] == "C":
            stack.pop()
        else:
            stack.append(current_char)
    return len(stack)
```

<!-- TOC --><a name="19-graphs"></a>
## 19. Graphs

```py
from queue import PriorityQueue
from collections import defaultdict
def network_delay_time(times, n, k):
    adjacency = defaultdict(list)
    for src, dst, t in times:
        adjacency[src].append((dst, t))
    pq = PriorityQueue()
    pq.put((0, k))
    visited = set()
    delays = 0
    while not pq.empty():
        time, node = pq.get()
        if node in visited:
            continue 
        visited.add(node)
        delays = max(delays, time)
        neighbors = adjacency[node]
        for neighbor in neighbors:
            neighbor_node, neighbor_time = neighbor 
            if neighbor_node not in visited:
                new_time = time + neighbor_time 
                pq.put((new_time, neighbor_node))
    if len(visited) == n:
        return delays 
    return -1 

def number_of_paths(n, corridors):
    neighbors = defaultdict(set)
    cycles = 0 
    for room1, room2 in corridors:
        neighbors[room1].add(room2)
        neighbors[room2].add(room1)
        cycles += len(neighbors[room1].intersection(neighbors[room2]))
    
from graph_utility import *
def clone_helper(root, nodes_completed):
    if root == None:
        return None 
    cloned_node = Node(root.data)
    nodes_completed[root] = cloned_node 
    for p in root.neighbors:
        x = nodes_completed.get(p)
        if not x:
            cloned_node.neighbors += [clone_helper(p, nodes_completed)]
        else:
            cloned_node.neighbors += [x]
    return cloned_node
def clone(root):
    nodes_completed = {}
    return clone_helper(root, nodes_completed)

def valid_tree(n, edges):
    if len(edges) != n - 1:
        return False 
    adjacency = [[] for _ in range(n)]
    for x, y in edges:
        adjacency[x].append(y)
        adjacency[y].append(x)
    visited = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return len(visited) == n 

def min_buses(routes, src, dest):
    adj_list = {}
    for i, stations in enumerate(routes):
        for station in stations:
            if station not in adj_list:
                adj_list[station] = []
            adj_list[station].append(i)
    queue = deque()
    queue.append([src, 0])
    visited_buses = set()
    while queue:
        station, buses_taken = queue.popleft()
        if station == dest:
            return buses_taken
        if station in adj_list:
            for bus in adj_list[station]:
                if bus not in visited_buses:
                    for s in routes[bus]:
                        queue.append([s, buses_taken + 1])
                    visited_buses.add(bus)
    return -1 

def find_itinerary(tickets):
    flight_map = defaultdict(list)
    result = []
    for departure, arrival in tickets:
        flight_map[departure].append(arrival)
    for departure in flight_map:
        flight_map[departure].sort(reverse=True)
    dfs_traversal('JFK', flight_map, result)
    return result[::-1]
def dfs_traversal(current, flight_map, result):
    destinations = flight_map[current]
    while destinations:
        next_destination = destinations.pop()
        dfs_traversal(next_destination, flight_map, result)
    result.append(current)

def find_judge(n, trust):
    if len(trust) < n - 1:
        return -1 
    indegree = [0] * (n + 1)
    outdegree = [0] * (n + 1)
    for a, b in trust:
        outdegree[a] += 1 
        indegree[b] += 1 
    for i in range(1, n + 1):
        if indegree[i] == n - 1 and outdegree[i] == 0:
            return i
    return -1 

def find_center(edges):
    first = edges[0]
    second = edges[1]
    if first[0] in second:
        return first[0]
    else:
        return first[1]
    
def lucky_numbers(matrix):
    M, N = len(matrix), len(matrix[0])
    r_largest_min = float('-inf')
    for i in range(N):
        r_min = min(matrix[i])
        r_largest_min = max(r_largest_min, r_min)
    c_smallest_max = float('inf')
    for i in range(M):
        c_max = max(matrix[j][i] for j in range(N))
        c_smallest_max = min(c_smallest_max, c_max)
    if r_largest_min == c_smallest_max:
        return [r_largest_min]
    else:
        return []
    
def max_probability(n, edges, succProb, start, end):
    graph = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        graph[u].append((v, succProb[i]))
        graph[v].append((u, succProb[i]))
    max_prob = [0.0] * n 
    max_prob[start] = 1.0 
    pq = [(-1.0, start)]
    while pq:
        cur_prob, cur_node = heapq.heappop(pq)
        if cur_node == end:
            return -cur_prob 
        if graph[cur_node]:
            for neighbor, edge_prob in graph[cur_node]:
                new_prob = -cur_prob * edge_prob 
                if new_prob > max_prob[neighbor]:
                    max_prob[neighbor] = new_prob
                    heapq.heappush(pq, (-new_prob, neighbor))
            # clear current node's list of neighbors to prevent revisiting it 
            graph[cur_node].clear() 
    return 0.0
```

<!-- TOC --><a name="20-tree-dfs"></a>
## 20. Tree DFS

```py
from TreeNode import * 
from BinaryTree import * 
def flatten_tree(root):
    if not root:
        return 
    current = root 
    while current:
        if current.left:
            last = current.left 
            while last.right:
                last = last.right 
            last.right = current.right
            current.right = current.left 
            current.left = None 
        current = current.right 
    return root 

def diameter_helper(node, diameter):
    if node is None:
        return 0, diameter 
    else:
        lh, diameter = diameter_helper(node.left, diameter)
        rh, diameter = diameter_helper(node.right, diameter)
        diameter = max(diameter, lh + rh)
        return max(lh, rh) + 1, diameter 
def diameter_of_binaryTree(root):
    diameter = 0 
    if not root:
        return 0 
    _, diameter = diameter_helper(root, diameter)
    return diameter 

from BinaryTree import * 
from TreeNode import * 
MARKER  = "M"
m = 1 
def serialize_rec(node, stream):
    global m 
    if node is None:
        stream.append(MARKER + str(m))
        m += 1 
        return 
    stream.append(node.data)
    serialize_rec(node.left, stream)
    serialize_rec(node.right, stream)
def serialize(root):
    stream = []
    serialize_rec(root, stream)
    return stream 
def deserialize_helper(stream):
    val = stream.pop()
    if type(val) is str and val[0] == MARKER:
        return None
    node = TreeNode(val)
    node.left = deserialize_helper(stream)
    node.right = deserialize_helper(stream)
    return node 
def deserialize(stream):
    stream.reverse()
    node = deserialize_helper(stream)
    return node 

from BinaryTree import * 
from TreeNode import * 
change = 0 
master_root = None 
def mirror_binary_tree(root):
    global change, master_root
    if not root:
        return None 
    if root.left:
        mirror_binary_tree(root.left)
    if root.right:
        mirror_binary_tree(root.right)
    root.left, root.right = root.right, root.left 
    if master_root and (root.left or root.right):
        change += 1 
        display_tree(master_root)
    return root 

global max_sum 
def max_contrib(root):
    global max_sum 
    if not root:
        return 0 
    max_left = max_contrib(root.left)
    max_right = max_contrib(root.right)
    left_subtree = 0 
    right_subtree = 0 
    if max_left > 0:
        left_subtree = max_left 
    if max_right > 0:
        right_subtree = max_right
    value_new_path = root.data + left_subtree + right_subtree
    max_sum = max(max_sum, value_new_path)
    return root.data + max(left_subtree, right_subtree)
def max_path_sum(root):
    global max_sum 
    max_sum = float('-inf')
    max_contrib(root)
    return max_sum 

from TreeNode import *
def sorted_array_to_bst_helper(nums, low, high):
    if (low > high):
        return None 
    mid = low + (high - low) // 2 
    root = TreeNode(nums[mid])
    root.left = sorted_array_to_bst_helper(nums, low, mid - 1)
    root.right = sorted_array_to_bst_helper(nums, mid + 1, high)
    return root 
def sorted_array_to_bst(nums):
    return sorted_array_to_bst_helper(nums, 0, len(nums) - 1)

def build_tree_helper(p_order, i_order, left, right, mapping, p_index):
    if left > right:
        return None 
    curr = p_order[p_index[0]]
    p_index[0] += 1 
    root = TreeNode(curr)
    if left == right:
        return root 
    in_index = mapping[curr]
    root.left = build_tree_helper(p_order, i_order, left, in_index - 1, mapping, p_index)
    root.right = build_tree_helper(p_order, i_order, in_index + 1, right, mapping, p_index)
    return root 
def build_tree(p_order, i_order):
    p_index = [0]
    mapping = {}
    for i in range(len(p_order)):
        mapping[i_order[i]] = i 
    return build_tree_helper(p_order, i_order, 0, len(p_order) - 1, mapping, p_index)

from TreeNode import * 
from BinaryTree import *
def righ_side_view(root):
    if root is None:
        return []
    rside = []
    dfs(root, 0, rside)
    return rside
def dfs(node, level, rside):
    if level == len(rside):
        rside.append(node.data)
    for child in [node.right, node.left]:
        if child: # recursively calling dfs on child node 
            dfs(child, level + 1, rside)

class Solution:
    def __init__(self):
        self.lca = None 
    def lowest_common_ancestor(self, root, p, q):
        self.lowest_common_ancestor_rec(root, p, q)
        return self.lca 
    def lowest_common_ancestor_rec(self, current_node, p, q):
        if not current_node:
            return False 
        left, right, mid = False, False, False 
        if p == current_node or q == current_node:
            mid = True 
        left = self.lowest_common_ancestor_rec(current_node.right, p, q)
        if not self.lca:
            right = self.lowest_common_ancestor_rec(current_node.right, p, q)
        if mid + left + right >= 2:
            self.lca = current_node 
        return mid or left or right 
    
import math 
def validate_bst(root):
    prev = [-math.inf]
    return validate_bst_helper(root, prev)
def validate_bst_helper(root, prev):
    if not root:
        return True 
    if not validate_bst_helper(root.left, prev):
        return False 
    if root.data <= prev[0]:
        return False 
    prev[0] = root.data 
    return validate_bst_helper(root.right, prev)

from CreateNestedList import * 
def find_max_depth(nested_list):
    max_depth = 0
    for obj in nested_list:
        if not obj.is_integer() and len(obj.get_list()) > 0:
            max_depth = max(max_depth, 1 + find_max_depth(obj.get_list()))
    return max_depth 
def weighted_depth_sum_rec(nested_list, depth, max_depth):
    result = 0 
    for obj in nested_list:
        if obj.is_integer():
            result += obj.get_integer() * (max_depth - depth + 1)
        else:
            result += weighted_depth_sum_rec(obj.get_list(), depth + 1, max_depth)
    return result 
def weighted_depth_sum(nested_list):
    max_depth = find_max_depth(nested_list)
    return weighted_depth_sum_rec(nested_list, 0, max_depth)

from BinarySearchTree import * 
def inorder_successor(root, p):
    successor = None 
    while root:
        if p.data >= root.data:
            root = root.right 
        else:
            successor = root 
            root = root.left 
    return successor

import collections 
def tree_dfs(node, depth, nodeDepth, nodeHeight):
    if not node:
        return -1 
    nodeDepth[node.data] = depth 
    height = max(tree_dfs(node.left, depth + 1, nodeDepth, nodeHeight), tree_dfs(node.right, depth + 1, nodeDepth, nodeHeight)) + 1
    nodeHeight[node.data] = height 
    return height 
def heights_after_queries(root, queries):
    nodeDepth, nodeHeight = {}, {}
    tree_dfs(root, 0, nodeDepth, nodeHeight)
    depthGroups = collections.defaultdict(list) # group nodes by their depth, keep top 2 heights
    for value, depth in nodeDepth.items():
        depthGroups[depth].append((nodeHeight[value], value))
        depthGroups[depth].sort(reverse=True)
        if len(depthGroups[depth]) > 2:
            depthGroups[depth].pop()
    result = []
    for q in queries:
        depth = nodeDepth[q]
        if len(depthGroups[depth]) == 1:
            result.append(depth - 1)
        elif depthGroups[depth][0][1] == q:
            result.append(depthGroups[depth][1][0] + depth)
        else:
            result.append(depthGroups[depth][0][0] + depth)
    return result 

def return_forest(root, delete_nodes):
    if not root:
        return []
    to_delete = set(delete_nodes)
    forest = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.left:
            stack.append(node.left)
            if node.left.data in to_delete:
                node.left = None 
        if node.right:
            stack.append(node.right)
            if node.right.data in to_delete:
                node.right = None 
        if node.data in to_delete:
            if node.left:
                forest.append(node.left)
            if node.right:
                forest.append(node.right)
    if root.data not in to_delete:
        forest.append(root)
    return forest
```

<!-- TOC --><a name="21-tree-bfs"></a>
## 21. Tree BFS 

```py
from collections import deque 
from BinaryTree import * 
from TreeNode import * 
def level_order_traversal(root):
    result = []
    if not root:
        result = "None"
        return result 
    current_queue = deque()
    current_queue.append(root)
    while current_queue:
        level_size = len(current_queue)
        level_nodes = []
        for _ in range(level_size):
            temp = current_queue.popleft()
            level_nodes.append(str(temp.data))
            if temp.left:
                current_queue.append(temp.left)
            if temp.right:
                current_queue.append(temp.right)
        result.append(", ".join(level_nodes))
    return " : ".join(result)

def zigzag_level_order(root):
    if root is None:
        return []
    results = []
    dq = deque([root])
    reverse = False 
    while len(dq):
        size = len(dq)
        results.insert(len(results), [])
        for i in range(size):
            if not reverse:
                node = dq.popleft()
                results[len(results) - 1].append(node.dta)
                if node.left:
                    dq.append(node.left)
                if node.right:
                    dq.append(node.right)
            else:
                node = dq.pop()
                results[len(results) - 1].append(node.data)
                if node.right:
                    dq.appendleft(node.right)
                if node.left:
                    dq.appendleft(node.left)
        reverse = not reverse # reverse direction of traversal for next level
    return results

from EduBinaryTree import *
from EduTreeNode import * 
def populate_next_pointers(root):
    if not root:
        return root 
    mostleft = root 
    while mostleft.left:
        current = mostleft 
        while current:
            # connect current node's left child to right child
            current.left.next = current.right 
            if current.next:
                # connect current node's right child to left child
                current.right.next = current.next.left 
            current = current.next 
        # move down to next level 
        mostleft = mostleft.left 
    return root 

from collections import defaultdict, deque 
def vertical_order(root):
    if root is None:
        return []
    node_list = defaultdict(list)
    min_column = 0 
    max_index = 0
    queue = deque([(root, 0)]) # push root into queue
    while queue:
        node, column = queue.popleft()
        if node is not None:
            temp = node_list[column]
            temp.append(node.data)
            node_list[column] = temp 
            min_column = min(min_column, column)
            max_index = max(max_index, column)
            queue.append((node.left, column - 1))
            queue.append((node.right, column + 1))
    return [node_list[x] for x in range(min_column, max_index + 1)]

def is_symmetric(root):
    queue = []
    queue.append(root.left)
    queue.append(root.right)
    while queue:
        left = queue.pop(0)
        right = queue.pop(0)
        # both element null, move to next iteration
        if not left and not right:
            continue 
        # any element is null, tree not symmetric
        if not left or not right:
            return False
        if left.data != right.data:
            return False 
        queue.append(left.left)
        queue.append(right.right)
        queue.append(left.right)
        queue.append(right.left)
    return True 

def word_ladder(src, dest, words):
    myset = set(words)
    if dest not in myset:
        return 0 
    q = []
    q.append(src)
    length = 0 
    while q:
        length += 1 
        size = len(q)
        for _ in range(size):
            curr = q.pop(0)
            for i in range(len(curr)):
                alpha = "abcdefghijklmnopqrstuvwxyz"
                for c in alpha:
                    temp = list(curr)
                    temp[i] = c 
                    temp = "".join(temp)
                    if temp == dest:
                        return length + 1 
                    if temp in myset:
                        q.append(temp)
                        myset.remove(temp)
    return 0 # if no sequence exists

def find_target(root, k):
    if not root:
        return False 
    seen = set()
    q = deque() # level-order traversal of BST
    q.append(root)
    while q: 
        # deque front node from queue
        curr = q.popleft()
        if curr:
            # if complement of current node's value exist in set 
            if (k - curr.data) in seen:
                return True 
            seen.add(curr.data) 
            q.append(curr.right)
            q.append(curr.left)
    return False 
```

<!-- TOC --><a name="22-trie"></a>
## 22. Trie

```py
from trie_node import * 
class Trie():
    def __init__(self):
        self.root = TrieNode()
    def insert(self, word):
        node = self.root 
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children.get(c)
        node.is_word = True 
    def search(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                return False 
            node = node.children.get(c)
        return node.is_word
    def search_prefix(self, prefix):
        node = self.root 
        for c in prefix:
            if c not in node.children:
                return False 
            node = node.children.get(c)
        return True 
    
class Trie(object):
    def __init__(self):
        self.root = TrieNode()
    def insert(self, data):
        node = self.root 
        idx = 0 
        for char in data:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            if len(node.search_words) < 3:
                node.search_words.append(data)
            idx += 1
    def search(self, word):
        result, node = [], self.root 
        for i, char in enumerate(word):
            if char not in node.children:
                temp = [[] for _ in range(len(word) - i)]
                return result + temp 
            else:
                node = node.children[char]
                result.append(node.search_words[:])
        return result 
def suggested_products(products, search_word):
    products.sort()
    trie = Trie()
    for x in products:
        trie.insert(x)
    return trie.search(search_word)

def replace_words(sentence, dictionary):
    trie = Trie()
    for prefix in dictionary:
        trie.insert(perfix)
    new_list = sentence.split()
    for i in range(len(new_list)):
        new_list[i] = trie.replace(new_list[i])
    return " ".join(new_list)

class WordDictionary:
    def __init__(self):
        self.root = TrieNode()
        self.can_find = False 
    def add_word(self, word):
        n = len(word)
        cur_node = self.root 
        for i, val in enumerate(word):
            index = ord(val) - ord('a')
            if cur_node.children[index] is None:
                cur_node.children[index] = TrieNode()
            cur_node = cur_node.children[index]
            if i == n - 1:
                if cur_node.complete:
                    print("\tWord already present")
                    return 
                cur_node.complete = True 
        print("\tWord added successfully")
    def search_word(self, word):
        self.can_find = False 
        self.search_helper(self.root, word, 0)
        return self.can_find
    def search_helper(self, node, word, i):
        if self.can_find:
            return 
        if not node:
            return 
        if len(word) == i:
            if node.complete:
                self.can_find = True 
            return 
        if word[i] == '.':
            for j in range(ord('a'), ord('z') + 1):
                self.search_helper(node.children[j - ord('a')], word, i + 1)
        else:
            index = ord(word[i]) - ord('a')
            self.search_helper(node.children[index], word, i + 1)
    def get_words(self):
        words_list = []
        if not self.root:
            return []
        return self.dfs(self.root, "", words_list)
    def dfs(self, node, word, words_list):
        if not node:
            return words_list 
        if node.complete:
            words_list.append(word)
        for j in range(ord('a'), ord('z') + 1):
            prefix = word + chr(j)
            words_list = self.dfs(node.children[j - ord('a')], prefix, words_list)
        return words_list
    
def print_grid(grid):
    for i in grid:
        output = ' '.join(i)
        print("\t", output)
def find_strings(grid, words):
    tried_for_words = Trie()
    result = []
    for word in words:
        tried_for_words.insert(word)
    for j in range(len(grid)):
        for i in range(len(grid[0])):
            dfs(tried_for_words, tried_for_words.root, grid, j, i, result)
    return result 
def dfs(words_trie, node, grid, row, col, result, word=''):
    if node.is_tring:
        result.append(word)
        node.is_string = False 
        words_trie.remove_characters(word)
    if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
        char = grid[row][col] 
        child = node.children.get(char)
        if child is not None:
            word += char 
            grid[row][col] = None 
            for row_offset, col_offset in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                dfs(words_trie, child, grid, row + row_offset, col + col_offset, result, word)
            grid[row][col] = char 

def top_k_frequent_words(words, k):
    frequency_map = defaultdict(int)
    buckets = [None] * (len(words) + 1)
    top_k = []
    for word in words:
        frequency_map[word] += 1 
    for word, frequency in frequency_map.items():
        if buckets[frequency] is None:
            buckets[frequency] = Trie()
        buckets[frequency].add_word(word)
    for i in range(len(buckets) - 1, -1, -1):
        if buckets[i] is not None:
            retrieve_words = []
            buckets[i].get_words(buckets[i].root, retrieve_words)
            if len(retrieve_words) < k:
                top_k.extend(retrieve_words)
                k -= len(retrieve_words)
            else:
                top_k.extend(retrieve_words[:k])
                break 
    return top_k
def generate_frequency_map(words):
    frequency_map = defaultdict(int)
    for word in words:
        frequency_map[word] += 1 
    for key, value in frequency_map.items():
        print(f"\t{key}: {value}")

def find_longest_common_prefix(trie):
    prefix = ""
    node = trie.get_root()
    while node and not node.is_end_of_word and len(node.children) == 1:
        char, next_node = list(node.children.items())[0]
        prefix += char 
        node = next_node 
def longest_common_prefix(strs):
    if not strs:
        return ""
    trie = Trie()
    for word in strs:
        trie.insert(word)
    return find_longest_common_prefix(trie)

def index_pairs(text, words):
    trie = Trie()
    for word in words:
        trie.insert(word)
    output = []
    for i in range(len(text)):
        node = trie.root 
        for j in range(i, len(text)):
            if text[j] not in node.children:
                break 
            node = node.children[text[j]]
            if node.is_end_of_word:
                output.append([i, j])
    return output 
```

<!-- TOC --><a name="23-hash-maps"></a>
## 23. Hash Maps 

```py
from bucket import * 
class DesignHashMap():
    def __init__(self):
        self.key_space = 2069
        self.bucket = [Bucket() for _ in range(self.key_space)]
    def put(self, key, value):
        hash_key = key % self.key_space 
        self.bucket[hash_key].update(key, value)
    def get(self, key):
        hash_key = key % self.key_space 
        return self.bucket[hash_key].get(key)
    def remove(self, key):
        hash_key = key % self.key_space 
        self.bucket[hash_key].remove(key)

def fraction_to_decimal(numerator, denominator):
    result, remainder_map = "", {}
    if numerator == 0:
        return '0'
    if (numerator < 0) ^ (denominator < 0):
        result += '-'
        nuumerator = abs(numerator)
        denominator = abs(denominator)
    quotient = numerator / denominator 
    remainder = (numerator % denominator) * 10 
    result += str(int(quotient))
    if remainder == 0:
        return result 
    else:
        result += "."
        while remainder != 0:
            if remainder in remainder_map.keys():
                beginning = remainder_map.get(remainder)
                left = result[0: beginning]
                right = result[beginning: len(result)]
                result = left + "(" + right + ")"
                return result 
            remainder_map[remainder] = len(result)
            quotient = remainder / denominator 
            result += str(int(quotient))
            remainder = (remainder % denominator) * 10 
        return result 
    
class RequestLogger:
    def __init__(self, time_limit):
        self.requests = {}
        self.limit = time_limit 
    def message_request_decision(self, timestamp, request):
        if request not in self.requests or timestamp - self.requests[request] >= self.limit:
            self.requests[request] = timestamp 
            return True 
        else:
            return False 

def next_greater_element(nums1, nums2):
    stack = []
    map = {}
    for current in nums2:
        while stack and current > stack[-1]:
            map[stack.pop()] = current 
        stack.append(current)
    while stack:
        map[stack.pop()] = -1 
    ans = []
    for num in nums1:
        ans.append(map[num])
    return ans 

def is_isomorphic(string1, string2):
    map_str1_str2 = {}
    map_str2_str1 = {}
    for i in range(len(string1)):
        char1 = string1[i]
        char2 = string2[i]
        if char1 in map_str1_str2 and map_str1_str2[char1] != char2:
            return False 
        if char2 in map_str2_str1 and map_str2_str1[char2] != char1:
            return False 
        map_str1_str2[char1] = char2 
        map_str2_str1[char2] = char1 
    return True 

from typing import List 
from collections import defaultdict 
def find_duplicate(paths):
    file_map = defaultdict(list)
    for path in paths:
        values = path.slit(' ')
        for i in range(1, len(values)):
            name_content = values[i].split('(')
            content = name_content[1][:-1]
            directory = values[0]
            file_name = name_content[0]
            file_path = f"{directory}/{file_name}"
            file_map[content].append(file_path)
    result = []
    for paths in file_map.values():
        if len(paths) > 1:
            result.append(paths)
    return result 

class SparseVector:
    def __init__(self, nums):
        self.hashmap = {}
        for i, n in enumerate(nums):
            if n != 0:
                self.hashmap[i] = n 
    def dot_product(self, vec):
        sum = 0 
        for i, n in self.hashmap.items():
            if i in vec.hashmap:
                sum += n * vec.hashmap[i]
        return sum 
    
from heapq import nlargest 
def high_five(items):
    dict = defaultdict(list)
    max_id = 0 
    for id, score in items:
        dict[id].append(score)
        if id > max_id:
            max_id = id 
    result = []
    for i in range(1, max_id + 1):
        if i in dict:
            scores = nlargest(5, dict[i])
            average = sum(scores) // 5 
            result.append([i, average])
    return result 

from collections import defaultdict
def get_hint(secret, guess):
    dict = defaultdict(int)
    bulls = cows = 0 
    for idx, s in enumerate(secret):
        g = guess[idx]
        if s == g:
            bulls += 1 
        else:
            cows += int(dict[s] < 0) + int(dict[g] > 0)
            dict[s] += 1 
            dict[g] -= 1 
    return "{}A{}B".format(bulls, cows)

def custom_sort_string(order, s):
    frequencies = {}
    for c in s:
        frequencies[c] = frequencies.get(c, 0) + 1 
    result = []
    for c in order:
        if c in frequencies:
            result.append(c * frequencies[c])
            del frequencies[c]
    for c, count in frequencies.items():
        result.append(c * count)
    return ''.join(result)

def dfs(grid, row, col, row_origin, column_origin, path, visited):
    if (row < 0 or col < 0 or row >= len(grid) or col >= len(grid[0]) or 
        (row, col) in visited or grid[row][col] == 0):
        return 
    visited.add((row, col))
    path.append((row - row_origin, col - column_origin))
    dfs(grid, row + 1, col, row_origin, column_origin, path, visited)
    dfs(grid, row - 1, col, row_origin, column_origin, path, visited)
    dfs(grid, row, col + 1, row_origin, column_origin, path, visited)
def num_distinct_islands(grid):
    visited = set()
    unique_islands = defaultdict(int)
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == 1 and (row, col) not in visited:
                path = []
                dfs(grid, row, col, row, col, path, visited)
                unique_islands[tuple(path)] += 1 
    return len(unique_islands)

def wonderful_substrings(word):
    freq = {}
    freq[0] = 1 
    mask = 0 
    res = 0 
    for c in word:
        bit = ord(c) - 97  # a is 0, b is 1, ...
        mask ^= (1 << bit)
        if mask in freq:
            res += freq[mask]
            freq[mask] += 1 
        else:
            freq[mask] = 1 
        for odd_c in range(0, 10):
            if (mask ^ (1 << odd_c)) in freq:
                res += freq[mask ^ (1 << odd_c)]
    return res 

from collections import defaultdict
def appeal_sum(s):
    track = defaultdict(lambda: -1)
    appeal = 0 
    n = len(s)
    for i, c in enumerate(s):
        appeal += (i - track[c]) * (n - i)
        track[c] = i
    return appeal 

def check_subarray_sum(nums, k):
    remainder_map = {0: -1}
    cumulative_sum = 0
    for i, num in enumerate(nums):
        cumulative_sum += num 
        remainder = cumulative_sum % k 
        if remainder in remainder_map:
            if i - remainder_map[remainder] > 1:
                return True 
        else:
            remainder_map[remainder] = i 
    return False 

def unique_occurrences(nums):
    frequency = {}
    for num in nums:
        if num in frequency:
            frequency[num] += 1 
        else:
            frequency[num] = 1 
    seen_frequencies = set()
    for count in frequency.values():
        if count in seen_frequencies:
            return False 
        seen_frequencies.add(count)
    return True 
```

<!-- TOC --><a name="24-knowing-what-to-track"></a>
## 24. Knowing what to Track

```py
def permute_palindrome(st):
    frequencies = {}
    for i in st:
        if i in frequencies:
            frequencies[i] += 1 
        else:
            frequencies[i] = 1 
    count = 0
    for ch in frequencies.keys():
        if frequencies[ch] % 2:
            count += 1
    if count <= 1:
        return True 
    else:
        return False 
    
def is_anagram(str1, str2):
    if len(str1) != len(str2):
        return False 
    table = {}
    for i in str1:
        if i in table:
            table[i] += 1
        else:
            table[i] = 1 
    for i in str2:
        if i in table:
            table[i] -= 1 
        else:
            return False 
    for key in table:
        if table[key] != 0:
            return False 
    return True

class TicTacToe:
    def __init__(self, n):
        self.rows = [0] * (n)
        self.cols = [0] * (n)
        self.diagonal = 0 
        self.anti_diagonal = 0 
    def move(self, row, col, player):
        current_player = -1 
        if player == 1:
            current_player = 1 
        n = len(self.rows)
        self.rows[row] += current_player 
        self.cols[col] += current_player
        if row == col:
            self.diagonal += current_player
        if abs(self.rows[row]) == n or abs(self.cols[col]) == n or abs(self.diagonal) == n or abs(self.anti_diagonal) == n:
            return player 
        return 0 

def group_anagrams(strs):
    res = {}
    for s in strs:
        count = [0] * 26 
        for i in s:
            index = ord(i) - ord('a')
            count[index] += 1 
        key = tuple(count)
        if key in res:
            res[key].append(s)
        else:
            res[key] = [s]
    return res.values()

from collections import defaultdict
class FreqStack:
    def __init__(self):
        self.frequency = defaultdict(int)
        self.group = defaultdict(list)
        self.max_frequency = 0
    def push(self, value):
        freq = self.frequency[value] + 1 
        self.frequency[value] = freq 
        if freq > self.max_frequency:
            self.max_frequency = freq 
        self.group[freq].append(value)
    def pop(self):
        value = ""
        if self.max_frequency > 0:
            value = self.group[self.max_frequency].pop()
            self.frequency[value] -= 1 
            if not self.group[self.max_frequency]:
                self.max_frequency -= 1
                if not self.group[self.max_frequency]:
                    self.max_frequency -= 1 
        else:
            return -1 
        return value 

def first_unique_char(s):
    character_count = {}
    string_length = len(s)
    for i in range(string_length):
        if s[i] in character_count:
            character_count[s[i]] += 1 
        else:
            character_count[s[i]] = 1 
    for i in range(string_length):
        if character_count[s[i]] == 1:
            return i 
    return -1 

def find_anagrams(a, b):
    if len(b) > len(a):
        return []
    ans = []
    hash_a = defaultdict(int)
    hash_b = defaultdict(int)
    for i in range(len(b)):
        hash_b[b[i]] += 1 
    for window_end in range(len(a)):
        hash_a[window_end] += 1 
        if window_end >= len(b):
            window_start = window_end - len(b)
            if hash_a[a[window_start]] == 1:
                del hash_a[a[window_start]]
            else:
                hash_a[a[window_start]] -= 1
        if hash_a == hash_b:
            start_index = window_end - len(b) + 1
            ans.append(start_index)
    return ans

from collections import Counter
def longest_palindrome(words):
    frequencies = Counter(words)
    count = 0
    central = False 
    for word, frequency in frequencies.items():
        if word[0] == word[1]:
            if frequency % 2 == 0:
                count += frequency
            else:
                count += frequency - 1 
                central = True 
        elif word[1] > word[0]:
            count += 2 * min(frequency, frequencies[word[1] + word[0]])
    if central:
        count += 1 
    return 2 * count 

def rank_teams(votes):
    counts = [[0] * 27 for _ in range(26)]
    for t in range(26):
        counts[t][26] = chr(ord('A') + t)
    for i in range(len(votes)):
        for j, c in enumerate(votes[i]):
            counts[ord(c) - ord('A')][j] -= 1 
    counts.sort()
    res = ""
    for i in range(len(votes[0])):
        res += counts[i][26]
    return res 

def num_pairs_divisible_by_60(time):
    remainders = [0] * 60 
    count = 0
    for t in time:
        remainder = t % 60 
        if remainder == 0:
            count += remainders[0]
        else:
            count += remainders[60 - remainder]
        remainders[remainder] += 1
    return count 

def min_pushes(word):
    frequencies = [0] * 26 
    for c in word:
        frequencies[ord(c) - ord('a')] += 1 
    frequencies.sort(reverse=True)
    pushes = 0
    for i in range(26):
        if frequencies[i] == 0:
            break 
        pushes += (i // 8 + 1) * frequencies[i]
    return pushes 

def least_interval(tasks, n):
    frequencies = [0] * 26 
    for task in tasks:
        frequencies[ord(task) - ord('A')] += 1 
    frequencies.sort(reverse=True)
    max_gaps = frequencies[0] - 1
    idle_slots = max_gaps * n 
    for i in range(1, 26):
        if frequencies[i] == 0:
            break 
        idle_slots -= min(max_gaps, frequencies[i])
    idle_slots = max(0, idle_slots)
    return len(tasks) + idle_slots
```

<!-- TOC --><a name="25-union-find"></a>
## 25. Union Find 

```py
class UnionFind:
    def __init__(self, n):
        self.parent = []
        self.rank = rank = [1] * (n + 1)
        for i in range(n + 1):
            self.parent.append(i)
    def find_parent(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find_parent(self.parent[x])
        return self.parent[x]
    def union(self, v1, v2):
        p1, p2 = self.find_parent(v1), self.find_parent(v2)
        if p1 == p2:
            return False 
        elif self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1 
            self.rank[p1] = self.rank[p1] + self.rank[p2]
        else:
            self.parent[p1] = p2 
            self.rank[p2] = self.rank[p2] + self.rank[p1]
        return True 
def redundant_connection(edges):
    graph = UnionFind(len(edges))
    for v1, v2 in edges:
        if not graph.union(v1, v2):
            return [v1, v2]

def num_islands(grid):
    if not grid:
        return 0
    cols = len(grid[0])
    rows = len(grid)
    union_find = UnionFind(grid)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                grid[r][c] == '0'
                if r + 1 < rows and grid[r + 1][c] == '1':
                    union_find.union(r * cols + c, (r + 1) * cols + c)
                if c + 1 < cols and grid[r][c + 1] == '1':
                    union_find.union(r * cols + c, r * cols + c + 1)
    count = union_find.get_count()
    return count 

from collections import defaultdict
def remove_stones(stones):
    offset = 100000
    stone = UnionFind()
    for x, y in stones:
        stone.union(x, (y + offset))
    groups = set()
    for i in stone.parents:
        groups.add(stone.find(i))
    return len(stones) - len(groups)

def longest_consecutive_sequence(nums):
    if len(nums) == 0:
        return 0
    ds = UnionFind(nums)
    for num in nums:
        if num + 1 in ds.parent:
            ds.union(num, num + 1)
    return ds.max_length 

def last_day_to_cross(rows: int, cols: int, water_cells):
    day = 0
    matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    left_node, right_node = 0, rows * cols + 1 
    water_directions = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    water_cells = [(r - 1, c - 1) for r, c in water_cells]
    uf = UnionFind(rows * cols + 2)
    for row, col in water_cells:
        matrix[row][col] = 1 
        for dr, dc in water_directions:
            if within_bounds(row + dr, col + dc, rows, cols) and matrix[row + dr][col + dc] == 1:
                uf.union(find_index(row, col, cols), find_index((row + dr), (col + dc), cols))
        if col == 0:
            uf.union(find_index(row, col, cols), left_node)
        if col == cols - 1:
            uf.union(find_index(row, col, cols), right_node)
        if uf.find(left_node) == uf.find(right_node):
            break 
        day += 1 
    return day 
def find_index(current_row, current_col, cols):
    return current_row * cols + (current_col + 1)
def within_bounds(row, col, rows, cols):
    if not (0 <= col < cols): return False 
    if not (0 <= row < rows): return False 
    return True 

def regions_by_slashes(grid):
    N = len(grid)
    find_union = UnionFind(4 * N * N)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            root = 4 * (r * N + c)
            if val in '/':
                find_union.union(root + 0, root + 1)
                find_union.union(root + 2, root + 3)
            if val in '\ ': 
                find_union.union(root + 0, root + 2)
                find_union.union(root + 1, root + 3)
            if r + 1 < N:
                find_union.union(root + 3, (root + 4 * N) + 0)
            if r - 1 >= 0:
                find_union.union(root + 0, (root - 4 * N) + 3)
            if c + 1 < N:
                find_union.union(root + 2, (root + 4) + 1)
            if c - 1 >= 0:
                find_union.union(root + 1, (root - 4) + 2)
    return sum(find_union.find(x) == x for x in range(4 * N * N))

def accounts_merge(accounts):
    uf = UnionFind(len(accounts))
    email_mapping = {}
    for i, account in enumerate(accounts):
        emails = account[1:]
        for email in emails:
            if email in email_mapping:
                if account[0] != accounts[email_mapping[email]][0]:
                    return 
                uf.union(email_mapping[email], i)
            email_mapping[email] = i 
    merged_accounts = defaultdict(list)
    for email, ids in email_mapping.items():
        merged_accounts[uf.find(ids)].append(email)
    final_merged = []
    for parent, emails in merged_accounts.items():
        final_merged.append([accounts[parent][0]] + sorted(emails))
    return final_merged

def min_malware_spread(graph, initial):
    length = len(graph)
    union_find = UnionFind(length)
    for x in range(length):
        for y in range(length):
            if graph[x][y]:
                union_find.union(x, y)
    infected = defaultdict(int)
    for x in initial:
        infected[union_find.find(x)] += 1 
    maximum_size, candidate_node = 0, min(initial)
    for i in initial:
        infection_count = infected[union_find.find(i)]
        component_size = union_find.rank[union_find.find(i)]
        if infection_count != 1:
            continue 
        if component_size > maximum_size:
            maximum_size = component_size
            candidate_node = i 
        elif component_size == maximum_size and i < candidate_node:
            candidate_node = i 
    return candidate_node

def valid_path(n, edges, source, destination):
    uf = UnionFind(n)
    for x, y in edges:
        uf.union(x, y)
    return uf.find(source) == uf.find(destination)

def get_skyline(buildings):
    coordinates = sorted(list(set([x for building in buildings for x in building[:2]])))
    n = len(coordinates)
    heights = [0] * n 
    index_map = {x: idx for idx, x in enumerate(coordinates)}
    buildings.sort(key=lambda x: -x[2])
    skyline = []
    uf = UnionFind(n)
    for left_x, right_x, height in buildings:
        left, right = index_map[left_x], index_map[right_x]
        while left < right:
            left = uf.find(left)
            if left < right:
                # merge left index with right index, connect two parts of skyline
                uf.union(left, right) 
                heights[left] = height 
                left += 1 
    # build final skyline by looping through heights
    for i in range(n):
        # only add points to skyline when height changes from previous point
        if i == 0 or heights[i] != heights[i - 1]:
            skyline.append([coordinates[i], heights[i]])
    return skyline 
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


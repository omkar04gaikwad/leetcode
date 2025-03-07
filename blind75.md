# LeetCode Blind 75 Cheatsheet

This cheatsheet covers a curated list of 75 LeetCode problems. For each problem, an optimized approach, the time and space complexity, and a sample algorithm (or code snippet) is provided.

---

## 1. Merge Strings Alternately
-  **Difficultiy:** Easy
-  **Optimized Approach:** Use two pointers to iterate over both strings and merge alternately.
-  **Time Complexity:** O(n + m)  
-  **Space Complexity:** O(n + m)
-  **Code:**
   ```python
      def mergeAlternately(word1, word2):
          i, j = 0, 0
          result = []
          while i < len(word1) or j < len(word2):
              if i < len(word1):
                  result.append(word1[i])
                  i += 1
              if j < len(word2):
                  result.append(word2[j])
                  j += 1
          return "".join(result)

## 2. Greatest Common Divisor of Strings
-  **Difficultiy:** Easy
-  **Optimized Approach:** Check if concatenating in both orders yields the same string; if so, return the prefix whose length is the GCD of the string lengths.
-  **Time Complexity:** O(n + m)  
-  **Space Complexity:** O(n + m)
-  **Code:**
   ```python
      def gcdOfStrings(str1, str2):
          if str1 + str2 != str2 + str1:
              return ""
          import math
          gcd_len = math.gcd(len(str1), len(str2))
          return str1[:gcd_len]

## 4. Can Place Flowers
-  **Difficultiy:** Easy
-  **Optimized Approach:** Iterate through the flowerbed and greedily plant flowers if the left, current, and right positions are empty.
-  **Time Complexity:** O(1)  
-  **Space Complexity:** O(1)
-  **Code:**
   ```python
      def canPlaceFlowers(flowerbed, n):
        count = 0
        i = 0
        while i < len(flowerbed):
            if flowerbed[i] == 0 and (i == 0 or flowerbed[i-1] == 0) and (i == len(flowerbed)-1 or flowerbed[i+1] == 0):
                flowerbed[i] = 1
                count += 1
            i += 1
        return count >= n

## 5. Reverse Vowels of a String
-  **Difficultiy:** Easy
-  **Optimized Approach:** Use two pointers to swap vowels in the string.
-  **Time Complexity:** O(n)  
-  **Space Complexity:** O(n)
-  **Code:**
   ```python
      def reverseVowels(s):
        vowels = set("aeiouAEIOU")
        s = list(s)
        i, j = 0, len(s)-1
        while i < j:
            if s[i] not in vowels:
                i += 1
            elif s[j] not in vowels:
                j -= 1
            else:
                s[i], s[j] = s[j], s[i]
                i += 1
                j -= 1
        return "".join(s)


## 6. Reverse Words in a String
-  **Difficultiy:** Medium
-  **Optimized Approach:** Split the string into words, reverse the list, and join them.
-  **Time Complexity:** O(n)  
-  **Space Complexity:** O(n)
-  **Code:**
   ```python
      def reverseWords(s):
        return " ".join(s.split()[::-1])

## 7. Product of Array Except Self
-  **Difficultiy:** Medium
-  **Optimized Approach:** Use two passes to compute left and right products without using extra arrays.
-  **Time Complexity:** O(n)  
-  **Space Complexity:** O(1) excluding the output
-  **Code:**
   ```python
      def productExceptSelf(nums):
        n = len(nums)
        output = [1] * n
        left = 1
        for i in range(n):
            output[i] = left
            left *= nums[i]
        right = 1
        for i in range(n-1, -1, -1):
            output[i] *= right
            right *= nums[i]
        return output

## 8. Increasing Triplet Subsequence
-  **Difficultiy:** Medium
-  **Optimized Approach:** Keep track of the smallest and second smallest values. If a larger number is found, return True.
-  **Time Complexity:** O(n)  
-  **Space Complexity:** O(1)
-  **Code:**
   ```python
      def increasingTriplet(nums):
        first = second = float('inf')
        for num in nums:
            if num <= first:
                first = num
            elif num <= second:
                second = num
            else:
                return True
        return False

## 9. String Compression
-  **Difficultiy:** Medium
-  **Optimized Approach:** Iterate and count consecutive characters, writing the count when needed.
-  **Time Complexity:** O(n)  
-  **Space Complexity:** O(n)
-  **Code:**
   ```python
      def compress(chars):
        anchor = write = 0
        for read, char in enumerate(chars):
            if read + 1 == len(chars) or chars[read+1] != char:
                chars[write] = char
                write += 1
                if read > anchor:
                    for digit in str(read - anchor + 1):
                        chars[write] = digit
                        write += 1
                anchor = read + 1
        return write


## 10. Move Zeroes (Two Pointers)
-  **Difficultiy:** Easy
-  **Optimized Approach:** Use two pointers to move non-zero elements to the front.
-  **Time Complexity:** O(n)  
-  **Space Complexity:** O(n)
-  **Code:**
   ```python
      def moveZeroes(nums):
        lastNonZero = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[lastNonZero], nums[i] = nums[i], nums[lastNonZero]
                lastNonZero += 1


## 11. Is Subsequence
-  **Difficultiy:** Easy
-  **Optimized Approach:** Two pointers to iterate through both strings and match characters.
-  **Time Complexity:** O(n)  
-  **Space Complexity:** O(1)
-  **Code:**
   ```python
      def isSubsequence(s, t):
        i = 0
        for char in t:
            if i < len(s) and s[i] == char:
                i += 1
        return i == len(s)

  
## 12. Container With Most Water
-  **Difficultiy:** Medium
-  **Optimized Approach:** Use two pointers moving inward to maximize area.
-  **Time Complexity:** O(n)  
-  **Space Complexity:** O(1)
-  **Code:**
   ```python
      def maxArea(height):
        i, j = 0, len(height)-1
        max_area = 0
        while i < j:
            max_area = max(max_area, min(height[i], height[j]) * (j - i))
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return max_area

## 13. Max Number of K-Sum Pairs
-  **Difficultiy:** Medium
-  **Optimized Approach:** Use a hash map to count frequencies and find valid pairs.
-  **Time Complexity:** O(n)  
-  **Space Complexity:** O(n)
-  **Code:**
   ```python
      def maxOperations(nums, k):
          count = {}
          ops = 0
          for num in nums:
              complement = k - num
              if count.get(complement, 0) > 0:
                  ops += 1
                  count[complement] -= 1
              else:
                  count[num] = count.get(num, 0) + 1
          return ops

## 14. Maximum Average Subarray I (Sliding Window)
-  **Difficultiy:** Easy
-  **Optimized Approach:** Use a sliding window to compute sums of subarrays.
-  **Time Complexity:** O(n)  
-  **Space Complexity:** O(1)
-  **Code:**
   ```python
      def findMaxAverage(nums, k):
        window_sum = sum(nums[:k])
        max_sum = window_sum
        for i in range(k, len(nums)):
            window_sum += nums[i] - nums[i-k]
            max_sum = max(max_sum, window_sum)
        return max_sum / k

  
## 15. Maximum Number of Vowels in a Substring of Given Length (Sliding Window)
-  **Difficultiy:** Medium
-  **Optimized Approach:** Maintain a sliding window while counting vowels.
-  **Time Complexity:** O(n)  
-  **Space Complexity:** O(1)
-  **Code:**
   ```python
      def maxVowels(s, k):
        vowels = set("aeiouAEIOU")
        count = 0
        for i in range(k):
            if s[i] in vowels:
                count += 1
        max_count = count
        for i in range(k, len(s)):
            if s[i] in vowels:
                count += 1
            if s[i-k] in vowels:
                count -= 1
            max_count = max(max_count, count)
        return max_count

## 16. Max Consecutive Ones III (Sliding Window)
-  **Difficultiy:** Medium
-  **Optimized Approach:** Use a sliding window that allows at most k zeros to be flipped.
-  **Time Complexity:** O(n)  
-  **Space Complexity:** O(1)
-  **Code:**
   ```python
      def longestOnes(nums, k):
        left = 0
        zeros = 0
        max_len = 0
        for right in range(len(nums)):
            if nums[right] == 0:
                zeros += 1
            while zeros > k:
                if nums[left] == 0:
                    zeros -= 1
                left += 1
            max_len = max(max_len, right - left + 1)
        return max_len


## 17. Longest Subarray of 1's After Deleting One Element
-  **Difficultiy:** Medium
-  **Optimized Approach:** Use a sliding window allowing one zero.
-  **Time Complexity:** O(n)  
-  **Space Complexity:** O(1)
-  **Code:**
   ```python
      def longestSubarray(nums):
        left = 0
        zero_count = 0
        max_len = 0
        for right in range(len(nums)):
            if nums[right] == 0:
                zero_count += 1
            while zero_count > 1:
                if nums[left] == 0:
                    zero_count -= 1
                left += 1
            max_len = max(max_len, right - left)
        return max_len


## 18. Find the Highest Altitude (Prefix Sum)
-  **Difficultiy:** Easy
-  **Optimized Approach:** Compute prefix sums and return the maximum altitude.
-  **Time Complexity:** O(n)
-  **Space Complexity:** O(1)
-  **Code:**
   ```python
      def largestAltitude(gain):
          max_alt = 0
          curr = 0
          for g in gain:
              curr += g
              max_alt = max(max_alt, curr)
          return max_alt

## 19. Find Pivot Index
-  **Difficultiy:** Easy
-  **Optimized Approach:** Compute total sum then iterate while updating left sum.
-  **Time Complexity:** O(n)
-  **Space Complexity:** O(1)
-  **Code:**
   ```python
        def pivotIndex(nums):
          total = sum(nums)
          left_sum = 0
          for i, num in enumerate(nums):
              if left_sum == total - left_sum - num:
                  return i
              left_sum += num
          return -1

## 20. Find the Difference of Two Arrays (Hash Map / Set)
-  **Difficultiy:** Easy
-  **Optimized Approach:** Use set difference operations.
-  **Time Complexity:** O(n)
-  **Space Complexity:** O(n)
-  **Code:**
   ```python
        def findDifference(nums1, nums2):
          set1, set2 = set(nums1), set(nums2)
          return [list(set1 - set2), list(set2 - set1)]
   
## 21. Unique Number of Occurrences
- **Difficulty:** Easy
- **Optimized Approach:** Count frequencies and compare the set of counts with the list length.
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)
- **Code:**
   ```python
      def uniqueOccurrences(arr):
        from collections import Counter
        freq = Counter(arr)
        return len(set(freq.values())) == len(freq)
   
## 22. Determine if Two Strings Are Close
- **Difficulty:** Medium
- **Optimized Approach:** Compare sorted frequency values and unique character sets.
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)
- **Code:**
  ```python
      def closeStrings(word1, word2):
        if len(word1) != len(word2):
            return False
        from collections import Counter
        c1, c2 = Counter(word1), Counter(word2)
        return sorted(c1.values()) == sorted(c2.values()) and set(word1) == set(word2)
## 23. Equal Row and Column Pairs
- **Difficulty:** Medium
- **Optimized Approach:** Use tuple representations to count row and column patterns.
- **Time Complexity:** O(n²)
- **Space Complexity:** O(n²)
- Pseudo - **Code:**
  ``` pgsql
      For each row, convert to tuple and count frequency.
      For each column, convert to tuple and add matching count to answer.

## 24. Removing Stars From a String (Stack)
- **Difficulty:** Medium
- **Optimized Approach:** Process the string using a stack, removing the last character when a star is encountered.
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)
- **Code:**
  ```python
      def removeStars(s):
        stack = []
        for char in s:
            if char == '*':
                stack.pop()
            else:
                stack.append(char)
        return "".join(stack)


## 25. Asteroid Collision
- **Difficulty:** Medium
- **Optimized Approach:** Use a stack to simulate collisions between asteroids.
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)
- **Code:**
  ```python
      def asteroidCollision(asteroids):
        stack = []
        for ast in asteroids:
            collision = False
            while stack and ast < 0 < stack[-1]:
                if abs(ast) > stack[-1]:
                    stack.pop()
                    continue
                elif abs(ast) == stack[-1]:
                    stack.pop()
                collision = True
                break
            if not collision:
                stack.append(ast)
        return stack
## 26. Decode String
- **Difficulty:** Medium
- **Optimized Approach:** Use a stack to decode the nested encoded strings.
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)
- **Code:**
  ```python
      def decodeString(s):
        stack, current_num, current_str = [], 0, ""
        for char in s:
            if char.isdigit():
                current_num = current_num * 10 + int(char)
            elif char == '[':
                stack.append((current_str, current_num))
                current_str, current_num = "", 0
            elif char == ']':
                prev_str, num = stack.pop()
                current_str = prev_str + current_str * num
            else:
                current_str += char
      return current_str
## 27. Number of Recent Calls (Queue)
- **Difficulty:** Easy
- **Optimized Approach:** Use a deque (queue) to keep track of calls within a fixed time window.
- **Time Complexity:** O(n) worst-case
- **Space Complexity:** O(n)
- **Code:**
  ```python
      from collections import deque
      class RecentCounter:
        def __init__(self):
            self.q = deque()
        def ping(self, t):
            self.q.append(t)
            while self.q and self.q[0] < t - ## 3000:
                self.q.popleft()
            return len(self.q)
## 28. Dota2 Senate
- **Difficulty:** Medium
- **Optimized Approach:** Use two queues to simulate the rounds of bans between parties.
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)
- Pseudo - **Code:**
    ```sql
        Initialize two queues for Radiant and Dire with their indices.
        While both queues are not empty:
          Compare front indices.
          The smaller index bans the other and is appended with index + n.
        Return the winning party.

## 29. Delete the Middle Node of a Linked List
- **Difficulty:** Medium
- **Optimized Approach:** Use slow and fast pointers to find the middle, then delete it.
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)
- **Code:**
    ```python
      def deleteMiddle(head):
        if not head or not head.next:
            return None
        slow, fast = head, head.next.next
        prev = head
        while fast and fast.next:
            prev = slow
            slow = slow.next
            fast = fast.next.next
        prev.next = slow.next
        return head

## 30. Odd Even Linked List
- **Difficulty:** Medium
- **Optimized Approach:** Separate odd and even nodes and then link them.
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)
- **Code:**
    ```python
        def oddEvenList(head):
          if not head:
              return head
          odd, even = head, head.next
          even_head = even
          while even and even.next:
              odd.next = even.next
              odd = odd.next
              even.next = odd.next
              even = even.next
          odd.next = even_head
        return head

## 31. Reverse Linked List
- **Difficulty:** Easy
- **Optimized Approach:** Iteratively reverse pointers.
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)
- **Code:**
  ```python
      def reverseList(head):
        prev = None
        while head:
            next_node = head.next
            head.next = prev
            prev = head
            head = next_node
        return prev

## 32. Maximum Twin Sum of a Linked List
- **Difficulty:** Medium
- **Optimized Approach:** Find the middle, reverse the second half, and compute maximum twin sums.
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)
- **Code:**
  ```python
      def pairSum(head):
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        prev = None
        curr = slow
        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
        max_sum = 0
        first, second = head, prev
        while second:
            max_sum = max(max_sum, first.val + second.val)
            first = first.next
            second = second.next
        return max_sum

## 33. Maximum Depth of Binary Tree (DFS)
- **Difficulty:** Easy
- **Optimized Approach:** Recursively compute the depth of subtrees.
- **Time Complexity:** O(n)
- **Space Complexity:** O(n) (worst-case recursion)
- **Code:**
  ```python
      def maxDepth(root):
        if not root:
            return 0
        return 1 + max(maxDepth(root.left), maxDepth(root.right))

## 34. Leaf-Similar Trees
- **Difficulty:** Easy
- **Optimized Approach:** Collect leaf values from both trees via DFS and compare.
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)
- **Code:**
  ```python
      def leafSimilar(root1, root2):
        def dfs(root):
            if not root:
                return []
            if not root.left and not root.right:
                return [root.val]
            return dfs(root.left) + dfs(root.right)
        return dfs(root1) == dfs(root2)

## 35. Count Good Nodes in Binary Tree
- **Difficulty:** Medium
- **Optimized Approach:** DFS while passing down the maximum value seen so far.
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)
- **Code:**
  ```python
      def goodNodes(root):
        def dfs(node, max_val):
            if not node:
                return 0
            total = 1 if node.val >= max_val else 0
            max_val = max(max_val, node.val)
            total += dfs(node.left, max_val)
            total += dfs(node.right, max_val)
            return total
        return dfs(root, root.val)

## 36. Path Sum III
- **Difficulty:** Medium
- **Optimized Approach:** DFS with prefix sum dictionary.
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)
- **Code:**
  ```python
      def pathSum(root, target):
        def dfs(node, curr_sum, prefix):
            if not node:
                return 0
            curr_sum += node.val
            count = prefix.get(curr_sum - target, 0)
            prefix[curr_sum] = prefix.get(curr_sum, 0) + 1
            count += dfs(node.left, curr_sum, prefix)
            count += dfs(node.right, curr_sum, prefix)
            prefix[curr_sum] -= 1
            return count
        return dfs(root, 0, {0: 1})

## 37. Longest ZigZag Path in a Binary Tree
- **Difficulty:** Medium
- **Optimized Approach:** DFS that tracks the current zigzag length and direction.
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)
- Pseudo - **Code:**
  ```python
      def longestZigZag(root):
        def dfs(node, is_left, length):
            if not node:
                return length
            left = dfs(node.left, False, length + 1)
            right = dfs(node.right, True, length + 1)
            return max(length, left, right)
        return max(dfs(root.left, True, 1), dfs(root.right, False, 1))

## 38. Lowest Common Ancestor of a Binary Tree
- **Difficulty:** Medium
- **Optimized Approach:** Recursively traverse the tree to find the split point.
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)
- **Code:**
  ```python 
      def lowestCommonAncestor(root, p, q):
        if not root or root == p or root == q:
            return root
        left = lowestCommonAncestor(root.left, p, q)
        right = lowestCommonAncestor(root.right, p, q)
        return root if left and right else left or right

## 39. Binary Tree Right Side View (BFS)
- **Difficulty:** Medium
- **Optimized Approach:** Level order traversal keeping the last node of each level.
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)
- **Code:**
  ```python
      def rightSideView(root):
        from collections import deque
        if not root:
            return []
        q = deque([root])
        result = []
        while q:
            size = len(q)
            for i in range(size):
                node = q.popleft()
                if i == size - 1:
                    result.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
        return result

## 40. Maximum Level Sum of a Binary Tree
- **Difficulty:** Medium
- **Optimized Approach:** Use BFS to compute the sum at each level and track the maximum.
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)
- **Code:**
  ```python
      def maxLevelSum(root):
        from collections import deque
        q = deque([root])
        max_sum = float('-inf')
        level = 0
        result = 0
        while q:
            level += 1
            level_sum = 0
            for _ in range(len(q)):
                node = q.popleft()
                level_sum += node.val
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            if level_sum > max_sum:
                max_sum = level_sum
                result = level
        return result

## 41. Search in a Binary Search Tree
- **Difficulty:** Easy
- **Optimized Approach:** Utilize the BST property for a recursive or iterative search.
- **Time Complexity:** O(h) average, O(n) worst-case
- **Space Complexity:** O(h)
- **Code:**
  ```python
      def searchBST(root, val):
        if not root or root.val == val:
            return root
        if val < root.val:
            return searchBST(root.left, val)
        else:
            return searchBST(root.right, val)
## 42. Delete Node in a BST
- **Difficulty:** Medium
- **Optimized Approach:** Recursively find the node, then handle deletion with three cases.
- **Time Complexity:** O(h) average, O(n) worst-case
- **Space Complexity:** O(h)
- **Code:**
  ```python
      def deleteNode(root, key):
        if not root:
            return None
        if key < root.val:
            root.left = deleteNode(root.left, key)
        elif key > root.val:
            root.right = deleteNode(root.right, key)
        else:
            if not root.left:
                return root.right
            if not root.right:
                return root.left
            temp = root.right
            while temp.left:
                temp = temp.left
            root.val = temp.val
            root.right = deleteNode(root.right, temp.val)
        return root

## 43. Keys and Rooms (Graphs - DFS)
- **Difficulty:** Medium
- **Optimized Approach:** Use DFS to explore rooms starting from room 0.
- **Time Complexity:** O(n + e)
- **Space Complexity:** O(n)
- **Code:**
  ```python
      def canVisitAllRooms(rooms):
        visited = set()
        def dfs(room):
            if room in visited:
                return
            visited.add(room)
            for key in rooms[room]:
                dfs(key)
        dfs(0)
        return len(visited) == len(rooms)

## 44. Number of Provinces
- **Difficulty:** Medium
- **Optimized Approach:** DFS (or Union-Find) on the connectivity matrix.
- **Time Complexity:** O(n²)
- **Space Complexity:** O(n)
- **Code:**
  ```python
      def findCircleNum(isConnected):
        n = len(isConnected)
        visited = set()
        def dfs(i):
            for j in range(n):
                if isConnected[i][j] == 1 and j not in visited:
                    visited.add(j)
                    dfs(j)
        count = 0
        for i in range(n):
            if i not in visited:
                visited.add(i)
                dfs(i)
                count += 1
        return count

## 45. Reorder Routes to Make All Paths Lead to the City Zero
- **Difficulty:** Medium
- **Optimized Approach:** Use DFS/BFS on a directed graph and count edges to be reversed.
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)
- Pseudo - **Code:**
  ```pgsql
        Build a graph with original and reverse edges.
        Starting from node 0, perform DFS and count edges that are in the wrong direction.
## 46. Evaluate Division
- **Difficulty:** Medium
- **Optimized Approach:** Build a graph where edges represent division values, then use BFS/DFS.
- **Time Complexity:** O(n + e)
- **Space Complexity:** O(n)
- **Code:**
  ```python
      def calcEquation(equations, values, queries):
        from collections import defaultdict, deque
        graph = defaultdict(list)
        for (A, B), value in zip(equations, values):
            graph[A].append((B, value))
            graph[B].append((A, 1 / value))
        def bfs(src, dst):
            if src not in graph or dst not in graph:
                return -1.0
            q = deque([(src, 1)])
            visited = {src}
            while q:
                node, prod = q.popleft()
                if node == dst:
                    return prod
                for nei, val in graph[node]:
                    if nei not in visited:
                        visited.add(nei)
                        q.append((nei, prod * val))
            return -1.0
        return [bfs(s, t) for s, t in queries]

## 47. Nearest Exit from Entrance in Maze (Graphs - BFS)
- **Difficulty:** Medium
- **Optimized Approach:** BFS starting from the entrance; return the distance when an exit is reached.
- **Time Complexity:** O(m*n)
- **Space Complexity:** O(m*n)
- **Code:**
    ```python
        def nearestExit(maze, entrance):
          from collections import deque
          rows, cols = len(maze), len(maze[0])
          q = deque([(entrance[0], entrance[1], 0)])
          maze[entrance[0]][entrance[1]] = '+'
          while q:
              r, c, d = q.popleft()
              if (r, c) != tuple(entrance) and (r in [0, rows-1] or c in [0, cols-1]):
                  return d
              for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                  nr, nc = r+dr, c+dc
                  if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == '.':
                      maze[nr][nc] = '+'
                      q.append((nr, nc, d+1))
        return -1

## 48. Rotting Oranges
- **Difficulty:** Medium
- **Optimized Approach:** Use multi-source BFS starting from all rotten oranges.
- **Time Complexity:** O(m*n)
- **Space Complexity:** O(m*n)
- **Code:**
  ```python
      def orangesRotting(grid):
        from collections import deque
        rows, cols = len(grid), len(grid[0])
        q = deque()
        fresh = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    q.append((r, c))
                elif grid[r][c] == 1:
                    fresh += 1
        minutes = 0
        while q and fresh:
            for _ in range(len(q)):
                r, c = q.popleft()
                for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                        grid[nr][nc] = 2
                        fresh -= 1
                        q.append((nr, nc))
            minutes += 1
        return minutes if fresh == 0 else -1

## 49. Kth Largest Element in an Array (Heap / Priority Queue)
- **Difficulty:** Medium
- **Optimized Approach:** Maintain a min-heap of size k.
- **Time Complexity:** O(n log k)
- **Space Complexity:** O(k)
- **Code:**
  ```python
      import heapq
      def findKthLargest(nums, k):
        heap = []
        for num in nums:
            heapq.heappush(heap, num)
            if len(heap) > k:
                heapq.heappop(heap)
        return heap[0]

## 50. Smallest Number in Infinite Set
- **Difficulty:** Medium
- **Optimized Approach:** Use a min-heap to keep track of the smallest available number.
- **Time Complexity:** O(log n) per operation
- **Space Complexity:** Depends on operations
- Pseudo - **Code:**
    ```arduino
    
      Initialize a min-heap with available numbers.
      For pop, remove the smallest element.
      For add, push the number back if not already present.
## 51. Maximum Subsequence Score
- **Difficulty:** Medium
- **Optimized Approach:** Greedy approach with heap and prefix/suffix computations.
- **Time Complexity:** O(n log n)
- **Space Complexity:** O(n)
- Pseudo - **Code:**
      ``` Compute prefix scores, use a heap to track potential indices, and choose the optimal subsequence.
## 52. Total Cost to Hire K Workers
- **Difficulty:** Medium
- **Optimized Approach:** Use two heaps to simulate hiring from both ends.
- **Time Complexity:** O(k log n)
- **Space Complexity:** O(n)
- **Code:**
    ```python
    import heapq
    def totalCost(costs, k, candidates):
      n = len(costs)
      left, right = 0, n - 1
      heap = []
      for _ in range(candidates):
          if left <= right:
              heapq.heappush(heap, (costs[left], left))
              left += 1
      for _ in range(candidates):
          if left <= right:
              heapq.heappush(heap, (costs[right], right))
              right -= 1
      total = 0
      for _ in range(k):
          cost, idx = heapq.heappop(heap)
          total += cost
          if left <= right:
              if idx < left:
                  heapq.heappush(heap, (costs[left], left))
                  left += 1
              else:
                  heapq.heappush(heap, (costs[right], right))
                  right -= 1
      return total
## 53. Guess Number Higher or Lower (Binary Search)
- **Difficulty:** Easy
- **Optimized Approach:** Apply binary search between 1 and n.
- **Time Complexity:** O(log n)
- **Space Complexity:** O(1)
- **Code:**
  ```python
  
  def guessNumber(n):
    low, high = 1, n
    while low <= high:
        mid = (low + high) // 2
        res = guess(mid)  # assume guess(mid) is provided
        if res == 0:
            return mid
        elif res < 0:
            high = mid - 1
        else:
            low = mid + 1
## 54. Successful Pairs of Spells and Potions
- **Difficulty:** Medium
- **Optimized Approach:** Sort potions and for each spell, use binary search.
- **Time Complexity:** O(m log n)
- **Space Complexity:** O(1)
- **Code:**
  ```python
  
  def successfulPairs(spells, potions, success):
    potions.sort()
    res = []
    n = len(potions)
    import bisect
    for spell in spells:
        target = (success + spell - 1) // spell
        idx = bisect.bisect_left(potions, target)
        res.append(n - idx)
    return res
## 55. Find Peak Element
- **Difficulty:** Medium
- **Optimized Approach:** Use binary search to find a peak element.
- **Time Complexity:** O(log n)
- **Space Complexity:** O(1)
- **Code:**
  ```python
  
  def findPeakElement(nums):
    low, high = 0, len(nums)-1
    while low < high:
        mid = (low + high) // 2
        if nums[mid] < nums[mid+1]:
            low = mid + 1
        else:
            high = mid
    return low
## 56. Koko Eating Bananas
- **Difficulty:** Medium
- **Optimized Approach:** Binary search for the minimum eating speed that allows finishing within h hours.
- **Time Complexity:** O(n log m)
- **Space Complexity:** O(1)
- **Code:**
  ```python
  
  def minEatingSpeed(piles, h):
    def canFinish(k):
        hours = 0
        for pile in piles:
            hours += -(-pile // k)  # ceiling division
        return hours <= h
    low, high = 1, max(piles)
    while low < high:
        mid = (low + high) // 2
        if canFinish(mid):
            high = mid
        else:
            low = mid + 1
    return low
## 57. Letter Combinations of a Phone Number (Backtracking)
- **Difficulty:** Medium
- **Optimized Approach:** Use backtracking with digit-to-letter mapping.
- **Time Complexity:** O(4^n * n)
- **Space Complexity:** O(n)
- **Code:**
  ```python
  
  def letterCombinations(digits):
    if not digits:
        return []
    phone = {
        "2": "abc", "3": "def", "4": "ghi", "5": "jkl",
        "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"
    }
    res = []
    def backtrack(index, path):
        if index == len(digits):
            res.append("".join(path))
            return
        for letter in phone[digits[index]]:
            path.append(letter)
            backtrack(index+1, path)
            path.pop()
    backtrack(0, [])
    return res
## 58. Combination Sum III (Backtracking)
- **Difficulty:** Medium
- **Optimized Approach:** Backtracking to select k numbers (1-9) that sum to target.
- **Time Complexity:** O(1) (bounded by constant 9)
- **Space Complexity:** O(k)
- **Code:**
  ```python
  
  def combinationSum3(k, n):
    res = []
    def backtrack(start, path, target):
        if len(path) == k and target == 0:
            res.append(path[:])
            return
        for num in range(start, 10):
            if num > target:
                break
            path.append(num)
            backtrack(num+1, path, target-num)
            path.pop()
    backtrack(1, [], n)
    return res
## 59. N-th Tribonacci Number (DP - 1D)
- **Difficulty:** Easy
- **Optimized Approach:** DP with constant space using three variables.
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)
- **Code:**
  ```python
  
  def tribonacci(n):
    if n == 0: return 0
    if n in (1,2): return 1
    a, b, c = 0, 1, 1
    for _ in range(3, n+1):
        a, b, c = b, c, a+b+c
    return c
## 60. Min Cost Climbing Stairs (DP - 1D)
- **Difficulty:** Easy
- **Optimized Approach:** Build up a DP table from the base with minimal cost.
- **Time Complexity:** O(n)
- **Space Complexity:** O(n) (can be optimized to O(1))
- **Code:**
  ```python
  
  def minCostClimbingStairs(cost):
    n = len(cost)
    dp = [0]*(n+1)
    for i in range(2, n+1):
        dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])
    return dp[n]
## 61. House Robber (DP - 1D)
- **Difficulty:** Medium
- **Optimized Approach:** DP with two variables tracking the max loot when skipping or robbing.
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)
- **Code:**
  ```python
  
  def rob(nums):
    prev, curr = 0, 0
    for num in nums:
        prev, curr = curr, max(curr, prev + num)
    return curr
## 62. Domino and Tromino Tiling
- **Difficulty:** Medium
- **Optimized Approach:** Use DP recurrence to count tilings.
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)
- **Code:**
  ```python
  
  def numTilings(n):
    MOD = 10**9 + 7
    if n <= 2: return n
    dp = [0]*(n+1)
    dp[0], dp[1], dp[2] = 1, 1, 2
    for i in range(3, n+1):
        dp[i] = (2*dp[i-1] + dp[i-3]) % MOD
    return dp[n]
## 63. Unique Paths (DP - Multidimensional)
- **Difficulty:** Medium
- **Optimized Approach:** DP grid where dp[i][j] = dp[i-1][j] + dp[i][j-1].
- **Time Complexity:** O(m*n)
- **Space Complexity:** O(m*n)
- **Code:**
  ```python
  
  def uniquePaths(m, n):
    dp = [[1]*n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[-1][-1]
## 64. Longest Common Subsequence (DP - Multidimensional)
- **Difficulty:** Medium
- **Optimized Approach:** Fill a 2D DP table comparing both strings.
- **Time Complexity:** O(m*n)
- **Space Complexity:** O(m*n)
- **Code:**
  ```python
  
  def longestCommonSubsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
## 65. Best Time to Buy and Sell Stock with Transaction Fee
- **Difficulty:** Medium
- **Optimized Approach:** DP with two states: holding or not holding stock.
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)
- **Code:**
  ```python
  
  def maxProfit(prices, fee):
    cash, hold = 0, -prices[0]
    for price in prices[1:]:
        cash = max(cash, hold + price - fee)
        hold = max(hold, cash - price)
    return cash
## 66. Edit Distance
- **Difficulty:** Medium
- **Optimized Approach:** 2D DP table to track minimum operations (insert, delete, substitute).
- **Time Complexity:** O(m*n)
- **Space Complexity:** O(m*n)
- **Code:**
  ```python
  
  def minDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
## 67. Counting Bits (Bit Manipulation)
- **Difficulty:** Easy
- **Optimized Approach:** DP where res[i] = res[i >> 1] + (i & 1).
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)
- **Code:**
  ```python
  
  def countBits(n):
    res = [0]*(n+1)
    for i in range(1, n+1):
        res[i] = res[i >> 1] + (i & 1)
    return res
## 68. Single Number (Bit Manipulation)
- **Difficulty:** Easy
- **Optimized Approach:** XOR all elements; duplicates cancel out.
- **Time Complexity:** O(n)
- **Space Complexity:** O(1)
- **Code:**
  ```python
  
  def singleNumber(nums):
    res = 0
    for num in nums:
        res ^= num
    return res
## 69. Minimum Flips to Make a OR b Equal to c
- **Difficulty:** Medium
- **Optimized Approach:** Process each bit of a, b, and c; simulate the needed flips.
- **Time Complexity:** O(n) (where n is number of bits)
- **Space Complexity:** O(1)
- **Code:**
  ```python
  
  def minFlips(a, b, c):
    flips = 0
    while a or b or c:
        bit_a = a & 1
        bit_b = b & 1
        bit_c = c & 1
        if bit_c == 0:
            flips += (bit_a + bit_b)
        else:
            if bit_a == 0 and bit_b == 0:
                flips += 1
        a >>= 1
        b >>= 1
        c >>= 1
    return flips
## 70. Implement Trie (Prefix Tree)
- **Difficulty:** Medium
- **Optimized Approach:** Create a TrieNode class with a dictionary to store children.
- **Time Complexity:** O(m) per operation, where m is the length of the word
- **Space Complexity:** O(total characters in Trie)
- **Code:**
  ```python
  
  class TrieNode:
    def __init__(self):
        self.children = {}
        self.isWord = False
  
  class Trie:
    def __init__(self):
        self.root = TrieNode()
  
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.isWord = True
  
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.isWord
  
    def startsWith(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
## 71. Search Suggestions System
- **Difficulty:** Medium
- **Optimized Approach:** Sort the products; for each character in searchWord, use binary search to get suggestions.
- **Time Complexity:** O(m log n) per query
- **Space Complexity:** O(1)
- **Code:**
  ```python
  
  def suggestedProducts(products, searchWord):
    import bisect
    products.sort()
    res = []
    prefix = ""
    for char in searchWord:
        prefix += char
        start = bisect.bisect_left(products, prefix)
        res.append([p for p in products[start:start+3] if p.startswith(prefix)])
    return res
## 72. Non-overlapping Intervals (Intervals)
- **Difficulty:** Medium
- **Optimized Approach:** Sort intervals by end time and greedily select non-overlapping intervals.
- **Time Complexity:** O(n log n)
- **Space Complexity:** O(1)
- **Code:**
  ```python
  
  def eraseOverlapIntervals(intervals):
    intervals.sort(key=lambda x: x[1])
    count = 0
    prev_end = float('-inf')
    for start, end in intervals:
        if start >= prev_end:
            prev_end = end
        else:
            count += 1
    return count
## 73. Minimum Number of Arrows to Burst Balloons (Intervals)
- **Difficulty:** Medium
- **Optimized Approach:** Greedily shoot an arrow at the end of the first interval and count.
- **Time Complexity:** O(n log n)
- **Space Complexity:** O(1)
- **Code:**
  ```python
  
  def findMinArrowShots(points):
    if not points:
        return 0
    points.sort(key=lambda x: x[1])
    arrows = 1
    end = points[0][1]
    for start, finish in points:
        if start > end:
            arrows += 1
            end = finish
    return arrows
## 74. Daily Temperatures (Monotonic Stack)
- **Difficulty:** Medium
- **Optimized Approach:** Use a monotonic stack to track indices of decreasing temperatures.
- **Time Complexity:** O(n)
- **Space Complexity:** O(n)
- **Code:**
  ```python
  
  def dailyTemperatures(T):
    n = len(T)
    res = [0]*n
    stack = []
    for i, t in enumerate(T):
        while stack and t > T[stack[-1]]:
            idx = stack.pop()
            res[idx] = i - idx
        stack.append(i)
    return res
## 75. Online Stock Span (Monotonic Stack)
- **Difficulty:** Medium
- **Optimized Approach:** Use a stack to maintain pairs of (price, span) for calculating the stock span.
- **Time Complexity:** O(n) amortized
- **Space Complexity:** O(n)
- **Code:**
  ```python
  
  class StockSpanner:
    def __init__(self):
        self.stack = []
  
    def next(self, price):
        span = 1
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1]
        self.stack.append((price, span))
        return span
## 13. Roman to Integer
Solved
Easy
Topics
Companies
Hint
Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
For example, 2 is written as II in Roman numeral, just two ones added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given a roman numeral, convert it to an integer.
```python
   def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        romantoint = {
            'I':1,
            'V':5,
            'X':10,
            'L':50,
            'C':100,
            'D':500,
            'M':1000
        }
        prevA = 0
        total = 0
        for x in reversed(s):
            val = romantoint[x]
            if prevA > val:
                total -= val
            else:
                total += val
            prevA = val
            print(prevA, total)
        return total
## 72. Edit Distance
Solved
Medium
Topics
Companies
Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.

You have the following three operations permitted on a word:

Insert a character
Delete a character
Replace a character
 

Example 1:

Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')
Example 2:

Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation: 
intention -> inention (remove 't')
inention -> enention (replace 'i' with 'e')
enention -> exention (replace 'n' with 'x')
exention -> exection (replace 'n' with 'c')
exection -> execution (insert 'u')

```python
       def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        m, n = len(word1), len(word2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[m][n]
## 12. Integer to Roman
   Medium
   Topics
   Companies
   Seven different symbols represent Roman numerals with the following values:
   
   Symbol	Value
   I	1
   V	5
   X	10
   L	50
   C	100
   D	500
   M	1000
   Roman numerals are formed by appending the conversions of decimal place values from highest to lowest. Converting a decimal place value into a Roman numeral has the following rules:
   
   If the value does not start with 4 or 9, select the symbol of the maximal value that can be subtracted from the input, append that symbol to the result, subtract its value, and convert the remainder to a Roman numeral.
   If the value starts with 4 or 9 use the subtractive form representing one symbol subtracted from the following symbol, for example, 4 is 1 (I) less than 5 (V): IV and 9 is 1 (I) less than 10 (X): IX. Only the following subtractive forms are used: 4 (IV), 9 (IX), 40 (XL), 90 (XC), 400 (CD) and 900 (CM).
   Only powers of 10 (I, X, C, M) can be appended consecutively at most 3 times to represent multiples of 10. You cannot append 5 (V), 50 (L), or 500 (D) multiple times. If you need to append a symbol 4 times use the subtractive form.
   Given an integer, convert it to a Roman numeral.
   
    
   
   Example 1:
   
   Input: num = 3749
   
   Output: "MMMDCCXLIX"
   
   Explanation:
   
   3000 = MMM as 1000 (M) + 1000 (M) + 1000 (M)
    700 = DCC as 500 (D) + 100 (C) + 100 (C)
     40 = XL as 10 (X) less of 50 (L)
      9 = IX as 1 (I) less of 10 (X)
   Note: 49 is not 1 (I) less of 50 (L) because the conversion is based on decimal places
   Example 2:
   
   Input: num = 58
   
   Output: "LVIII"
   
   Explanation:
   
   50 = L
    8 = VIII
   Example 3:
   
   Input: num = 1994
   
   Output: "MCMXCIV"
   
   Explanation:
   
   1000 = M
    900 = CM
     90 = XC
      4 = IV
   ```python
      def intToRoman(self, num: int) -> str:
           cs = ('M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I')
           vs = (1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1)
           ans = []
           for c, v in zip(cs, vs):
               while num >= v:
                   num -= v
                   ans.append(c)
           return ''.join(ans)
## 36. Valid Sudoku
   Solved
   Medium
   Topics
   Companies
   Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:
   
   Each row must contain the digits 1-9 without repetition.
   Each column must contain the digits 1-9 without repetition.
   Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
   Note:
   
   A Sudoku board (partially filled) could be valid but is not necessarily solvable.
   Only the filled cells need to be validated according to the mentioned rules.
   ```python
      def isValidSudoku(self, board: List[List[str]]) -> bool:
           rows = [set() for _ in range(9)]
           cols = [set() for _ in range(9)]
           boxes = [set() for _ in range(9)]
           for i in range(9):
               for j in range(9):
                   num = board[i][j]
                   if num == ".":
                       continue
                   box_index = (i//3)*3 + (j//3)
                   if num in rows[i] or num in cols[j] or num in boxes[box_index]:
                       return False
                   rows[i].add(num)
                   cols[j].add(num)
                   boxes[box_index].add(num)
           return True

## 37. Sudoku Solver
   Hard
   Topics
   Companies
   Write a program to solve a Sudoku puzzle by filling the empty cells.
   
   A sudoku solution must satisfy all of the following rules:
   
   Each of the digits 1-9 must occur exactly once in each row.
   Each of the digits 1-9 must occur exactly once in each column.
   Each of the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.
   The '.' character indicates empty cells.
   ```python
      def solveSudoku(self, board: List[List[str]]) -> None:
           def dfs(k):
               nonlocal ok
               if k == len(t):
                   ok = True
                   return
               i, j = t[k]
               for v in range(9):
                   if row[i][v] == col[j][v] == block[i // 3][j // 3][v] == False:
                       row[i][v] = col[j][v] = block[i // 3][j // 3][v] = True
                       board[i][j] = str(v + 1)
                       dfs(k + 1)
                       row[i][v] = col[j][v] = block[i // 3][j // 3][v] = False
                   if ok:
                       return
   
           row = [[False] * 9 for _ in range(9)]
           col = [[False] * 9 for _ in range(9)]
           block = [[[False] * 9 for _ in range(3)] for _ in range(3)]
           t = []
           ok = False
           for i in range(9):
               for j in range(9):
                   if board[i][j] == '.':
                       t.append((i, j))
                   else:
                       v = int(board[i][j]) - 1
                       row[i][v] = col[j][v] = block[i // 3][j // 3][v] = True
           dfs(0)

# leetcode
# LeetCode Hard Problem Cheatsheet

This cheatsheet covers data structures and techniques for tackling hard problems on LeetCode. It includes tips for problem analysis, common data structures, tricks, and examples for popular problem patterns.

---

## 1. Problem Analysis

- **Identify Operations:**
  - **Searching:** Need fast lookup? Use hash tables.
  - **Insertion/Deletion:** Frequent changes? Consider linked lists or trees.
  - **Ordering:** Data must be sorted? Use balanced trees or heaps.
  - **Range Queries:** Sum/min/max operations over ranges? Use segment trees or Binary Indexed Trees.
- **Assess Constraints:**
  - **Input Size:** Guides time complexity (O(1), O(log n), O(n)).
  - **Space Limitations:** Choose more space-efficient structures when needed.

---

## 2. Data Structures Overview

- **Arrays/Lists:**
  - *Usage:* Indexed storage and as a basis for dynamic programming (DP).
  
- **Hash Tables (Dictionaries/Sets):**
  - *Usage:* Fast lookups, frequency counts, and membership tests (average O(1) time).

- **Linked Lists:**
  - *Usage:* When you need efficient insertions/deletions and maintaining sequential order.

- **Stacks & Queues:**
  - *Usage:* LIFO (stacks) and FIFO (queues) operations, often used in tree traversals.

- **Trees (BSTs, AVL, Red-Black):**
  - *Usage:* Ordered data with efficient search, insertion, and deletion.

- **Heaps (Priority Queues):**
  - *Usage:* Quickly extract the smallest/largest element (useful in greedy algorithms).

- **Graphs (Adjacency Lists/Matrices):**
  - *Usage:* Modeling networks and connectivity; use BFS, DFS, or Dijkstra’s algorithm for traversal.

- **Union-Find (Disjoint Set):**
  - *Usage:* For dynamic connectivity problems and merging groups.

- **Tries:**
  - *Usage:* Fast prefix-based lookups, such as in auto-complete features.

- **Segment Trees / Binary Indexed Trees:**
  - *Usage:* Efficient dynamic range queries and updates.

---

## 3. Techniques and Tricks

### Two-Pointer Problems
- **When to Use:** For problems requiring evaluation of pairs or subarrays/substrings.
- **Tricks:**
  - **Sorting:** Sort the array (if unsorted) to simplify pointer movement.
  - **Initialize Two Pointers:** Typically, one at the start and one at the end.
  - **Convergence:** Adjust pointers based on current sum or condition.
- **Example:**  
  *Problem:* Find two numbers in an array that add up to a target.  
  *Approach:* Sort the array, set pointers at both ends, and move them inward based on whether the sum is too high or too low.

### Sliding Window Problems
- **When to Use:** For problems involving subarrays or substrings with a dynamic "window."
- **Tricks:**
  - **Dynamic Window Adjustment:** Expand the window to meet the condition and contract it when necessary.
  - **Hash Map for Frequency:** Track counts or indices within the window.
- **Example:**  
  *Problem:* Find the longest substring without repeating characters.  
  *Approach:* Use two pointers to maintain a window and a hash map to store the last seen positions of characters.

### Dynamic Programming (DP)
- **When to Use:** For problems with overlapping subproblems and optimal substructure.
- **Tricks:**
  - **Memoization:** Cache computed results using arrays or hash maps.
  - **Clear State Definition:** Define what each state (e.g., `dp[i]`) represents.
- **Example:**  
  *Problem:* Longest Increasing Subsequence.  
  *Approach:* Define `dp[i]` as the length of the longest subsequence ending at index `i` and build up the solution iteratively.

### Graph Traversals
- **When to Use:** For problems modeling networks, connectivity, or pathfinding.
- **Tricks:**
  - **BFS vs. DFS:** Use BFS for shortest paths or level order traversal; use DFS for connectivity and cycle detection.
  - **Mark Visited:** Use a set or a boolean array to track visited nodes.
- **Example:**  
  *Problem:* Check if a graph is bipartite.  
  *Approach:* Use BFS to color nodes alternately and check for conflicts.

### Backtracking Problems
- **When to Use:** For exploring all possibilities (e.g., permutations, combinations).
- **Tricks:**
  - **Recursive Exploration:** Try out all possibilities and backtrack when a path doesn't lead to a valid solution.
  - **Early Pruning:** Discard paths that cannot possibly lead to a valid solution.
- **Example:**  
  *Problem:* Generate all subsets of a set.  
  *Approach:* Use recursion to include/exclude elements and backtrack accordingly.

---

## 4. Additional Tips

- **Start Simple:** Begin with a brute-force solution to understand the problem, then optimize.
- **Sketch Your Ideas:** Draw diagrams or write pseudocode to outline your approach.
- **Practice Regularly:** Recognize patterns by solving multiple problems.
- **Combine Data Structures:** Some challenges might require hybrid solutions (e.g., a hash map paired with a doubly linked list for an LRU cache).

---

Happy coding and good luck with your LeetCode challenges!

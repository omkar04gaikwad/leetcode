**Ultimate Guide to Solving Graph Problems with DFS and BFS**

---

## 🧠 Introduction
Graph problems come in many flavors: pathfinding, cycle detection, connectivity, ordering, and optimization. Most of them boil down to using either **Depth-First Search (DFS)** or **Breadth-First Search (BFS)**. This document gives you a comprehensive and crystal-clear way to identify which algorithm to use, how to structure your code, and how to modify it based on specific problem constraints.

---

## 🔢 Identify Graph Problem Types
Before choosing DFS or BFS, ask:

1. **What is the goal?**
   - All paths?
   - Shortest path?
   - Any valid path?
   - Topological order?
   - Use all edges once?

2. **What constraints exist?**
   - Lexicographical order?
   - Weighted/unweighted edges?
   - Can nodes/edges be reused?
   - Directed or undirected?

---

## 🧰 DFS: Depth-First Search
Used when:
- You want to explore all possibilities (backtracking)
- You want post-order behavior (e.g., use all edges before adding to result)
- You want to find cycles
- You want all paths

### 📚 DFS Recursion Template
```python
visited = set()
def dfs(node):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor)
```

### 🔹 DFS with Path Tracking (All Paths)
```python
path = []
result = []

def dfs(node):
    path.append(node)
    if node == TARGET:
        result.append(path[:])
    else:
        for neighbor in graph[node]:
            dfs(neighbor)
    path.pop()
```

### 🔹 Post-order DFS (e.g., Leetcode 332 - Reconstruct Itinerary)

In post-order DFS, you recursively visit all destinations before adding the current node to the result. This is especially useful when you need to construct an itinerary or topological order after consuming all outgoing edges.

Here’s the implementation:
```python
route = []

def dfs(node):
    while graph[node]:
        next_city = heapq.heappop(graph[node])
        dfs(next_city)
    route.append(node)  # post-order
```

**Visual Stack Unwinding Example**
Imagine this graph:
```
JFK -> ATL
JFK -> SFO
ATL -> LAX
```
DFS starts from JFK and goes deep:
- dfs("JFK")
  - dfs("ATL")
    - dfs("LAX") → append "LAX"
  → append "ATL"
  - dfs("SFO") → append "SFO"
→ append "JFK"

So the stack unwinds in the reverse visiting order: `['LAX', 'ATL', 'SFO', 'JFK']`, and you return `[::-1]` to get the itinerary.
```python
route = []

def dfs(node):
    while graph[node]:
        next_city = heapq.heappop(graph[node])
        dfs(next_city)
    route.append(node)  # post-order
```

### 🔹 DFS Topological Sort
```python
visited = set()
stack = []

def dfs(node):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor)
    stack.append(node)  # post-order

for node in graph:
    if node not in visited:
        dfs(node)

stack.reverse()
```

---

## 🌐 BFS: Breadth-First Search
Used when:
- You want the shortest path in **unweighted** graphs
- You want level-by-level traversal

**Visual Example: Level-wise Traversal**
```
Graph:
    1
   / \
  2   3
 /     \
4       5

Level 0: [1]
Level 1: [2, 3]
Level 2: [4, 5]
```
BFS explores level by level:
- Visit node 1 → enqueue [2, 3]
- Visit node 2 → enqueue [4]
- Visit node 3 → enqueue [5]
- Visit nodes 4 and 5 → done

### 📚 BFS Template
```python
from collections import deque

def bfs(start):
    queue = deque([start])
    visited = set([start])

    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### 🔹 BFS for Shortest Path
```python
from collections import deque
queue = deque([(start, [start])])
visited = set([start])

while queue:
    node, path = queue.popleft()
    if node == TARGET:
        return path
    for neighbor in graph[node]:
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
```
Used when:
- You want the shortest path in **unweighted** graphs
- You want level-by-level traversal

### 📚 BFS Template
```python
from collections import deque

def bfs(start):
    queue = deque([start])
    visited = set([start])

    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### 🔹 BFS for Shortest Path
```python
from collections import deque
queue = deque([(start, [start])])
visited = set([start])

while queue:
    node, path = queue.popleft()
    if node == TARGET:
        return path
    for neighbor in graph[node]:
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
```

---

## ⚖️ Choosing DFS vs BFS
| Goal                     | Choose | Notes                                  |
|--------------------------|--------|----------------------------------------|
| Find **any path** A → B       | DFS    | Use recursion or iterative             |
| Find **shortest path**   | BFS    | Unweighted graph                       |
| **All paths**            | DFS    | Use path list + backtracking           |
| **Topological sort**     | DFS    | Post-order traversal                   |
| **Cycle detection**      | DFS    | Track recursion stack or visited set   |
| **Use all edges once**   | DFS    | Eulerian path (Hierholzer's Algorithm) |

---

## 📈 DFS vs BFS Comparison Table

| Feature           | DFS                      | BFS                      |
|-------------------|---------------------------|---------------------------|
| Space             | O(h) (h = height)         | O(w) (w = width)          |
| Use Case          | Backtracking, Topo, All Paths | Shortest Path, Level Traversal |
| Supports pruning  | Yes                       | Sometimes (visited early) |
| Easy to go deep   | Yes                       | No                        |

---

### 🔄 Decision Tree for DFS vs BFS
```
          +----------------------------+
          |      Problem Type          |
          +----------------------------+
                     |
        +------------+-------------+
        |                          |
  Need shortest path?       Track full path?
        |                          |
      Yes                        Yes
        |                          |
      BFS                        DFS
                                  |
                    +-------------+-------------+
                    |                           |
     Is first valid path enough?     Need all combinations?
                    |                           |
                  Yes                         Yes
                    |                           |
        Return early with DFS         Explore all paths with DFS
```
| Feature           | DFS                      | BFS                      |
|-------------------|---------------------------|---------------------------|
| Space             | O(h) (h = height)         | O(w) (w = width)          |
| Use Case          | Backtracking, Topo, All Paths | Shortest Path, Level Traversal |
| Supports pruning  | Yes                       | Sometimes (visited early) |
| Easy to go deep   | Yes                       | No                        |

---

## 🔁 Flowchart for DFS vs BFS Decision
```
Track full path?
     |
   Yes
     |
    DFS
     |
Best result = first solution found?
     |             \
   Yes             No
     |               |
 Return early   Explore all paths
```

---

## 🧠 Pro Tips for Interviews
- **"Use all tickets"** → Think post-order DFS (LeetCode 332)
- **"Shortest path"** → Think BFS unless weights involved
- **"All paths"** → Think DFS with backtracking
- **"Lexical order"** → Sort or use `heapq` (priority queue)
- **Weighted shortest** → Use Dijkstra (not DFS/BFS)
- **Edge constraints** → Avoid visited set; remove edges when used

---

## 🔗 Graph Representation Tips
- Use `defaultdict(list)` or `defaultdict(heapq)`
- For undirected graphs: add both ways `graph[a].append(b)` and `graph[b].append(a)`
- For weighted graphs: `graph[u].append((v, weight))`

---

## 🎡 Practice Problems

**Visual Walkthrough: Leetcode 797 - All Paths from Source to Target**

```
Graph:
0 -> 1 -> 3
 \        ↑
  -> 2 ----
```
DFS Traversal:
- Start at node 0
  - Go to 1 → Go to 3 → Path: [0,1,3]
  - Backtrack
  - Go to 2 → Go to 3 → Path: [0,2,3]

Result:
```
[[0,1,3], [0,2,3]]
```

This shows backtracking in action as DFS explores all possible routes.

| Problem                        | Type                 | Algorithm     |
|-------------------------------|----------------------|---------------|
| Leetcode 797                  | All paths            | DFS + path    |
| Leetcode 332                  | Use all edges once   | Post-order DFS|
| Leetcode 207 / 210            | Topo Sort            | DFS/BFS       |
| Leetcode 200 / 547            | Connected Components | DFS/BFS       |
| Leetcode 743                  | Weighted Shortest    | Dijkstra      |
| Leetcode 133 / 417 / 994      | Multi-source BFS     | BFS           |

---

## 🔮 Final Mindset
> "Graph problems test your ability to track state over time. DFS is surgical, BFS is systematic. Know when to go deep, and when to go wide."

Practice recognizing the patterns. Build DFS/BFS muscle memory. You’ll go from confused to confident.

---

Let me know if you want a visual markdown version or quiz-based practice next!


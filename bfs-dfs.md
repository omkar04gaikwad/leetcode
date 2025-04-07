**Ultimate Graph Guide: Mastering DFS and BFS**

---

## 🧠 Introduction
Graphs are everywhere—networks, maps, relationships, task scheduling, and more. Solving graph problems effectively means choosing the right traversal strategy: **Depth-First Search (DFS)** or **Breadth-First Search (BFS)**. This guide combines clarity, structure, and depth from multiple resources to give you the ultimate reference for understanding, choosing, and implementing DFS/BFS.

Whether you're preparing for interviews or building intuition, this guide will help you:
- Identify which graph strategy to use
- Understand key use-cases and code templates
- Master recursion, iteration, path tracking, and traversal control
- Use visuals and real problems to reinforce concepts

---

## 🔢 Identify Graph Problem Types
Before jumping into code, ask these two questions:

### 1. What’s the goal?
- ✅ Find **any** valid path? → DFS
- ✅ Find **shortest** path (unweighted)? → BFS
- ✅ Find **all** possible paths? → DFS with backtracking
- ✅ Sort tasks or detect cycles? → Topological sort using DFS/BFS
- ✅ Use **every edge once**? → Post-order DFS (Eulerian path)
- ✅ Explore all nodes or components? → Either DFS or BFS

### 2. What constraints apply?
- ✅ **Lexicographical order**? → Sort neighbors or use a heap
- ✅ **Weighted or unweighted** graph?
  - Unweighted → BFS
  - Weighted → Dijkstra or A*
- ✅ Can nodes/edges be reused?
- ✅ Directed or undirected?

Use your answers to guide your algorithm choice in the sections ahead.

---

## 🧰 Depth-First Search (DFS)
DFS is used when you want to go **deep** before you go **wide**. It's perfect for:
- Exploring all paths
- Backtracking solutions (e.g., all permutations, paths)
- Detecting cycles
- Using all edges (e.g., Eulerian path)
- Topological sorting (post-order)

### 📚 DFS Recursion Template
```python
visited = set()

def dfs(node):
    visited.add(node)
    for neighbor in graph.get(node, []):
        if neighbor not in visited:
            dfs(neighbor)
```

### 🔁 DFS Iterative Template
```python
def dfs_iterative(start):
    stack = [start]
    visited = set()
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            for neighbor in reversed(graph.get(node, [])):
                if neighbor not in visited:
                    stack.append(neighbor)
    return visited
```

### 🔹 DFS with Path Tracking (All Paths from Source to Target)
Used when you need to generate **all** valid paths:
```python
path = []
result = []

def dfs_paths(node):
    path.append(node)
    if node == TARGET:
        result.append(path[:])
    else:
        for neighbor in graph.get(node, []):
            dfs_paths(neighbor)
    path.pop()  # backtrack
```

### 🔹 Post-order DFS (Leetcode 332: Reconstruct Itinerary)
This is DFS where we add the current node **after** visiting all its neighbors:
```python
import heapq

graph = defaultdict(list)
# heapq used for lexical order
for a, b in tickets:
    heapq.heappush(graph[a], b)

route = []

def dfs(node):
    while graph[node]:
        next_city = heapq.heappop(graph[node])
        dfs(next_city)
    route.append(node)  # post-order

# Start from "JFK"
dfs("JFK")
return route[::-1]
```

This approach helps when you must **use all edges once** and need **reverse order** on the way back.

### 🔹 DFS for Topological Sort
Used in problems like course schedule and task dependency resolution:
```python
visited = set()
stack = []

def dfs(node):
    visited.add(node)
    for neighbor in graph.get(node, []):
        if neighbor not in visited:
            dfs(neighbor)
    stack.append(node)  # post-order

for node in graph:
    if node not in visited:
        dfs(node)

stack.reverse()  # topological order
```

---

## 🌐 Breadth-First Search (BFS)
BFS is a level-by-level traversal used when:
- You want the **shortest path** in an unweighted graph
- You need to explore **all neighbors before going deeper**
- You solve **multi-source problems** (like rotten oranges, network delay)

### 📚 BFS Template
```python
from collections import deque

def bfs(start):
    queue = deque([start])
    visited = set([start])

    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### 🔹 BFS for Shortest Path (Unweighted Graph)
Tracks the path along with traversal:
```python
from collections import deque

def bfs_shortest_path(start, target):
    queue = deque([(start, [start])])
    visited = set([start])

    while queue:
        node, path = queue.popleft()
        if node == target:
            return path
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None
```

### 🔹 Multi-Source BFS
Used when you need to spread or grow from multiple starting nodes simultaneously.
```python
from collections import deque

def multi_source_bfs(sources):
    queue = deque(sources)
    visited = set(sources)

    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

---

## ⚖️ DFS vs BFS Comparison

### 📊 Quick Comparison Table
| Feature           | DFS                             | BFS                            |
|------------------|----------------------------------|----------------------------------|
| Space Complexity | O(h) (stack depth)              | O(w) (queue width)              |
| Use Cases        | All paths, topo sort, backtrack | Shortest path, level traversal  |
| Early Termination| Yes                             | Yes                             |
| Go deep first?   | Yes                             | No                              |
| Easy to implement| Recursive                       | Iterative                       |

### 🔄 Decision Flow
```
Need shortest path? → Yes → BFS
                         ↓
                   No
                   ↓
Track full path? → Yes → DFS with backtracking
                         ↓
                   No
                   ↓
Using all edges? → Yes → Post-order DFS
                         ↓
                   No
                   ↓
Topological order? → Yes → DFS (or BFS with Kahn's)
```

---

## 💡 Pro Interview Tips
- 🔍 If you see "shortest path" in an **unweighted** graph → **BFS**
- 🔄 If you must "use all edges once" → **Post-order DFS**
- 📊 If there's an ordering or schedule → **Topological Sort (DFS/BFS)**
- 🧠 For all possible combinations or paths → **DFS with backtracking**
- 🔤 For lexicographical order → **Sort neighbors or use `heapq`**
- ⚖ For weighted shortest paths → **Dijkstra**, not DFS/BFS

---

## 🧪 Practice Problems (LeetCode)

| Problem                             | Type                     | Technique           |
|-------------------------------------|--------------------------|---------------------|
| 797. All Paths From Source to Target| All paths                | DFS + path tracking |
| 332. Reconstruct Itinerary          | Use all edges once       | Post-order DFS      |
| 207 / 210. Course Schedule          | Topological Sort         | DFS / BFS           |
| 200. Number of Islands              | Connected Components     | DFS / BFS           |
| 743. Network Delay Time             | Shortest Path (weighted) | Dijkstra            |
| 994. Rotting Oranges                | Multi-source BFS         | BFS                 |
| 133. Clone Graph                    | Graph traversal          | DFS / BFS           |
| 417. Pacific Atlantic Water Flow    | Reverse multi-source BFS | BFS / DFS           |

---

## 🧠 Final Mindset
> Graph problems test your ability to track state over time. DFS is surgical. BFS is systematic. Know when to go deep, and when to go wide.

Practice the decision flow until it's second nature. Once the pattern clicks, graphs stop feeling mysterious.

---

Want this as a PDF or markdown export? Or want to add a section on Dijkstra + A* next?


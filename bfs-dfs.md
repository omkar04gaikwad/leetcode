
# 🚀 BFS & DFS Patterns in Leetcode

This guide helps you identify, classify, and solve graph-related problems using BFS and DFS with full pseudocode and decision logic.

---

## 🧩 1. Reachability (Single Source)

**🔍 Key Identifier**:  
- “Can I reach all nodes from node 0?”
- Input looks like a graph (adjacency list/matrix).
- Usually only one DFS/BFS is needed.

**✅ Use**:  
DFS (iterative or recursive) or BFS  
Only one traversal is needed from the source.

**🧠 Examples**:  
- Leetcode 841: Keys and Rooms  
- Leetcode 1466: Reorder Routes  

**📝 Pseudocode (DFS Stack)**:
```
stack = [start]
visited = set()

while stack:
    node = stack.pop()
    if node not in visited:
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                stack.append(neighbor)
```

---

## 🧩 2. Connected Components

**🔍 Key Identifier**:  
- “How many groups/components?”
- You must visit every node.
- Graph is undirected or directed.

**✅ Use**:  
DFS or BFS in a loop  
You must run DFS/BFS for each unvisited node and count.

**🧠 Examples**:  
- Leetcode 547: Number of Provinces  
- Leetcode 200: Number of Islands

**📝 Pseudocode**:
```
count = 0
visited = set()

for node in range(n):
    if node not in visited:
        dfs(node)
        count += 1

function dfs(node):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor)
```

---

## 🧩 3. Shortest Path (Unweighted Graph)

**🔍 Key Identifier**:  
- “What’s the fewest steps?”
- No weights on edges.
- Grid or list-based traversal.

**✅ Use**:  
BFS (guarantees shortest path in unweighted graphs)

**🧠 Examples**:  
- Leetcode 752: Open the Lock  
- Leetcode 1091: Shortest Path in Binary Matrix

**📝 Pseudocode**:
```
queue = [(start, 0)]
visited = set([start])

while queue:
    node, level = queue.pop(0)
    if node == target:
        return level
    for neighbor in graph[node]:
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append((neighbor, level + 1))
```

---

## 🧩 4. Value Propagation / Evaluation

**🔍 Key Identifier**:  
- “Find value through path traversal”
- Path may not exist; return -1 if so
- Graph is weighted

**✅ Use**:  
DFS with multiplication or tracking path value

**🧠 Examples**:  
- Leetcode 399: Evaluate Division

**📝 Pseudocode**:
```
function dfs(curr, target, product):
    if curr == target:
        return product
    visited.add(curr)
    for neighbor, value in graph[curr]:
        if neighbor not in visited:
            result = dfs(neighbor, target, product * value)
            if result != -1:
                return result
    return -1
```

---

## 🧩 5. Cycle Detection

**🔍 Key Identifier**:  
- “Does the graph contain a cycle?”
- DFS usually with tracking parents or recursion stack.

**✅ Use**:  
DFS with parent tracking (undirected), color states (directed)

**🧠 Examples**:  
- Leetcode 207: Course Schedule

**📝 Pseudocode** (Directed - color based):
```
WHITE, GRAY, BLACK = 0, 1, 2
color = {node: WHITE for node in graph}

function dfs(node):
    if color[node] == GRAY:
        return True  # cycle
    if color[node] == BLACK:
        return False
    color[node] = GRAY
    for neighbor in graph[node]:
        if dfs(neighbor):
            return True
    color[node] = BLACK
    return False
```

---

## 🧩 6. Topological Sorting

**🔍 Key Identifier**:  
- “What is the order to complete tasks?”
- Graph is a DAG (Directed Acyclic Graph)

**✅ Use**:  
DFS postorder or BFS with in-degree (Kahn’s Algorithm)

**🧠 Examples**:  
- Leetcode 210: Course Schedule II

**📝 Pseudocode (DFS version)**:
```
visited = set()
stack = []

function dfs(node):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor)
    stack.append(node)

for node in graph:
    if node not in visited:
        dfs(node)

stack.reverse()  # Topological order
```

---

## ❗ Choosing Between DFS and BFS

| Problem Type                 | Use          | Reason                                 |
|-----------------------------|--------------|----------------------------------------|
| Reachability                | DFS/BFS      | One traversal from source              |
| Count Components            | DFS/BFS loop | Multiple independent groups            |
| Shortest Path (Unweighted)  | BFS          | Layered exploration gives shortest path|
| All Paths / Backtracking    | DFS          | Explore all routes                     |
| Topological Sort            | DFS/BFS      | Postorder or in-degree based           |
| Evaluate Equations          | DFS          | Value propagation via traversal        |
| Minimum Reorientation       | DFS/BFS      | Track original direction during visit  |

---

## ✅ Final Notes

- Always check:
  - Directed vs Undirected?
  - Weighted?
  - Start from all or one node?
  - Need path, count, or value?
- Trace sample inputs manually to verify approach.


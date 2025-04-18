# Detailed Approach To Solve [Code Land](https://www.codechef.com/problems/PRCNSR2?tab=statement) on Codechef

## 1. Understanding the problem
#### 1. We are given an undirected weighted connected graph without multiple edges. 
#### 2. We need to remove exactly 1 edge so the graph doesn't stay connected.
#### 3. Then we need to add exactly 1 edge (possibly the same edge which we removed) to connect the graph again.
#### 4. We need to do these operations to achieve maximum gain in strength(Total sum of weight of all edges).

## 2. Understanding the given test case
#### INPUT:
1\
6 7 1000000\
1 2\
2 3\
3 1\
4 5\
5 6\
6 4\
3 4

OUTPUT:
24

#### Let's represent this with a diagram.

![Graph1](codeland_sample1.svg)

####
Strength/weight of edges\
Base Cases:
f(1, y) = 1 % MOD
f(x, 1) = 1 % MOD
f(1, 1) = 0

Function Definition:
if x != y : f(x, y) = ( f(x-1, y) + f(x, y-1) + f(x-1, y-1) ) % MOD
else : f(x, y) = 0

|| 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| **1** | 0 | 1 | 1 | 1 | 1 | 1 |
| **2** | 1 | 0 | 2 | 4 | 6 | 8 |
| **3** | 1 | 2 | 0 | 6 | 16 | 30 |
| **4** | 1 | 4 | 6 | 0 | 22 | 68 |
| **5** | 1 | 6 | 16 | 22 | 0 | 90 | 
| **6** | 1 | 8 | 30 | 68 | 90 |0 |


#### After adding weights to the diagram

![Graph2](codeland_sample2.svg)

#### Graph representing the solution to the problem

![Graph3](codeland_sample3.svg)

#### We observe that only if the edge (3,4) is removed from the graph then the graph gets disconnected. There is no other edge which could be removed solely to disconnect the graph.

#### Now using the King's superpower we can possibly replace it with another edge so that the graph still stays connected.

#### After removing the edge (3,4) with weight 6, we have 2 connected components (1,2,3) and (4,5,6). Now we need to find an edge from one of the nodes in (1,2,3) to one of the nodes in (4,5,6) with the maximum weight. We find that the edge (6,3) is of maximum weight 30.

#### Therefore maximum gain in strength = 30-24 =6

## 3. Breaking down the problem.
#### 1. We need to find the weights of the graph.
#### 2. We need to find the list of edges of the graph such that if exactly 1 edge from that list is removed from the graph then the graph gets disconnected.
#### 3. We need to find an edge from that list and remove it which would disconnect the graph and then find its maximum strength replacement edge which would connect the graph and maximize the gain in strength.

## 4. Calculation of strength(weight) of roads
#### Given:
```
f(1,y)=1%MOD
f(x,1)=1%MOD
f(1,1)=0
if x != y : f(x, y) = ( f(x-1, y) + f(x, y-1) + f(x-1, y-1) ) % MOD
else : f(x, y) = 0
```

#### We can write a simple recursive function to find the strength of a road.
```c++
int strength(int x, int y){
    if(x==1&&y==1){
        return 0;
    } else if(x==1 || y==1){
        return 1%MOD;
    } else {
        return (strength(x-1,y)+strength(x,y-1)+strength(x-1,y-1))%MOD;
    }
}
```
#### Time and Space Complexity of the above function
```
Time Complexity: O(3<sup>(x+y)</sup>)
Because if we draw the recursion tree we will find 3<sup>(x+y)</sup> nodes in the tree.
Space Complexity: O(3*(max(x,y)))
Because there will be at most that 3*(max(x,y)) nodes in the recursion stack at a time.
```
#### Its exponential and undesirable. We can optimize it drastically by using memoization. 
```c++
// Initialize the dp vector to store and lookup strength values
void initialize(int n, int MOD, vector<vector<int>>& dp) {
    vector<vector<int>> dp(n+1);
    for(int i = 1; i <= n; i++) {
        dp[i].assign(n+1, -1);
    }
}

// If dp[x][y] = -1 then we calculate the strength and update dp[x][y] and then return it. Othrwise we just return it. 
int strength(int x, int y, vector<vector<int>>& dp){
    int& ans = dp[x][y];
    if(ans != -1) {
        return ans;
    }
    if(x==1&&y==1){
        ans = 0;
    } else if(x==1 || y==1){
        ans = 1%MOD;
    } else {
        ans = (strength(x-1,y)+strength(x,y-1)+strength(x-1,y-1))%MOD;
    }
    return ans;
}
```
#### Time and Space Complexity of the above function
```
Time Complexity: O(xy)
Because overall there are xy UNIQUE nodes of f(x,y)'s recursion tree and each will be called once. By definition of memoization technique we won't calculate a function if it's already calculated once.
Space Complexity: O(3*(max(x,y)))
Because there will be at most that 3*(max(x,y)) nodes in the recursion stack at a time.
```
#### This is a TOP DOWN approach to solve the problem. i.e. We start with the calculation of f(x,y) and in order to solve it we need to solve the smaller subproblems in its recursion tree.
#### Alternatively we can approach this problem in a BOTTOM UP way. Where we first calculate the smallest problems and then use their results to solve bigger problems.
```c++
// Initialize the dp vector to store and lookup strength values
void initialize(int n, int MOD, vector<vector<int>>& dp) {
    vector<vector<int>> dp(n+1);
    for(int i = 1; i <= n; i++) {
        dp[i].resize(n+1);
        for(int j = 1; j <= n; j++) {
            if(i==1&&j==1){
                dp[i][j]=0;
            } else if(i==1||j==1){
                dp[i][j]=1%MOD;
            } else {
                dp[i][j]=(dp[i-1][j]+dp[i][j-1]+dp[i-1][j-1])%MOD;
            }
        }
    }
}
int strength(int x, int y, vector<vector<int>>& dp){
    return dp[x][y];
}
```
#### Time Complexity of the above function
#### O(1) to answer any query
#### O(n<sup>2</sup>) to precompute. Because we have 1 loop inside another both running n times.

## 6. Find the list of edges of the graph such that if exactly 1 edge from that list is removed from the graph then the graph gets disconnected.
#### It turns out that this is a standard problem. This list of edges are called bridges of a graph.
#### A bridge is an edge in an undirected graph which when removed disconnects the graph. i.e. If edge(u,v) is a bridge of a graph then there is only a single path between u and v and it's the edge (u,v).
#### 1. Brute Force approach to find bridges of a connected graph is to loop through each edge(u,v), remove it and then run DFS from u. If your'e not able to find v in the DFS then the edge(u,v) is a bridge otherwise it's not a bridge. Time Complexity O(M(N+M))
```c++
    class Edge {
        public:
            int u;
            int v;
        Edge(u,v): u(u), v(v){}
    };

    bool dfs(int u, vector<vector<int>>& adjacency, vector<bool>& visited, Edge possibleBridge) {
        visited[u] = true;
        bool isABridge=(u!=possibleBridge.v); // if u == possibleBridge.v then we found it's not a bridge
        for (int to : adj[u]) {
            if(u==possibleBridge.u && to==possibleBridge.v){//Skip the possible bridge
                continue;
            }
            if (!visited[to]) {
                if(!dfs(to, adjacency, visited, possibleBridge)){//if we were able to find possibleBridge.v in the DFS then we say isABridge=false otherwise we return true
                    isABridge=false;
                }
            }
        }
        return isABridge;
    }

    vector<Edge> findBridges(vector<vector<int>>& adjacency, vector<Edge>& edges) {
        vector<Edge> bridges;
        for(Edge edge: edges) {
            vector<bool> visited(adjacency.size(), false);
            if(dfs(edge.u, adjacency, visited, edge)){
                bridges.push_back(edge);
            }
        }
        return bridges;
    }
```
#### Time Complexity of the above function O(M(N+M)). M for looping the edges and N+M for the DFS inside the loop.

#### Tarjan's approach using dynamic programming.
Let's introduce a concept of backedge with some examples. If during DFS of a node U we find an edge from one of it's descendants back to U [apart from the edge taken in the DFS from U] then we call that edge a back edge. Let's see some examples.

![Graph2](codeland_sample4.svg)
#### We will be running a simple DFS but we will be storing and using some more information to tell us whether there's a backedge to the current node (let's say U) from one of its descendants(the edge taken in the DFS from U).
#### We will be using a counter in the DFS. Each time we are running DFS for an unvisited node, we increment that counter. Let's call that counter *timer*. This counter will help us determine the order of discovery of nodes in the DFS.
#### Let's define an array tin[]. tin\[v\] will store the time in which DFS was run for v or the discovery time of v.
#### Let's define an array low[]. low\[v\] will store the minimum discovery time of(all the nodes in the subtree of v or adjacent to any of the nodes in the subtree of v) in the DFS. The values in this array will help us determine if there exists a back edge from any of the nodes in the subtree of v or adjacent to any of the nodes in the subtree of v to v or any of its ancestors.
```c++
class TarjansBridgesFinder {
    class Edge {
        public:
            int u;
            int v;
            Edge(int u, int v) : u(u), v(v) {}
            Edge(){}
    };

    public:

    int n,m; // n = number of nodes, m = number of edges
    vector<vector<int> > adj; // adjacency list of graph
    vector<bool> visited; // visited array
    vector<int> tin, tout, low; // tin = time of insertion, tout = time of leaving, low = lowest time of insertion
    int timer; // timer
    vector<Edge> bridges; // list of bridges
    TarjansBridgesFinder(int n, int m) : n(n), m(m), adj(n) {}

    void addEdge(int x, int y) {
        adj[x].push_back(y);
        adj[y].push_back(x);
    }

    void dfs(int v, int p = -1) {
        visited[v] = true;
        tin[v] = low[v] = timer++;//low[v] initialized to timer
        bool parent_skipped = false;
        for (int to : adj[v]) {
            if (to == p && !parent_skipped) {// parent_skipped is used to handle multiple edges
                parent_skipped = true;
                continue;
            }
            if (visited[to]) {
                low[v] = min(low[v], tin[to]);//low[v] is minimum of low[v] and discovery time of its adjacent nodes.
            } else {
                dfs(to, v);
                low[v] = min(low[v], low[to]);//low[v] is minimum of low[v] and low[to] for to in subtree of v.
                if (low[to] > tin[v]) {//backedge from any node in node to's subtree or adjacent to a node in to's subtree
                    addBridge(v, to);
                }
            }
        }
    }

    void addBridge(int u, int v) {
        if(u>v){
            swap(u,v);
        }
        bridges.push_back(Edge(u,v));
    }

    void findBridges() {
        time = 0;
        visited.assign(n, false);
        tin.assign(n, -1);
        low.assign(n, -1);
        for (int i = 0; i < n; ++i) {
            if (!visited[i])
                dfs(i);
        }
    }

};
```
[See DFS Visualization Video](https://youtu.be/Men4lQiJUDo)

Time Complexity is O(N+M) N is number of vertices and M is number of edges. You can find the details of the algorithm and it's implementation in the link https://cp-algorithms.com/graph/bridge-searching.html

## 5. Approaches to find the best edge and replacement edge pair to maximize the gain in strength.
### 1st Approach
#### 1. Loop through all the bridges and for each bridge find 2 connected components as a result of removing it.
#### 2. Then loop through nodes in first component and within this loop iterate through nodes in the second component and then find the maximum strength edge among them. This maximum strength edge could be used to replace that bridge.
#### This way we could find the maximum gain in strength. But the problem with this approach is high Time complexity O(n<sup>3</sup>) and there's no room for optimization in this approach.

### 2nd approach
 #### 1. Loop through all the nodes and within this loop loop again through all the nodes. 
 #### 2. For each pair of nodes in the loop find a bridge with maximum weight in the path between the 2 nodes. 
 #### 3. You can possibly remove that bridge with an edge between these nodes. It's also possible that there won't be any bridges in the path between the nodes and in that case the selection of these nodes won't be a valid edge replacement. 
 #### This again needs O(n<sup>3</sup>) time complexity. We optimize this approach by using a bridge tree and binary lifting in O(n<sup>2</sup>log(k)) where k is the number of bridges in the tree.

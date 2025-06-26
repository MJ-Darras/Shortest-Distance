class MinHeap():
    """
    Class representing the minimum-heap data structure.
    Credit: The skeleton from which this min-heap was built upon was inspired from FIT1008's max heap. 
    """
    def __init__(self, size):
        """
        Initializes the heap with a fixed size.
            Parameters
                size: the size of the heap.
            Returns
                None
            Complexity
                Time and Space: O(n) for the array creation, where n is the size of the list;
                in this implementation n is the number of vertices in graph.
        """
        self.array = [None] * (size + 1)    # as first index is not used
        self.heap_length = 0                # two lengths, used to maintain order in the heap.              

    def is_full(self) -> bool:
        """
        Checks if the heap is full.
            Parameters
                None
            Returns
                True/False depending on the heap's fullness.
            Complexity
                Time and Space: O(1) for checking the fullness of the heap.
        """
        return self.heap_length + 1 == len(self.array)
    
    def rise(self, k):
        """
        Rise element at index k to its correct position.
            Parameters
                k: the position of the element to rise
            Returns
                None
            Preconditions
                k: 1 <= k <= self.length
            Complexity
                Time: O(log n) for the rise, where n is the size/length of the current heap.
                Space: O(1) for the space used to aid in performing the swaps.
        """
        item = self.array[k]                                    # Store the element at index
        while k > 1 and item[1] < self.array[k // 2][1]:        # While its value is less than its parent, in this implementation we store cost at state position 1.
            self.array[k] = self.array[k // 2]                  # Swap elements, update k and index of vertex,
            self.array[k // 2][3].index = k                     # in this implementation we store the vertex object at state position 3.
            k = k // 2
        self.array[k] = item                                    # Once you can no longer rise, place the item at the final correct index
        self.array[k][3].index = k                              # Update the index of the vertex

    def sink(self, k):
        """
        Sink element at index k to its correct position.
            Parameters
                k: the position of the element to sink.
            Returns
                None.
            Preconditions
                k: 1 <= k <= self.length
            Complexity
                Time: O(log n) for the sink, where n is the size/length of the current heap.
                Space: O(1) for the space used to aid in performing the swaps.
        """
        item = self.array[k]                        # Store element at index
        while 2 * k <= self.heap_length:            # While k has children
            min_child = self.smallest_child(k)      #Find the smallest child of k
            if self.array[min_child] >= item:       # If the item is not smaller than the smallest child, break
                break
            self.array[k] = self.array[min_child]   # Swap the elements, update the index and make k point to the minimum child
            self.array[min_child][3].index = k
            k = min_child

        self.array[k] = item                        # Once sinking is done, place the item at the corresponding index and update the index of the vertex

        if self.array[k] != None:                   # This is done to avoid errors when the heap is empty, usually at the start of ordering
            self.array[k][3].index = k
    
    def smallest_child(self, k: int) -> int:
        """
        Finds the index of the smallest child of node at k.
            Parameters
                k: the index of the parent node.
            Returns
                index of k's child with smallest value.
            Preconditions
                k: 1 <= k <= self.length // 2
            Complexity
                Time and Space: O(1) for determining the smallest child.
        """
        if 2 * k == self.heap_length or \
                self.array[2 * k] < self.array[2 * k + 1]:  # if only one child exists, return it 
            return 2 * k                                    # if two exist, return the smaller one
        else:                                               
            return 2 * k + 1
    
    def add(self, element):
        """
        Adds an element to the end of the heap, then rises it to its correct position.
            Parameters
                element: the element to add.
            Returns
                None
            Complexity
                Time: O(log n) for the adding then using the rise, where n is the size/length of the current heap.
                Space: O(1) for the space used to aid in performing the swaps.
        """
        if self.is_full():
            raise IndexError("Heap is full")
        
        self.heap_length += 1                       # Increment length of the heap by 1, to account for the new element
        self.array[self.heap_length] = element      # Add the element to the end of the heap
        self.rise(self.heap_length)                 # Rise it to its correct position, from the end position

    def update(self, k, new):
        """
        Replaces an element at index k with a new element, then sinks/rises it to its correct position.
            Parameters
                k: the index of the element to replace.
                new: the new element to add in place of the old.
            Returns
                None
            Complexity
                Time: O(log n) for the rise, where n is the size/length of the current heap.
                Space: O(1) for the space used to aid in performing the swaps.
        """
        self.array[k] = new                  # replace the old with the new, and rise it to the correct position
        self.rise(k)                         # PS: we only account for rise as we only need to when we find a more optimal path
    
    def get_min(self):
        """
        Finds the mminimum element in the heap, removes and returns it from the heap, and readjusts the heap.
            Parameters
                None
            Returns
                Tuple (distance, Vertex) where it is the minimum element in the heap. 
            Complexity
                Time: O(log n) for the sink, where n is the size/length of the current heap.
                Space: O(1) for the space used to aid in performing the swaps.
        """
        if self.heap_length == 0:               # If the heap is empty, raise an error
            raise IndexError("Heap is empty")
        
        min_elem = self.array[1]                # Since this is a min-heap, the minimum element is always at index 1

        if self.heap_length > 1:                
            # If heap has multiple elements, swap them and bring the last element of the heap to the top
            # Store the removed minimum element at the end of the array
            self.array[1], self.array[self.heap_length]  = self.array[self.heap_length], min_elem
            # Update the indexes of vertices
            min_elem[3].index = self.heap_length
            self.array[1][3].index = 1
            # Readjust the size of the heap and the array
            self.heap_length -= 1
            # Sink the element to its correct position
            self.sink(1)

        elif self.heap_length == 1:
                # To avoid errors when the heap is empty, we creat a condition for this specific case
                self.array[1] = None 
                self.array[self.heap_length] = min_elem
                min_elem[3].index = self.heap_length
                self.heap_length -= 1

        return min_elem

class Edge:
    """
    Class representing an edge in the graph.
    """
    def __init__(self, u, v, w, t):
        """
        Initializes an edge with the incoming vertex, outgoing vertex and weight.
        Parameters
            u: the incoming vertex
            v: the outgoing vertex
            w: the weight of the edge
        Returns
            None
        Complexity
            Space and time: O(1) for the initialization of the edge.
        """
        self.u = u               # u is the incoming vertex
        self.v = v               # v is the outgoing vertex
        self.w = w               # w is the weight of the edge
        self.t = t               # t is the time taken to travel the edge

    def __str__(self):
        """
        Provides a human readable representation of the edge.
        Parameters
            None
        Returns
            A string representation of the edge.
        Complexity
            Time and Space: O(1) for the string representation.
        """
        return_string = "from " + str(self.u) + \
                        " to " + str(self.v) + \
                        ", weight: " + str(self.w) + \
                        ", time: " + str(self.t)
        return return_string
    
    def __repr__(self):
        """
        Provides a human readable representation of the edge, for ease of debugging purposes.
        Parameters
            None
        Returns
            A string representation of the edge.
        Complexity
            Time and Space: O(1) for the string representation.
        """
        return "<" + str(self.u) + ", " + str(self.v) + ", " + str(self.w) + ", " + str(self.t) + ">"

class Vertex:
    """
    Class representing a vertex in the graph.
    """
    def __init__(self, id:int, modulus:int):
        """
        Initializes a vertex with the attributes such as id, edges, discovered, visited, distance and predecessor.
        Parameters
            id: the id of the vertex.
        Returns
            None
        Complexity
            Space and time: O(1) for the initialization of the vertex.
            Analysis: Since the problem limits the number of stations to a max of 20, each having a max time of 5 minutes, we can have up to 100 states,
             which while seeming large, is still of constant complexity.
        """
        self.id = id                    # id of the vertex
        self.edges = []                 # edges consisting of <u,v,w,t> where u and v are the incoming and outgoing vertices, w being the weight and t being the cost 
        self.discovered = False         
        self.visited = False
        self.distance = float("inf")    # initially set all distances from the source to infinity
        self.predecessors = []          # to keep track of path
        self.index = None               # index in the heap
        self.time = 0                   # time taken to travel the edge
        self.current_state = None       # current state based on time % the train loop total time
        self.all_states = [None] * modulus  # an array of size modulus, i.e. the total time for the train loop
        self.added_states = []              # an array to keep track of all newly added states to be able to check them only for new possibilities
    
    def __str__(self):
        """
        Provides a human readable representation of the Vertex.
        Parameters
            None
        Returns
            A string representation of the Vertex.
        Complexity
            Time and Space: O(1) for creating the string representation.
        """
        return_string = "id: " + str(self.id) + "\n" + \
                        "discovered: " + str(self.discovered) + "\n" + \
                        "visited: " + str(self.visited) + "\n" + \
                        "distance cost: " + str(self.distance) + "\n" + \
                        "time taken: " + str(self.time) + "\n" + \
                        "edges: " + str(self.edges) + "\n" + \
                        "current state: " + str(self.current_state) + "\n" + \
                        "all states: " + str(self.all_states) + "\n" + \
                        "added states: " + str(self.added_states) + "\n"
        
        if self.predecessors != None:
            return_string = return_string + "predecessors: " + str(self.predecessors) + "\n"

        return return_string
    
    def __repr__(self):
        """
        Provides a human readable representation of the Vertex, which is usually used in debugging.
        Parameters
            None
        Returns
            A string representation of the Vertex.
        Complexity
            Time and Space: O(1) for creating the string representation.
        """
        return "V" + str(self.id)
    
    # Some comparisons to make using vertices a bit easier, all take in other vertex as a parameter and return a boolean value 
    def __lt__(self, other):   
        """
        Provides an easier way of comparing vertices distance costs, not necessary but added in case needed.
        Checks if current vertex distance cost is less than some other one.
        Parameters
            other: some object we want to compare with the current vertex (self).
        Returns
            A boolean stating if the self object's distance cost is less than the other object's distance cost.
        Complexity
            Time and Space: O(1) for the comparison
        """         
        return self.distance < other.distance
    
    def __le__(self, other):
        """
        Provides an easier way of comparing vertices distance costs, not necessary but added in case needed.
        Checks if current vertex distance cost is less than or equal some other one.
        Parameters
            other: some object we want to compare with the current vertex (self).
        Returns
            A boolean stating if the self object's distance cost is less than or equal the other object's distance cost.
        Complexity
            Time and Space: O(1) for the comparison
        """
        return self.distance <= other.distance     
    
    def __gt__(self, other):  
        """
        Provides an easier way of comparing vertices distance costs, not necessary but added in case needed.
        Checks if current vertex distance cost is greater than some other one.
        Parameters
            other: some object we want to compare with the current vertex (self).
        Returns
            A boolean stating if the self object's distance cost is greater than the other object's distance cost.
        Complexity
            Time and Space: O(1) for the comparison
        """  
        return self.distance > other.distance   
    
    def __ge__(self, other):
        """
        Provides an easier way of comparing vertices distance costs, not necessary but added in case needed.
        Checks if current vertex distance cost is greater than or equal some other one.
        Parameters
            other: some object we want to compare with the current vertex (self).
        Returns
            A boolean stating if the self object's distance cost is greater than or equal the other object's distance cost.
        Complexity
            Time and Space: O(1) for the comparison
        """  
        return self.distance >= other.distance
    
    def added_to_queue(self):
        """
        Marks the vertex as added to the queue and updates its attribute.
        Parameters
            None
        Returns
            None
        Complexity
            Time and Space: O(1) for the update.
        """
        self.discovered = True
    
    def visit_node(self):
        """
        Marks the vertex as visited and updates its attribute.
        Parameters
            None
        Returns
            None
        Complexity
            Time and Space: O(1) for the update.
        """
        self.visited = True
    
    def add_edge(self, edge):
        """
        Adds an edge to the vertex.
        Parameters
            edge: the edge to add to the vertex.
        Returns
            None
        Complexity
            Time and Space: O(1) for the addition.
        """
        self.edges.append(edge)
    
class Graph:
    """
    Class representing a graph using an adjacency list.
    """
    def __init__(self, vertex_count: int, modulus: int):
        """
        Initializes the graph, creates a list for the vertices based on the count, then initializes each vertex.
        Parameters
            vertex_count: number of vertices in the graph.
        Returns
            None.
        Complexity
            Space and Time: O(L) for initialization, with L as the number of vertices/locations.
            Analysis: both complexities depend on the vertex count only, which is a constant.
        """  
        # Initialize vertices list         
        self.vertices = [None] * vertex_count
        # Create vertices
        for i in range (vertex_count):
            self.vertices[i] = Vertex(i, modulus)
    
    def __str__(self):
        """
        Provides a string representation of all the vertices in the graph.
        Parameters
            None
        Returns
            string, representing the graph
        Complexity
            Space and timme: O(L), with L as the number of vertices/locations.
        """
        return_string = ""
        for vertex in self.vertices:
            return_string = return_string + "Vertex " + str(vertex) + "\n"
        return return_string

    def add_edges(self, argv_edges):
        """
        Add multiple edges to the graph, these edges come in as tuple of (u,v,w) where u is the incoming vertex, v is the outgoing vertex and w is the weight of the edge.
        Parameters
            argv_edges: list of edges to add to the graph.
        Returns
            None
        Complexity
            Time and Space: O(R) for the addition of edges, where R is the number of edges/roads.
        """
        for edge in argv_edges:
            u = edge[0]
            v = edge[1]
            w = edge[2]
            t = edge[3]
            # create an edge object
            current_edge = Edge(u, v, w, t)                   
            current_vertex = self.vertices[u]
            # add created edges to vertex u, i.e. the incoming vertex
            current_vertex.add_edge(current_edge)          

    def dijkstra(self, source:int, modulus: int):
        """
        Implemetation of Dijkstra's algorithm that is used to find the shortest path from some source to some destination, on a graph.
        Parameters
            source: source vertex to start searching from.
            destination: destination vertex to search for and end at.
            modulus: the total time of the train loop.
        Returns
            None
        Complexity
            Time: O(R log L), where R is the number of roads/edges and L being the number of locations/vertices.

            Analysis: We go over each edge in the graph, which takes O(R), and perform one of the min-heap's operations, which all have a worst case complexity of O(log n),
             where n is the size of the heap array, but assuming the worst case which could mean all vertices are in the heap at once, we can say O(log L).

            Space: O(L + R) for the space used to store the vertices and edges.

            Analysis: We store vertices in a priority queue, which at worst can hold up to L vertices giving us O(L), 
             and we store the states that are dependent on the edges in an adjacency list of size O(R), which gives us a total complexity of O(L + R).
             Keep in mind, this can be confused to be O(L*R), but in reality this would account for more edges and hence states than the existing ones.
        """
        # get the source vertex 
        source = self.vertices[source]                      
        source.distance = 0

        source.state = 0
        discovered = MinHeap(modulus * len(self.vertices))
        # Discovered source and added it to the queue
        discovered.add((source.state, source.distance, source.time, source, source.predecessors))   
        source.added_to_queue()                             
        # Visited source
        source.visit_node()    
        # Set the first state of the algorithm                             
        source.current_state = (0,0,0,source)
        source.all_states[0] = source.current_state

        # As long as we have new possible states to explore.
        while discovered.heap_length > 0:
            # Find the highest priority location to visit 
            u = discovered.get_min() 
            #get the vertex object                       
            u = u[3]
            # mark the current node as visited
            u.visit_node()   

            # Perform edge relaxation on all adjacent vertices
            for edge in u.edges:
                v = edge.v
                # get the vertex object
                v = self.vertices[v]                            

                # Calculate new state attributes
                v_distance = u.distance + edge.w
                v_time = u.time + edge.t
                v_state = v_time % modulus
                v_pred = u.predecessors + [u.id]

                if not v.discovered:
                    # Set them in the vertex
                    v.distance = v_distance
                    v.time = v_time
                    v.predecessors = v_pred
                    v.current_state = v_state
                    v.added_to_queue()

                    # If no existing previous data for this state
                    if v.all_states[v.current_state] == None:
                        # add it to list of states
                        v.all_states[v.current_state] = (v.current_state, v.distance, v_time, v, v.predecessors)
                        v.added_states.append(v_state)
                        # add to the queue to be explored
                        discovered.add((v.current_state, v.distance, v_time, v, v.predecessors))
                    
                    # Check out other possible states when arriving to discover a location, from added states, which only includes unchecked states
                    for state_id in u.added_states:
                        # Perform the same steps as the last code block accordingly
                        state = u.all_states[state_id]
                        if state:
                            s_distance = state[1] + edge.w
                            s_time = state[2] + edge.t
                            s_state = s_time % modulus
                            s_pred = state[4] + [u.id]

                            if v.all_states[s_state] == None:
                                v.all_states[s_state] = (s_state, s_distance, s_time, v, s_pred)
                                v.added_states.append(s_state)
                                discovered.add((s_state, s_distance, s_time,  v, s_pred))
                            else:
                                # If the cost of the previous state in the place of the one we found is more expensive, replace it
                                # and add new state to queue
                                if v.all_states[s_state][1] > s_distance:
                                    v.all_states[s_state] = (s_state, s_distance, s_time, v,  s_pred)
                                    v.added_states.append(s_state)
                                    discovered.add((s_state, s_distance, s_time, v, s_pred))

                elif not v.visited:
                    # If the algorithm found a path with less distance cost
                    if v_distance < v.distance:
                        # Update the state's attributes
                        v.distance = v_distance
                        v.time = v_time
                        v.predecessors = u.predecessors + [u.id]
                        v.current_state = v_state
                        v.predecessors = v_pred

                        # Update the information already in the queue and set the state at its corresponding index (Example: state 3's data stored at index 3 in vertex).
                        if v.all_states[v.current_state] == None:
                            v.all_states[v.current_state] = (v.current_state, v.distance, v_time, v, v.predecessors)
                            discovered.update(v.index,(v.current_state, v.distance, v_time, v, v.predecessors))
                
                elif v.visited and v.discovered:
                    # Check for any possible new states based on the unchecked states in the added states list
                    for state_index in u.added_states:
                        state = u.all_states[state_index]
                        if state:
                            s_distance = state[1] + edge.w
                            s_time = state[2] + edge.t
                            s_state = s_time % modulus
                            s_pred = state[4] + [u.id]

                            # repeat the same logic for checking and updating possible new states 
                            if v.all_states[s_state] == None:
                                v.all_states[s_state] = (s_state, s_distance, s_time, v,  s_pred)
                                v.added_states.append(s_state)
                                discovered.add((s_state, s_distance, s_time, v,  s_pred))
                            else:
                                if v.all_states[s_state][1] > s_distance:
                                    v.all_states[s_state] = (s_state, s_distance, s_time, v,  s_pred)
                                    v.added_states.append(s_state)
                                    discovered.add((s_state, s_distance, s_time, v, s_pred))
                                
            # Reset added states at the end of each visit to a vertex
            u.added_states = []

    
def intercept(roads, stations, start, friendStart):
    """
    Determines the cheapest cost path to intercept the friend that has a specific start on the graph (the algorithm also has a start position), 
    at the shortest time possible. 
    
    Parameters
        roads: a list of edges that represent roads that are supposed to connect a group of vertices on a graph that simulates a map, each road has a time and cost.
        stations: a list of train stations and the time they take to reach the next station.
        start: an integer that represents the starting position of the algorithm.
        friendStart: an integer that represents the starting position of the friend at a train station, which is a moving target.
    Returns
        a tuple containing cost, time and list of vertices that show the optimal path that was taken.
    Complexity
        Time: O(R log L), where R is the total number of roads and L is the number of locations.
        Space: O(L + R)
        Analysis (both): as it utilizes the dijkstra algorithm which in this case has a worst case time complexity of O(R log L)
         and a worst case complexity of O(L + R), where R is the total number of roads and L is the total number of locations in the graph. 
    """

    # Find the sum of time needed for the train to loop around, in order to be used for state calculation by dijkstra.
    modulus = 0
    for (station, time) in stations:
        modulus += time

    # Find the largest vertex in both roads and stations in order to set an efficient graph size.
    largest_vertex = 0
    for road in roads:
        largest_vertex = max(largest_vertex, road[0])
    for station in stations:
        largest_vertex = max(largest_vertex, station[0])
    # To account for 0 indexing in the naming convention of locations.
    largest_vertex += 1

    # Set up the graph, add edges and run dijkstra on them
    my_graph = Graph(largest_vertex, modulus)
    my_graph.add_edges(roads)
    my_graph.dijkstra(start, modulus)

    # List of possible answers, that contains tuples of (station id, state which algorithm must reach station in to possibly intercept target)
    answers = []

    start_index = None
    # Find the index of the starting station 
    for i in range(len(stations)):
        if stations[i][0] == friendStart:
            start_index = i
            break
    
    stations_len = len(stations)
    # Loop over every station in stations, from the starting one
    for i in range(stations_len):
        # To allow us to start iterating from any position
        index = (start_index + i) % stations_len
        # Add the starting station to the list of possible answers
        if i == 0:
            answers.append((stations[index][0], 0))
        # Determine the time to reach each station after the starting one, based on the edge leading to it and the time to reach the station before it.
        else:
            answers.append((stations[index][0], stations[index-1][1] + answers[-1][1]))

    match = None

    # Look for matches based on possible answers
    for (vertex, state) in answers:
        if not my_graph.vertices[vertex].all_states[state]:
            continue
        # No prior matches means we just add
        elif match == None:
            match = my_graph.vertices[vertex].all_states[state]
        # If prior match has greater cost than current, replace it
        elif my_graph.vertices[vertex].all_states[state][1] < match[1]:
            match = my_graph.vertices[vertex].all_states[state]
        # If both matches have the same cost, choose the one with the lesser time
        elif my_graph.vertices[vertex].all_states[state][1] == match[1]:
            if my_graph.vertices[vertex].all_states[state][2] < match[2]:
                match = my_graph.vertices[vertex].all_states[state]
    
    # If you found a match, add its cost, time taken, list of path taken in a tuple and return it 
    if match:
        match = (match[1], match[2], match[4] + [match[3].id])
    return match
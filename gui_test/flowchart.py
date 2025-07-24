import random

class ChoreChart():
    def __init__(self):
        self.nodes = {}
        self.completed = {}

    def __str__(self):
        display = "Chore chart: \n"
        for task in self.top_level():
            display += str(task)
        return display + f"Points Completed: {self.point_total()}"
    
    def contains_all(self, parents, children):
        for p in parents:
            if p not in self.nodes.keys():
                print(f'Error: {p} is not in chart')
                return False
        for c in children:
            if c not in self.nodes.keys():
                print(f'Error: {c} is not in chart')
                return False
        return True
    
    def add(self, name, value, parents=set(), children=set()):
        #children is necesarry if you're adding a node to the middle
        if type(parents) == str:
            parents = {parents}
        if type(children) == str:
            children = {children}
        #check that all of parents and children exist in the graph
        if not self.contains_all(parents, children):
            return
        # Safe to add new node
        # Scenarios: no parents or children, just children, just parents, both
        if not parents and not children:
            self.nodes[name] = Node(name, value, set(), set())
        else:    
            child_nodes = {self.nodes[cname] for cname in children}
            parent_nodes = {self.nodes[pname] for pname in parents}
            new_node = Node(name, value, parent_nodes, child_nodes)
            for parent in parent_nodes:
                parent.add_child(new_node)
            for child in child_nodes:
                child.add_parent(new_node)
            self.nodes[name] = new_node
        
    def add_many(self, todolist):
        for item in todolist:
            if len(item) == 2:
                n, v = item
                self.add(n, v)
            elif len(item) == 3:
                n, v, p = item
                self.add(n, v, p)
            elif len(item) == 4:
                n, v, p, c = item
                self.add(n, v, p, c)
            else:
                print(f"Error: {item[0]} is not valid")
        print(str(self))
        
    def top_level(self):
        result = []
        for node in self.nodes.values():
            if not node.parents:
                result.append(node)
        return result
    
    def mark_done(self, node):
        #again, name or node object problem
        #if node in self.nodes.items(): #don't really need this check
        
        if node.children:
            for child_node in node.children:
                child_node.parents.remove(node)
            
        self.nodes.pop(node.name)
        self.completed[node.name] = node

    def point_total(self):
        return sum([task.value for task in self.completed])
    
    def pick_next(self):
        options = self.top_level()
        num = random.randint(0,len(options)-1)
        choice = options[num]
        display = ""
        for opt in options:
            if opt == choice:
                display += f"-> {opt.name}\n"
            else:
                display += f"   {opt.name}\n"
        print(display)
        done = input("Done? y/n  ")
        if done == "y":
            self.mark_done(opt)
            print("Good work!")
        elif done == "n":
            print("We'll get to it later")
        else:
            print("Invalid input")
    #figure out visual representation
    #def remove


class Node():
    def __init__(self, name, value, parent_nodes, child_nodes):
        self.name = name
        self.value = value
        self.parents = parent_nodes #can also be none
        self.children = child_nodes

    def add_parent(self, newparent):
        if not self.parents:
            self.parents = {newparent}
        else:
            self.parents.add(newparent)

    def add_child(self, newchild):
        if not self.children:
            self.children = {newchild}
        else:
            self.children.add(newchild)

    def __str__(self):
        display = f"{self.name} ({self.value})"
        if not self.children:
            display += "\n"
        for child in self.children:
            display += f"\t - {str(child)}"
        return display



# for pre, fill, node in RenderTree(udo):
#     print("%s%s" % (pre, node.name))

# chores today
# dishes, clean sink(parent=dishes), clean counters
# make curry (parent=clean counters, dishes), clean stove(make curry)
# pack for provence

if __name__== "__main__":
    test = ChoreChart()
    # test.add("dishes", 20)
    # test.add("clean sink", 10, parents={"dishes"})
    # test.add("laundry", 5)
    # test.add("put away clothes", 5, children={"laundry"})
    # test.add("hang laundry", 5, parents={"laundry"})
    # test.add("put away dishes", 2, parents={"dishes"})
    # test.add("relax", 0, parents={"clean sink", "hang laundry"})
    test.add_many([
        ("dishes", 15),
        ("clean sink", 10, "dishes"),
        ("put away clothes", 5, set()),
        ("clean out fridge", 8, set(), "dishes"),
        ("organize desk/tv cabinet", 5),
        ("refill meds", 3),
        ("clean stove and countertops", 5, "dishes"),
    ])
    test.pick_next()


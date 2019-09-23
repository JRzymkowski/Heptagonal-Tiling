import math

class Heptagon:
    
    def __init__(self):
        self.number = 0
        self.parent = 0
        self.center = [0,0]
        self.vertices = []
        self.code = ""
        self.layer = 0
        

# n = 7, k = 3

class Tiling:

    def __init__(self, layers):
        
        self.layers = layers
        
        self.innerPoly = 0
        self.totalPoly = 1
        self.heptagons = [[]]*10000
        self.polygons = [[]]*10000
        self.polygon_centers = [[0,0]]*10000
        self.rule = [0]*10000
        self.codes = []
        self.layer_ends = []
        self.layer_starts = []
        self.parents = [0]*10000
        self.codes = [""]*10000
        self.code_seed = ["R","R0","R1"]

        self.countPolys(self.layers)
        self.determinePolys()

        
        


    def countPolys(self, layer):
        self.totalPoly = 1
        self.innerPoly = 0
        
        self.layer_ends.append(0)
        self.layer_starts.append(0)
        layer_end = 0

        a = 0
        b = 7

        for i in range(layer):
            self.innerPoly = self.totalPoly
            self.layer_ends.append(layer_end+a+b)
            self.layer_starts.append(layer_end+1)
            layer_end = layer_end+a+b
            next_a = a + b
            next_b = a + 2*b
            self.totalPoly += a + b
            a = next_a
            b = next_b

    def determinePolys(self):
        self.polygons = [[]]*10000 #poygon vertices
        self.rule = [0]*10000 #rule array
        self.polygons[0] = self.constructCenterPolygon()
        self.polygon_centers[0] = [0,0]
        self.rule[0] = 0
        
        central = Heptagon()
        central.center = [0,0]
        central.number = 0
        central.vertices = self.constructCenterPolygon()
        central.code = self.code_seed[0]
        central.layer = 0
        
        self.heptagons[0] = central
        
        j = 1
        for i in range(self.innerPoly):
            #print(i)
            #print(self.polygons[i])
            j = self.applyRule(i,j)

    def applyRule(self, i, j):
        r = self.rule[i]
        #special = (r == 1)
        #if special:
        #    r = 2
        if r == 4:
            start = 3
        else:
            start = 2

        if r != 0:
            quantity = 6 - r
        else:
            quantity = 7 - r

        for s in range(start, start+quantity):
            self.polygons[j] = self.createNextPoly(self.polygons[i], s%7, j)
            self.heptagons[j] = self.createNextHept(self.heptagons[i], s%7, j, s-start)
            self.parents[j] = i
            
            C = [self.polygons[i][s%7], self.polygons[i][(s+1)%7]]
            self.polygon_centers[j] = new_reflect(self.polygon_centers[i], C[0], C[1])
            if s == start and r != 0:
                self.rule[j] = 4
            else:
                self.rule[j] = 3

            j += 1
            # aktualne tylko dla k != 3
            # if special:
            #     m = 2
            # elif s == 2 and r != 0:
            #     m = 1
            # else:
            #     m = 0
        return j

    def constructCenterPolygon(self):
        vertices = [[]]*7
        angleA = math.pi/7
        angleB = math.pi/3
        angleC = math.pi/2

        sinA = math.sin(angleA)
        sinB = math.sin(angleB)
        s = math.sin(angleC - angleB - angleA) / math.sqrt(1.0 - sinB * sinB - sinA * sinA)

        for i in range(7):
            vertices[i] = [s * math.cos((3+2*i)*angleA), s * math.sin((3+2*i)*angleA)]

        #print(vertices)
        return vertices

    def createNextPoly(self, vertices, s,j):
        newVertices = [[]]*7
        #print(vertices)
        #print(s)
        C = [vertices[s], vertices[(s+1)%7]]

        for i in range(7):
            #newVertices[i] = self.reflect(C, vertices[i])
            newVertices[(8+s-i)%7] = new_reflect(vertices[i], C[0], C[1])
            
        return newVertices
    
    def clockwise(self, j):
        if j in self.layer_starts:      
            i = self.layer_starts.index(j)
            return self.layer_ends[i]
        else:
            return j-1
    
    def createNextHept(self, parent, s, j, ss):
        newVertices = [[]]*7
        #print(vertices)
        #print(s)
        vertices = parent.vertices
        C = [vertices[s], vertices[(s+1)%7]]

        for i in range(7):
            #newVertices[i] = self.reflect(C, vertices[i])
            newVertices[(8+s-i)%7] = new_reflect(vertices[i], C[0], C[1])
            
        nextHept = Heptagon()
        nextHept.vertices = newVertices
        nextHept.center = new_reflect(parent.center, C[0], C[1])
        nextHept.parent = parent
        nextHept.layer = parent.layer + 1
        nextHept.number = j
    
        
        if j == 1 or j == 2:
            nextHept.code = self.code_seed[j]
        elif j < 8:
            nextHept.code = self.determine_code_by_heptagons(j-2,j-1,0)
        else:
            if j in self.layer_starts:
                i = self.layer_starts.index(j)
                nextHept.code = self.determine_code_by_heptagons(self.layer_ends[i-2], parent.number,
                                                                 self.clockwise(parent.number))
            elif ss == 1:
                nextHept.code = self.determine_code_by_heptagons(self.clockwise(parent.number), self.heptagons[j-1].parent.number,
                                                                 j-1)
#                 if j == self.layer_ends[parent.layer] + 1:
                    #print(j)
                    #print(self.layer_ends[parent.layer])
                    #print(self.heptagons[self.layer_ends[parent.layer]].parent.number)
                    #print(type(self.heptagons[self.layer_ends[parent.layer]].parent.parent))
#                 nextHept.code = self.determine_code_by_heptagons(
#                     self.heptagons[self.layer_ends[parent.layer]].parent.number, parent.number,
#                             self.layer_ends[parent.layer])
#                 else:
#                     nextHept.code = self.determine_code_by_heptagons(j-2, j-1, self.heptagons[j-1].parent.number)
            else:
                nextHept.code = self.determine_code_by_heptagons(j-2, self.heptagons[j-1].parent.number,
                                                                 j-1)
                    
        #nextHept.code = str(s)
                    
            
        
        return nextHept
    
    def determine_code_by_heptagons(self,C, A, B):
        return self.determine_code(self.heptagons[C].code, self.heptagons[A].code, self.heptagons[B].code)
        #return str(C)+","+str(A)+","+str(B)
    
    def determine_code(self, C, A, B):
        # C, A, B are codes
        
        BA = self.is_parent(B,A)
        BC = self.is_parent(B,C)
        CB = self.is_parent(C,B)
        #we always chose A as the older or the child
        if BA or BC or CB:
            A, B = B, A
        
        CA = self.is_parent(C,A)
        CB = self.is_parent(C,B)
        AB = self.is_parent(A,B)
        AC = self.is_parent(A,C)
        last = self.last
        
        
        try:
            if CA and CB:
                if last(A)+last(B) in ["01","10"]:
                    return C+"02"
                elif last(A)+last(B) in ["21","12"]:
                    return C+"20"
            if CA and not CB:
                if last(C) == "0":
                    if last(A) == "0":
                        return B+"2"
                    elif last(A) == "2":
                        return B+"0"
                elif last(C) == "2":
                    if last(A) == "0":
                        assert last(B) == "1"
                        return B+"1"
                    elif last(A) == "2":
                        return B+"0"
                elif last(C) == "1":
                    if last(A) == "0":
                        return B+"2"
                    elif last(A) == "1":
                        return B+"0"
            if AB and AC:
                if last(A) == "0":
                    #assert last(B) == "1"
                    if last(C) == "0":
                        return A+"2"
                    elif last(C) == "2":
                        return A+"0"
                    elif last(C) == "1":
                        if last(B) == "0":
                            return self.right(A)
                        elif last(B) == "2":
                            return self.left(A)
                elif last(A) == "2":
                    if last(C) == "1":
                        #assert last(B) == "0"
                        return self.right(A)+"1"
                    elif last(C) == "0":
                        #assert last(B) == "1"
                        return A+"2"
                    elif last(C) == "2":
                        #assert last(B) == "1"
                        return A+"0"
                elif last(A) == "1":
                    if last(C) == "1":
                        #assert last(B) == "0"
                        return self.prefix(A)+"02"
                    elif last(C) == "0":
                        #assert last(B) == "1"
                        return self.prefix(A)+"20"
            if AB and not AC:
                #assert last(A) in ["0","2"]
                #assert last(C) in ["0","2"]
                if last(C) == "0" and last(A) == "1":
                    return A+"0"
                elif last(C) == "2" and last(A) == "1":
                    return A+"1"
                if last(C) == "2" or last(C) == "1":
                    return A+"1"
                return C+"1"
            if AC and not AB:
                if last(A) == "0":
                    if last(C) == "0":
                        return self.right(A)
                    elif last(C) == "2":
                        return self.prefix(A)
                elif last(A) == "1":
                    if last(C) == "0":
                        if last(B) == "2":
                            return self.prefix(B)
                        else:
                            return self.prefix(A)+"0"
                    elif last(C) == "1":
                        return self.prefix(B)
                elif last(A) == "2":
                    if last(C) == "0":
                        return self.prefix(A)
                    elif last(C) == "2":
                        return self.left(self.prefix(A))
        except AssertionError:
            print("Assertion error for:",A,B,C)
                
                
        print("No result for:",A, B, C, "AB AC CA CB", AB, AC, CA, CB, "last A B C", last(A), last(B), last(C))
        return "X"
    
    def prefix(self, code):
        
        if code[-1] == "R":
            return code+"R10"
        else:
            return code[:-1]

    def last(self, code):
        if code[-1] == "R":
            return "0"
        else:
            return code[-1]
        
    def right(self, code):
        if code[-1] == "1":
            #return prefix+"02"
            return self.prefix(code)+"02"
        elif code[-1] == "0" or code[-1] == "R":
            return self.right(self.prefix(code))+"2"
        else:
            return self.prefix(code)+"1"
        
    def left(self, code):
        if code[-1] == "0" or code[-1] == "R":
            return self.prefix(code)+"1"
        else:
            return self.left(self.prefix(code))+"0"
    
    def is_parent(self, parent, child):
        return self.prefix(child) == parent
    
    

    def reflect(self, line, R):
        A = line[0]
        B = line[1]
        den = A[0]*B[1] - A[1]*B[0]
        isStraight = abs(den) < 1.e-16
        if isStraight:
            P = A
            Q = [[]]*2
            den = math.sqrt((A[0] - B[0]) * (A[0] - B[0]) + (A[1] - B[1]) * (A[1] - B[1]))
            D = [(B[0] - A[0]) / den,
                  (B[1] - A[1]) / den]

            factor = 2.0 * ((R[0] - P[0]) * D[0] + (R[1] - P[1]) * D[1])
            Q[0] = 2.0 * P[0] + factor * D[0] - R[0]
            Q[1] = 2.0 * P[1] + factor * D[1] - R[1]

        else:
            Q = [[]]*2
            s1 = (1.0 + A[0] * A[0] + A[1] * A[1]) / 2.0
            s2 = (1.0 + B[0] * B[0] + B[1] * B[1]) / 2.0
            C = [(s1 * B[1] - s2 * A[1]) / den,
                  (A[0] * s2 - B[0] * s1) / den]
            r = math.sqrt(C[0] * C[0] + C[1] * C[1] - 1.0)

            factor = r * r / ((R[0] - C[0]) * (R[0] - C[0]) + (R[1] - C[1]) * (R[1] - C[1]))
            Q[0] = C[0] + factor * (R[0] - C[0])
            Q[1] = C[1] + factor * (R[1] - C[1])

        return Q

print("AAAAA")



t = Tiling(4)
print(t.layer_starts)
#print(t.code_seed)
#print(t)
#print(t.polygons)

import matplotlib.pyplot as plt

from matplotlib.patches import Circle, PathPatch

fig, ax = plt.subplots(figsize=(6, 6))
plt.ylim((-1,1))
plt.xlim((-1,1))

circle = plt.Circle((0, 0), 1, facecolor='none',
                edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
ax.add_patch(circle)



def translate(point, vector):
    z = complex(*point)
    a = complex(*vector) 
    Ta = (z+a)/(a.conjugate()*z+1)
    return([Ta.real, Ta.imag])

translatedpoly = []
vector = [0.0,0.0]
for p in t.polygons:
    if p != []:
        newp = [translate(v, vector) for v in p]
        translatedpoly.append(newp)
translatedcent = []
        
for i, c in enumerate(t.polygon_centers):
    translatedcent.append(translate(c, vector))
    if i < 100:
        #print(t.codes[i])
        ax.annotate(str(i)+" "+t.heptagons[i].code, translate(c, vector))
        ax.annotate(str(i), translate(c, vector))
        
#pts1 = [item for sublist in translatedpoly for item in sublist]

pts1 = [item for item in translatedcent] #bez uwzględniania translacji

xs = list(map(lambda x: x[0], pts1))
ys = list(map(lambda x: x[1], pts1))


patches = []

import numpy as np


from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection



for p in translatedpoly:
    if p != []:
        #print(p)
        patches.append(Polygon(np.array(p), True))
        

#patches = patches[::-1]
#colors = 100*np.random.rand(len(patches))
#colors = [100*pow(x,0.2) for x in np.arange(0,1,1.0/len(patches))]
colors = []
for h in t.heptagons:
    if type(h) is Heptagon:
        colors.append(90.0/(h.layer+8))
p = PatchCollection(patches, alpha=1)
p.set_array(np.array(colors))
ax.add_collection(p)


plt.scatter(xs, ys,s=10, color='r')

fig = plt.gcf()
fig.set_size_inches(15, 15)  
lines = []
for p in range(1,len(translatedpoly)):
    line = np.array([translatedcent[p],translatedcent[t.parents[p]]])
    lines.append(line)
#print(len(lines))
#print(lines[:3])

from matplotlib import collections  as mc
#print(polygons)
#print(raport)
#plt.set_linewidth(0.3)
lc = mc.LineCollection(lines, color='#CC0033',linewidth=1.25)
ax.add_collection(lc)

print(t.layer_ends)

#print(t.polygons[0])
plt.show()

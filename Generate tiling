def new_reflect(R, S, T):
    disp = S[0]*T[1] - S[1]*T[0]
    if abs(disp) < 1.e-14:
        through_center = True
    else:
        through_center = False
    
    if through_center:
        P = [R[0]-S[0],R[1]-S[1]]
        Q = [T[0]-S[0],T[1]-S[1]]
        
        dotp = P[0]*Q[0]+P[1]*Q[1]
        Q2 = Q[0]*Q[0]+Q[1]*Q[1]
        
        Pprim = [2*dotp/Q2*Q[0]-P[0], 2*dotp/Q2*Q[1]-P[1]]
        return [R[0]+Pprim[0], R[1]+Pprim[1]]
    
    else:
        dist = S[0]*S[0]+S[1]*S[1]
        Sprim = [S[0]/dist, S[1]/dist]
        #kolejność S, Sprim, T
        mA = (Sprim[1]-S[1])/(Sprim[0]-S[0])
        mB = (T[1]-Sprim[1])/(T[0]-Sprim[0])
        
        O = [0,0]
        O[0] = (mA*mB*(S[1]-T[1])+mB*(S[0]+Sprim[0])-mA*(Sprim[0]+T[0]))/(2*(mB-mA))
        if abs(mA) > 0.0000000000001:
            O[1] = -1/mA*(O[0]-(S[0]+Sprim[0])/2)+(S[1]+Sprim[1])/2
        else:
            A = (Sprim[0]-S[0])/2
            B = ((Sprim[0]+S[0])/2-T[0])
            O[1] = -1*(A*A-B*B-T[1]*T[1])/(2*T[1])
        
        r = math.sqrt((O[0]-S[0])*(O[0]-S[0]) + (O[1]-S[1])*(O[1]-S[1]))
        
        den = (O[0]-R[0])*(O[0]-R[0]) + (O[1]-R[1])*(O[1]-R[1])
        x = O[0] + (r*r*(R[0]-O[0]))/den
        y = O[1] + (r*r*(R[1]-O[1]))/den
        
#         plt.scatter([S[0], Sprim[0], T[0], R[0], O[0], x, 0], [S[1], Sprim[1], T[1], R[1], O[1], y, 0],
#                     c=['r','g','b','c','y','m','k'])
#         ax.add_patch(plt.Circle((O[0],O[1]), r,facecolor='none',
#                 edgecolor=(0.7, 0.0, 0.8)))
        
        return [x,y]
        
  import math

class Heptagon:
    
    def __init__(self):
        self.number = 0
        self.father = 0
        self.center = [0,0]
        self.vertices = []
        self.code = ""
        self.layer = 0

# n = 7, k = 3

class Tiling:

    def __init__(self):
        self.innerPoly = 0
        self.totalPoly = 1
        self.heptagons = [[]]*10000
        self.polygons = [[]]*10000
        self.polygon_centers = [[0,0]]*10000
        self.rule = [0]*10000
        self.codes = []
        self.layer_ends = []
        self.parents = [0]*10000

        self.countPolys(6)
        self.determinePolys()
        
        self.code_seed = ["R","R0","R1"]


    def countPolys(self, layer):
        self.totalPoly = 1
        self.innerPoly = 0
        
        self.layer_ends.append(0)
        layer_end = 0

        a = 0
        b = 7

        for i in range(layer):
            self.innerPoly = self.totalPoly
            self.layer_ends.append(layer_end+a+b)
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
        #central.code = self.code_seed[0]
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
            self.polygons[j] = self.createNextPoly(self.polygons[i], s%7)
            #self.heptagons[j] = self.createNextHept(self.heptagons[i], s%7, j)
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

    def createNextPoly(self, vertices, s):
        newVertices = [[]]*7
        #print(vertices)
        #print(s)
        C = [vertices[s], vertices[(s+1)%7]]

        for i in range(7):
            j = (8+s-i)%7
            #newVertices[i] = self.reflect(C, vertices[i])
            newVertices[j] = new_reflect(vertices[i], C[0], C[1])

        return newVertices
    
    def createNextHept(self, parent, s, j):
        newVertices = [[]]*7
        #print(vertices)
        #print(s)
        vertices = parent.vertices
        C = [vertices[s], vertices[(s+1)%7]]

        for i in range(7):
            j = (8+s-i)%7
            #newVertices[i] = self.reflect(C, vertices[i])
            newVertices[j] = new_reflect(vertices[i], C[0], C[1])
            
        nextHept = Heptagon()
        nextHept.vertices = newVertices
        nextHept.center = new_reflect(parent.center, C[0], C[1])
        nextHept.partent = parent
        nextHept.layer = parent.layer + 1
    
        
        if j == 1 or j == 2:
            nextHept.code = self.code_seed[j]
        elif j < 8:
            nextHept.code = self.determine_code_by_heptagons(j-3,j-2,j-1)
        else:
            if s == 0:
                if j == self.layer_ends[parent.layer] + 1:
                    nextHept.code = self.determine_code_by_heptagons(
                        self.heptagons[self.layer_ends[parent.layer]].parent.number, parent.number,
                                self.layer_ends[parent.layer])
                else:
                    nextHept.code = self.determine_code_by_heptagons(
                        self.heptagons[parent.number-1].parent.number, parent.number,
                                parent.number-1)
            else:
                if s == 0:
                    pass
                else:
                    nextHept.code = self.determine_code_by_heptagons(
                        self.heptagons[parent.number-1].parent.number, parent.number,
                                j-1)
            
            

        return newVertices
    
    def determine_code_by_heptagons(i, j, k):
        return determine_code(self.heptagons[i].code, self.heptagons[j].code, self.heptagons[k].code)

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



t = Tiling()
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
vector = [0.6,-0.3]
for p in t.polygons:
    if p != []:
        newp = [translate(v, vector) for v in p]
        translatedpoly.append(newp)
translatedcent = []
        
for i, c in enumerate(t.polygon_centers):
    translatedcent.append(translate(c, vector))
    if i < 500:
        ax.annotate(i, translate(c, vector))
        
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
colors = [100*pow(x,0.2) for x in np.arange(0,1,1.0/len(patches))]
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

#print(t.polygons[0])
plt.show()

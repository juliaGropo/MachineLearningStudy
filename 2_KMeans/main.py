import matplotlib.pyplot as plt
import numpy as np

def Minkowski(point,center,lamb):
    return round((sum(pow(abs(a-b),lamb) for a,b in zip(point, center))) ** (1/float(lamb)),2)

plt.title("K-means clustering")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")

f = open("cluster.dat", "r")
text = f.readlines()
lines = len(text)

coords = np.random.rand(3, 2) * 25

print(coords)

histCenters = []

z = 0
while z < 5:
  first = []
  second = []
  third = []
  
  for i in range(0, lines):
    a = text[i].split('\n')
    xy = a[0].split(" ")
    xy = [float(xy[0]), float(xy[1])]
    distances = []
    
    for i in range(0,3):
      distances.append(Minkowski(xy, coords[i], 1))
    
    xy.append(distances.index(min(distances)))
    if(xy[2] == 0):
      first.append(xy)
    elif(xy[2] == 1):
      second.append(xy)
    else:
      third.append(xy)
  
  print(str(len(first)))
  print(str(len(second)))
  print(str(len(third)))
  
  fmt = "ok"
  fmtFinal = "ob"
  fmtDots = "+c"
  
  for i in range(0,3):
    if(i == 0):
      sumx = 0
      sumy = 0
      for j in range(0, len(first)):
        sumx += first[j][0]
        sumy += first[j][1]
        plt.plot(float(first[j][0]), float(first[j][1]), "+m")
      coords[i] = [(sumx/len(first)),(sumy/len(first))]
      print(coords[i])
      res = plt.plot(coords[i][0], coords[i][1], fmt)
    if(i == 1):
      sumx = 0
      sumy = 0
      for j in range(0, len(second)):
        sumx += second[j][0]
        sumy += second[j][1]
        plt.plot(float(second[j][0]), float(second[j][1]), "+g")
      coords[i] = [(sumx/len(second)),(sumy/len(second))]
      print(coords[i])
      res = plt.plot(coords[i][0], coords[i][1], fmt)
    if(i == 2):
      sumx = 0
      sumy = 0
      for j in range(0, len(third)):
        sumx += third[j][0]
        sumy += third[j][1]
        plt.plot(float(third[j][0]), float(third[j][1]), "+r")
      coords[i] = [(sumx/len(third)),(sumy/len(third))]
      print(coords[i])
      res = plt.plot(coords[i][0], coords[i][1], fmt)
  
    if(z == 4):
      res = plt.plot(coords[i][0], coords[i][1], fmtFinal)
  
  plt.savefig("graph" + str(z) + ".png")
  z = z + 1

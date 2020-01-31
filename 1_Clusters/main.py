import matplotlib.pyplot as plt

plt.title("My first Machine Learning graph")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")

f = open("points.dat", "r")

text = f.readlines()
lines = len(text)

d = {
  1 : "b",
  2 : "g",
  3 : "r",
  4 : "m",
  5 : "c",
  6 : "y",
  7 : "k"
}

for i in range(0, lines):
  a = text[i].split('\n')
  xy = a[0].split('\t')
  fmt = "+" + d[int(xy[2])]
  plt.plot(float(xy[0]), float(xy[1]), fmt)

plt.savefig("graph.png")